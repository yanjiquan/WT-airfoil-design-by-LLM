"""气动预测 LLM 客户端（双后端）。

后端选择（环境变量 ``LLM_BACKEND``，默认 ``vllm``）：
- ``vllm``（默认）：连接远程 vLLM OpenAI 兼容服务，按攻角自动选择 high/low LoRA
  适配器，以非思考模式（enable_thinking=false）生成，内置失败重试。
- ``local``：本地 transformers + peft 推理（保留原实现，需服务器上有模型与 LoRA 权重）。

配置来源（优先级从高到低）：
1. 环境变量：``VLLM_BASE_URL`` / ``VLLM_API_KEY`` / ``VLLM_MODEL_HIGH`` /
   ``VLLM_MODEL_LOW`` / ``VLLM_TIMEOUT`` / ``VLLM_RETRIES``
2. ``src/config.json`` 的 ``vllm`` 段

对外接口保持兼容（供 aerodynamic_prediction.py / airfoil_optimizer.py 调用）：
- ``generate_response(prompt, temperature=None, max_new_tokens=8192) -> str``
- ``parse_aerodynamic_coefficients(response) -> dict``
- ``extract_attack_angle(prompt)``
- ``clear_all()``
"""
import json
import os
import re
import time
import urllib.error
import urllib.request

# local 后端专用（仅 LLM_BACKEND=local 时使用）
MODEL_PATH = "/mnt/workspace/.cache/modelscope/hub/models/Qwen--Qwen3-8B/snapshots/master"
LORA_HIGH_PATH = "/mnt/workspace/LLM/WEIGHTS/HIGH"
LORA_LOW_PATH = "/mnt/workspace/LLM/WEIGHTS/LOW"

# 攻角阈值：≥14° 使用 high 适配器（高攻角段），否则 low
ANGLE_THRESHOLD_HIGH = 14.0


class LLMModel:
    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not LLMModel._initialized:
            self.backend = os.environ.get("LLM_BACKEND", "vllm").strip().lower()
            if self.backend == "local":
                self._init_local_backend()
            else:
                self._init_vllm_backend()
            LLMModel._initialized = True

    # ================= vllm 后端 =================
    def _init_vllm_backend(self):
        """加载 vLLM 配置（环境变量优先，其次 config.json 的 vllm 段）。"""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        cfg = {}
        try:
            with open(os.path.join(script_dir, "config.json"), encoding="utf-8") as f:
                cfg = json.load(f).get("vllm", {})
        except (OSError, ValueError):
            cfg = {}
        self.vllm_base_url = (
            os.environ.get("VLLM_BASE_URL") or cfg.get("base_url") or "http://127.0.0.1:8000/v1"
        ).rstrip("/")
        self.vllm_api_key = os.environ.get("VLLM_API_KEY") or cfg.get("api_key") or "EMPTY"
        self.vllm_model_high = os.environ.get("VLLM_MODEL_HIGH") or cfg.get("model_high") or "high"
        self.vllm_model_low = os.environ.get("VLLM_MODEL_LOW") or cfg.get("model_low") or "low"
        self.vllm_model_base = os.environ.get("VLLM_MODEL_BASE") or cfg.get("model_base") or "base"
        self.vllm_timeout = float(os.environ.get("VLLM_TIMEOUT") or cfg.get("timeout") or 120.0)
        self.vllm_retries = int(os.environ.get("VLLM_RETRIES") or cfg.get("retries") or 3)
        print(f"[LLM] 后端=vLLM API  {self.vllm_base_url}  "
              f"适配器 high={self.vllm_model_high} / low={self.vllm_model_low} / base={self.vllm_model_base}")

    def _select_model(self, attack_angle):
        """按任务选择模型：
        - 攻角≥14° → high 适配器（高攻角气动预测）
        - 攻角<14° → low 适配器（低攻角气动预测）
        - 无攻角（LLM 优化引导）→ base 基础模型，不加载气动 LoRA，
          避免 LoRA 将"生成下一代参数"任务拉偏。
        """
        if attack_angle is None:
            return self.vllm_model_base
        return self.vllm_model_high if attack_angle >= ANGLE_THRESHOLD_HIGH else self.vllm_model_low

    def _chat_once(self, model_name, prompt, temperature, max_tokens, enable_thinking=False):
        """单次 chat 请求，返回正文内容。"""
        body = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "chat_template_kwargs": {"enable_thinking": enable_thinking},
        }
        req = urllib.request.Request(
            f"{self.vllm_base_url}/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.vllm_api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.vllm_timeout) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")
            raise ConnectionError(f"vLLM HTTP {e.code}: {detail}") from e
        message = payload["choices"][0]["message"]
        content = message.get("content") or ""
        reasoning = message.get("reasoning_content") or ""
        if reasoning:
            # Qwen3 thinking 内容在独立字段，拼到正文前便于记录/排查；解析层会剥离 <think> 块
            return f"<think>\n{reasoning}\n</think>\n\n{content}"
        return content

    def _generate_vllm(self, prompt, attack_angle, temperature, max_new_tokens):
        """带重试的 vLLM 生成（线性退避）。

        全程开启 thinking 模式：LoRA 权重在思考模式下训练，开启 thinking
        的预测与训练分布一致，输出更可靠。解析层负责剥离 <think> 块。
        """
        model_name = self._select_model(attack_angle)
        if temperature is None:
            temperature = 0.0
        enable_thinking = True
        if os.environ.get("VLLM_DEBUG"):
            print(f"[LLM] 请求 model={model_name} enable_thinking={enable_thinking} "
                  f"max_new_tokens={max_new_tokens}")
        last_err = None
        for attempt in range(1, self.vllm_retries + 1):
            try:
                return self._chat_once(model_name, prompt, temperature,
                                       max_new_tokens, enable_thinking)
            except Exception as e:  # noqa: BLE001 网络/服务异常统一重试
                # 参数/长度类错误是确定性的，重试无意义，直接抛出并给出修复提示
                err_text = str(e)
                if "context length" in err_text or "maximum context" in err_text.lower():
                    raise ConnectionError(
                        "vLLM 请求超过上下文长度上限（max_model_len）。"
                        "请降低 max_new_tokens 或精简 prompt 历史。原始错误：\n" + err_text
                    ) from e
                last_err = e
                if attempt < self.vllm_retries:
                    print(f"[WARN] vLLM 请求失败（第{attempt}/{self.vllm_retries}次重试）：{e}")
                    time.sleep(1.0 * attempt)
        raise ConnectionError(
            f"vLLM API 请求失败（已重试 {self.vllm_retries} 次，model={model_name}）：{last_err}"
        )

    # ================= local 后端（保留原实现，延迟导入） =================
    def _init_local_backend(self):
        """加载基础模型（显存优化版）。"""
        import gc
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        print("[LLM] 后端=本地推理（transformers + peft）")
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH, trust_remote_code=True, local_files_only=True
        )
        self.base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            load_in_4bit=False,
            low_cpu_mem_usage=True,
            local_files_only=True,
        )
        self.base_model.eval()
        self.model = self.base_model
        self.current_lora = None

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        mem = (f"GPU显存占用: {torch.cuda.memory_allocated()/1024**3:.2f} GB"
               if torch.cuda.is_available() else "（无GPU）")
        print(f"[OK] 基础模型加载完成！{mem}")

    def _clean_lora(self):
        """彻底卸载当前LoRA（local 后端）。"""
        import gc
        import torch
        from peft import PeftModel

        if self.current_lora is not None and isinstance(self.model, PeftModel):
            self.model = self.model.unload()
            if hasattr(self.model, 'peft_config'):
                delattr(self.model, 'peft_config')
            del self.model
            self.model = self.base_model
            self.current_lora = None

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            mem = (f"显存占用: {torch.cuda.memory_allocated()/1024**3:.2f} GB"
                   if torch.cuda.is_available() else "（无GPU）")
            print(f"[OK] 已卸载 LoRA，{mem}")

    def _load_lora_safe(self, lora_path):
        """安全加载LoRA（local 后端，低显存）。"""
        import gc
        import safetensors.torch
        from peft import LoraConfig, PeftModel
        from peft.utils.save_and_load import set_peft_model_state_dict

        if not os.path.exists(lora_path):
            raise FileNotFoundError(f"LoRA权重路径不存在: {lora_path}")
        with open(os.path.join(lora_path, "adapter_config.json"), "r") as f:
            lora_config = LoraConfig(**json.load(f))
        safetensors_files = [f for f in os.listdir(lora_path)
                             if f.endswith(".safetensors") and "adapter" in f.lower()]
        if not safetensors_files:
            raise FileNotFoundError(f"未找到LoRA权重文件: {lora_path}")
        weight_path = os.path.join(lora_path, safetensors_files[0])

        # 先加载到 CPU，避免 GPU 显存峰值过高
        lora_weights = safetensors.torch.load_file(weight_path, device="cpu")
        lora_model = PeftModel(self.base_model, lora_config)
        set_peft_model_state_dict(lora_model, lora_weights)
        lora_model.eval()

        del lora_weights
        gc.collect()
        return lora_model

    def _switch_lora(self, target_type):
        """切换LoRA（local 后端，极低显存占用）。"""
        import gc
        import torch

        if target_type == self.current_lora:
            return
        self._clean_lora()

        if target_type != "BASE":
            lora_path = LORA_HIGH_PATH if target_type == "HIGH" else LORA_LOW_PATH
            print(f"[LLM] 加载{target_type} LoRA（低显存模式）...")
            self.model = self._load_lora_safe(lora_path)
            self.current_lora = target_type

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            mem = (f"显存占用: {torch.cuda.memory_allocated()/1024**3:.2f} GB"
                   if torch.cuda.is_available() else "（无GPU）")
            print(f"[OK] {target_type} LoRA加载完成！{mem}")
        else:
            self.model = self.base_model
            self.current_lora = None

    def _generate_local(self, prompt, attack_angle, temperature, max_new_tokens):
        """本地推理生成（local 后端）。"""
        import gc
        import torch
        from transformers import GenerationConfig

        if attack_angle is not None:
            target_type = "HIGH" if attack_angle >= ANGLE_THRESHOLD_HIGH else "LOW"
        else:
            target_type = "BASE"
        if temperature is None:
            temperature = 0.0   # 预测器与优化器统一贪心解码
        self._switch_lora(target_type)

        full_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.model.device)

        generation_config = GenerationConfig.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True if temperature > 0.005 else False,
            pad_token_id=self.tokenizer.eos_token_id,
            top_p=0.9,
            local_files_only=True,
        )

        with torch.no_grad():
            outputs = self.model.generate(**inputs, generation_config=generation_config)

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
        if "<|im_start|>assistant" in response:
            response = response.split("<|im_start|>assistant")[-1]
        if "<|im_end|>" in response:
            response = response.split("<|im_end|>")[0]

        del inputs, outputs
        gc.collect()
        return response.strip()

    # ================= 公共接口 =================
    def extract_attack_angle(self, prompt):
        """从 prompt 中提取待预测攻角。"""
        pattern = r'(?<!平均)攻角为：\s*([-+]?\d*\.?\d+)°?'
        match = re.search(pattern, prompt)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                return None
        return None

    def generate_response(self, prompt, temperature=None, max_new_tokens=8192):
        """生成模型响应，按后端分发。"""
        attack_angle = self.extract_attack_angle(prompt)
        if self.backend == "local":
            return self._generate_local(prompt, attack_angle, temperature, max_new_tokens)
        return self._generate_vllm(prompt, attack_angle, temperature, max_new_tokens)

    def parse_aerodynamic_coefficients(self, response):
        """解析气动系数（先剥离思考内容，避免 think 块中的数值干扰解析）。"""
        response = re.sub(r'<think>.*?</think>', '', response or '', flags=re.DOTALL)
        cl_match = re.search(r'升力系数为[:,：]\s*([-+]?\d*\.?\d+)', response)
        cd_match = re.search(r'阻力系数为[:,：]\s*([-+]?\d*\.?\d+)', response)
        cl = float(cl_match.group(1)) if cl_match else None
        cd = float(cd_match.group(1)) if cd_match else None
        return {'Cl': cl, 'Cd': cd, 'L/D': cl / cd if cl and cd and cd != 0 else None}

    def clear_all(self):
        """释放所有资源（vllm 后端无本地资源）。"""
        if self.backend == "local":
            self._clean_lora()
            if self.base_model is not None:
                del self.base_model
                self.base_model = None
            if self.model is not None:
                del self.model
                self.model = None
            import gc
            import torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            print("[OK] 本地模型资源已清理")


# 测试代码
if __name__ == "__main__":
    llm = LLMModel()
    test_prompt = "攻角为：15°，请计算升力系数和阻力系数"
    response = llm.generate_response(test_prompt, max_new_tokens=200, temperature=0.0)
    print(f"响应: {response}")
    print(f"解析: {llm.parse_aerodynamic_coefficients(response)}")

    test_prompt2 = "攻角为：10°，请计算升力系数和阻力系数"
    response2 = llm.generate_response(test_prompt2, max_new_tokens=200, temperature=0.0)
    print(f"响应2: {response2}")
    print(f"解析2: {llm.parse_aerodynamic_coefficients(response2)}")
    llm.clear_all()
