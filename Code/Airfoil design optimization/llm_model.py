from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from peft import PeftModel, LoraConfig
import torch
import re
import os
import json
from peft.utils.save_and_load import set_peft_model_state_dict
import safetensors.torch
import gc

# 配置参数
MODEL_PATH = "/mnt/workspace/.cache/modelscope/hub/models/Qwen/Qwen3-8B/"
LORA_HIGH_PATH = "/mnt/workspace/demos/TRAIN_WEIGHTS/20260206/HIGH"
LORA_LOW_PATH = "/mnt/workspace/demos/TRAIN_WEIGHTS/20260206/LOW"

# 修复：使用新的内存配置环境变量
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"  # 替换旧的PYTORCH_CUDA_ALLOC_CONF
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


class LLMModel:
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not LLMModel._initialized:
            self.tokenizer = None
            self.base_model = None
            self.current_lora = None
            self.model = None
            self._load_base_model()
            LLMModel._initialized = True
    
    def _load_base_model(self):
        """加载基础模型（显存优化版）"""
        print("加载基础LLM模型中（显存优化模式）...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH, 
            trust_remote_code=True,
            local_files_only=True
        )
        
        self.base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            load_in_4bit=False,
            low_cpu_mem_usage=True,
            local_files_only=True
        )
        self.base_model.eval()
        self.model = self.base_model
        self.current_lora = None
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print(f"基础模型加载完成！GPU显存占用: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
    
    def _clean_lora(self):
        """彻底卸载当前LoRA（修复Peft警告）"""
        if self.current_lora is not None and isinstance(self.model, PeftModel):
            # 1. 卸载LoRA
            self.model = self.model.unload()
            # 2. 修复：手动清除peft_config属性，避免多adapter警告
            if hasattr(self.model, 'peft_config'):
                delattr(self.model, 'peft_config')
            # 3. 释放资源
            del self.model
            self.model = self.base_model
            self.current_lora = None
            
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            
            print(f"✅ 已卸载{self.current_lora} LoRA，显存占用: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
    
    def _load_lora_safe(self, lora_path):
        """安全加载LoRA（低显存版）"""
        if not os.path.exists(lora_path):
            raise FileNotFoundError(f"LoRA权重路径不存在: {lora_path}")
        
        config_path = os.path.join(lora_path, "adapter_config.json")
        with open(config_path, "r") as f:
            lora_config_dict = json.load(f)
        lora_config = LoraConfig(**lora_config_dict)
        
        safetensors_files = [f for f in os.listdir(lora_path) if f.endswith(".safetensors") and "adapter" in f.lower()]
        if not safetensors_files:
            raise FileNotFoundError(f"未找到LoRA权重文件: {lora_path}")
        weight_path = os.path.join(lora_path, safetensors_files[0])
        
        # 先加载到CPU，避免GPU显存峰值过高
        lora_weights = safetensors.torch.load_file(weight_path, device="cpu")
        
        # 创建PeftModel
        lora_model = PeftModel(self.base_model, lora_config)
        set_peft_model_state_dict(lora_model, lora_weights)
        lora_model.eval()
        
        # 释放CPU权重副本
        del lora_weights
        gc.collect()
        
        return lora_model
    
    def _switch_lora(self, target_type):
        """切换LoRA（极低显存占用）"""
        if target_type == self.current_lora:
            return
        
        self._clean_lora()
        
        if target_type != "BASE":
            lora_path = LORA_HIGH_PATH if target_type == "HIGH" else LORA_LOW_PATH
            print(f"🔄 加载{target_type} LoRA（低显存模式）...")
            
            self.model = self._load_lora_safe(lora_path)
            self.current_lora = target_type
            
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            print(f"✅ {target_type} LoRA加载完成！显存占用: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
        else:
            self.model = self.base_model
            self.current_lora = None
    
    def extract_attack_angle(self, prompt):
        """从prompt中提取待预测攻角"""
        pattern = r'(?<!平均)攻角为：\s*([-+]?\d*\.?\d+)°?'
        match = re.search(pattern, prompt)
        if match:
            try:
                angle = float(match.group(1))
                return angle
            except ValueError:
                return None
        return None
    
    def generate_response(self, prompt, temperature=None, max_new_tokens=8192):
        """生成模型响应（移除输入限制，输出限制默认2048 tokens，兼容传入的max_new_tokens参数）"""
        # 提取攻角
        attack_angle = self.extract_attack_angle(prompt)
        if attack_angle is not None:
            target_type = "HIGH" if attack_angle >= 14 else "LOW"
        else:
            target_type = "BASE"
        
        # 设置温度
        if temperature is None:
            temperature = 0.1 if target_type in ["HIGH", "LOW"] else 0.85
        
        # 切换LoRA
        self._switch_lora(target_type)
        
        # 移除输入截断逻辑，不再限制输入长度
        full_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.model.device)
        
        # 生成配置：使用传入的max_new_tokens（默认2048）限制输出长度
        generation_config = GenerationConfig.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True,
            max_new_tokens=max_new_tokens,  # 兼容传入的参数，默认2048
            temperature=temperature,
            do_sample=True if temperature > 0.005 else False,
            pad_token_id=self.tokenizer.eos_token_id,
            top_p=0.9,
            local_files_only=True
        )
        
        # 生成响应
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                generation_config=generation_config,
            )
        
        # 解析响应
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
        if "<|im_start|>assistant" in response:
            response = response.split("<|im_start|>assistant")[-1]
        if "<|im_end|>" in response:
            response = response.split("<|im_end|>")[0]
        
        # 清理临时张量
        del inputs, outputs
        gc.collect()
        
        return response.strip()
    
    def parse_aerodynamic_coefficients(self, response):
        """解析气动系数"""
        cl_match = re.search(r'升力系数为[:,：]\s*([-+]?\d*\.?\d+)', response)
        cd_match = re.search(r'阻力系数为[:,：]\s*([-+]?\d*\.?\d+)', response)
        
        cl = float(cl_match.group(1)) if cl_match else None
        cd = float(cd_match.group(1)) if cd_match else None
        
        return {'Cl': cl, 'Cd': cd, 'L/D': cl/cd if cl and cd and cd != 0 else None}
    
    def clear_all(self):
        """彻底清理所有资源（修复显存方法警告）"""
        self._clean_lora()
        # 释放基础模型
        if self.base_model is not None:
            del self.base_model
            self.base_model = None
        if self.model is not None:
            del self.model
            self.model = None
        # 修复：使用新的显存重置方法
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()  # 替换reset_max_memory_allocated
        print("🗑️ 所有模型资源已彻底清理")


# 测试代码
if __name__ == "__main__":
    llm = LLMModel()
    # 测试HIGH LoRA（传入max_new_tokens=200，覆盖默认值）
    test_prompt = "攻角为：15°，请计算升力系数和阻力系数"
    response = llm.generate_response(test_prompt, max_new_tokens=200, temperature=0.1)
    print(f"响应: {response}")
    
    # 测试LOW LoRA（使用默认max_new_tokens=2048）
    test_prompt2 = "攻角为：10°，请计算升力系数和阻力系数"
    response2 = llm.generate_response(test_prompt2, temperature=0.1)
    print(f"响应2: {response2}")
    
    # 清理资源
    llm.clear_all()