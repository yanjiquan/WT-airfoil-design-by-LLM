#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Beddoes-Leishman 动态失速状态空间模型（Leishman-Crouse 1989）Python 实现。

权威文献：
  [1] Leishman, J., Crouse, G., 1989. "State-space model for unsteady airfoil
      behavior and dynamic stall." 30th Structures, Structural Dynamics and
      Materials Conference, AIAA-89-1319.
  [2] Leishman, J. G., 2006. Principles of Helicopter Aerodynamics, 2nd ed.,
      Cambridge University Press, Appendix B.
  实现参考：BLcode2.0 (Pla Olea, 2021, UC Irvine) —— MATLAB 逐模块移植。

模型模块：附着流(indicial) / 失速起始(前缘压力滞后) / 后缘分离(Kirchhoff)
         / 前缘涡 / 弦向力。输出法向力 CN 与弦向力 Cc，再投影为 CL、CD。

输入：运动攻角时间序列 alpha(t) [rad] 及 dalpha/dt [rad/s]。
输出：CL, CD, CN, Cc（与攻角同步的向量）。
"""
import numpy as np


class BLModel:
    """Beddoes-Leishman 动态失速模型（升力/阻力）。"""

    def __init__(self, **kw):
        # 翼型气动与模型参数（默认 NACA0012 @ 低 Mach）
        p = dict(
            C_Nalpha=6.5,      # 法向力线斜率 [1/rad]
            alpha0=0.003,      # 零升攻角 [rad]
            Cd0=0.007,         # 零升阻力
            alpha1=0.26,       # 分离点 f=0.7 对应攻角 [rad] (~15 deg)
            dalpha1=0.04,      # alpha1 的动态偏移 [rad]
            S1=0.052,          # 失速特性系数 [rad]
            S2=0.040,          # 失速特性系数 [rad]
            K0=0.0025, K1=-0.135, K2=0.04,
            T_P=1.7,           # 前缘压力响应时间常数（非定常时间）
            T_f=3.0,           # 分离点移动时间常数
            C_N1=1.45,         # 临界法向力系数（涡起始）
            T_v=6.0,           # 涡升力时间常数
            T_vl=7.0,          # 涡传播时间常数
            D_f=8.0,           # 动态失速弦向力损失系数
            eta=0.965,         # 粘性效应因子
            M=0.05,            # Mach 数（由 Re 估算，低速）
            e=-0.25,           # 转轴相对中弦位置（1/4 弦）归一化弦长
            c=1.0,             # 弦长 [m]（归一化）
            a=340.0,           # 音速 [m/s]
        )
        p.update(kw)
        self.p = p

    # ---------------- 内部模块 ----------------

    def _attached(self, t, alpha, q):
        """附着流：indicial 响应（环量项 + 非环量项）。"""
        p = self.p
        V = p['M'] * p['a']
        N = len(alpha)
        beta = np.sqrt(1 - p['M']**2)
        A1, A2, A3, A4, A5 = 0.636, 1 - 0.636, 1.5, -0.5, 1.0
        b1, b2, b3, b4, b5 = 0.339, 0.249, 0.25, 0.1, 0.5
        kNa, kNq, kMa, kMq = 0.75, 0.75, 0.8, 0.8

        T_I = p['c'] / p['a']
        K_alpha = kNa / ((1 - p['M']) + 0.5 * p['C_Nalpha'] * beta**2 * p['M']**2 * (A1*b1 + A2*b2))
        K_q = kNq / ((1 - p['M']) + p['C_Nalpha'] * beta**2 * p['M']**2 * (A1*b1 + A2*b2))
        K_alphaM = kMa * (A3*b4 + A4*b3) / (b3 * b4 * (1 - p['M']))
        K_qM = 7 * kMq / (15*(1 - p['M']) + 1.5 * p['C_Nalpha'] * beta**2 * p['M']**2 * b5)
        T_alpha = K_alpha * T_I
        T_q = K_q * T_I

        s = 2.0 * V * t / p['c']
        X1 = np.zeros(N); X2 = np.zeros(N); X3 = np.zeros(N); X4 = np.zeros(N)
        dat = np.zeros(N); dqt = np.zeros(N)
        for i in range(1, N):
            ds = s[i] - s[i-1]
            dt = t[i] - t[i-1]
            da = alpha[i] - alpha[i-1]
            dq = q[i] - q[i-1]
            if dt > 1e-12:
                dat[i] = da / dt
                dqt[i] = dq / dt
            e1 = np.exp(-b1 * beta**2 * ds)
            e2 = np.exp(-b2 * beta**2 * ds)
            X1[i] = X1[i-1]*e1 + A1*np.exp(-b1*beta**2*ds/2) * (da + dq*(0.25 - p['e']))
            X2[i] = X2[i-1]*e2 + A2*np.exp(-b2*beta**2*ds/2) * (da + dq*(0.25 - p['e']))
            X3[i] = X3[i-1]*np.exp(-dt/T_alpha) + (dat[i]-dat[i-1])*np.exp(-dt/(2*T_alpha))
            X4[i] = X4[i-1]*np.exp(-dt/T_q) + (dqt[i]-dqt[i-1])*np.exp(-dt/(2*T_q))

        alpha_E = alpha + q*(0.25 - p['e']) - p['alpha0'] - X1 - X2
        Cnc = p['C_Nalpha'] * alpha_E
        Cni_a = 4.0 * T_alpha / p['M'] * (dat - X3)
        Cni_q = -T_q / p['M'] * (dqt - X4)
        Cni = Cni_a + Cni_q
        Cnp = Cnc + Cni
        Cc = Cnc * np.tan(alpha_E + p['alpha0'])
        return Cnp, Cni, alpha_E, Cc, s

    def _stall_onset(self, s, Cnp):
        """前缘压力滞后 → 等效临界法向力。"""
        T_P = self.p['T_P']
        N = len(s)
        Dp = np.zeros(N); Cnprime = np.zeros(N)
        for i in range(1, N):
            ds = s[i] - s[i-1]
            Ep = np.exp(-ds / T_P)
            Dp[i] = Dp[i-1]*Ep + (Cnp[i]-Cnp[i-1])*Ep**0.5
            Cnprime[i] = Cnp[i] - Dp[i]
        return Cnprime

    @staticmethod
    def _separation_point(alpha, alpha1, S1, S2):
        """Kirchhoff 分离点：|a|<=alpha1 附着为主；否则分离。"""
        a = np.abs(alpha)
        f = np.where(a <= alpha1,
                     1.0 - 0.3*np.exp((a - alpha1)/S1),
                     0.04 + 0.66*np.exp((alpha1 - a)/S2))
        return np.clip(f, 1e-12, 1 - 1e-12)

    @staticmethod
    def _vortex_time(Cnprime, s, C_N1):
        """涡传播非定常时间：Cnprime 超 C_N1 后开始计数。"""
        N = len(s)
        tauv = np.zeros(N)
        dtau = s[1] - s[0] if N > 1 else 0.0
        for i in range(1, N):
            r = np.abs(Cnprime[i]) - C_N1
            rp = np.abs(Cnprime[i-1]) - C_N1
            if r >= 0 and rp < 0:
                den = abs(Cnprime[i]) - abs(Cnprime[i-1])
                tauv[i] = (s[i]-s[i-1]) * r / (den if den != 0 else 1e-12)
            elif r >= 0 and rp >= 0:
                tauv[i] = tauv[i-1] + dtau
        return tauv

    @staticmethod
    def _sigmaf(Cnprime, C_N1, Sa, df, fpp, fr, tauv, T_vl):
        sigmafN, sigmafM = 1.0, 5.0
        if abs(Cnprime) < C_N1:
            if df <= 0:
                sigmafN, sigmafM = 1.0, 1.0
            else:
                sigmafN, sigmafM = 0.5, 5.0
        else:
            if df <= 0:
                sigmafN, sigmafM = 1.75, 1.75
            else:
                sigmafN, sigmafM = 1.0, 5.0
                if 0 < tauv <= T_vl:
                    sigmafN = 0.25
                    if Sa > 0:
                        sigmafN = 0.75
        if abs(Cnprime) > C_N1 and df <= 0:
            if Sa < 0 or fpp <= 0.7 or fr <= 0.7:
                sigmafN, sigmafM = 2.0, 2.0
        return sigmafN, sigmafM

    @staticmethod
    def _sigmav(tau_v, T_vl, Sa, df):
        sigma = 1.0
        if Sa < 0:
            sigma = 4.0
        if T_vl < tau_v <= 2*T_vl:
            sigma = 2.0 if Sa < 0 else 3.0
        elif df > 0:
            sigma = 4.0
        elif Sa < 0 and df > 0:
            sigma = 1.0
        return sigma

    def _TE_separation(self, s, Cnprime, Cni, alpha_E, alpha):
        """后缘分离：动态分离点 + 法向力/弦向力修正。"""
        p = self.p
        N = len(s)
        alpha_f = Cnprime / p['C_Nalpha'] + p['alpha0']
        tauv = self._vortex_time(Cnprime, s, p['C_N1'])

        Df = np.zeros(N); Dfr = np.zeros(N)
        fpp = np.ones(N); fr = np.ones(N); fprime = np.ones(N); fM = np.ones(N)
        for i in range(1, N):
            ds = s[i] - s[i-1]
            Sa = alpha[i] - alpha[i-1]
            Da1 = (1 - fpp[i-1])**0.25 * p['dalpha1'] if Sa < 0 else 0.0
            aalpha1 = p['alpha1'] - Da1
            fprime[i] = self._separation_point(alpha_f[i], aalpha1, p['S1'], p['S2'])
            fM[i] = fprime[i] if Sa >= 0 else self._separation_point(alpha[i], aalpha1, p['S1'], p['S2'])

            # 内层迭代（替代 MATLAB while 收敛）
            fpp_ant = 0.8 * fpp[i-1]; fr_ant = 0.8 * fr[i-1]
            for _ in range(8):
                sigN, sigM = self._sigmaf(Cnprime[i], p['C_N1'], Sa,
                                          fpp_ant - fpp[i-1], fpp_ant, fr_ant,
                                          tauv[i], p['T_vl'])
                Ef = np.exp(-sigN * ds / p['T_f'])
                Df[i] = Df[i-1]*Ef + (fprime[i]-fprime[i-1])*Ef**0.5
                fpp[i] = fprime[i] - Df[i]
                Em = np.exp(-sigM * ds / p['T_f'])
                Dfr[i] = Dfr[i-1]*Em + (fM[i] - fM[i-1])*Em**0.5
                fr[i] = fM[i] - Dfr[i]
                fpp_ant = fpp[i]; fr_ant = fr[i]

        Cnc = p['C_Nalpha'] * alpha_E
        Cnf = Cni + Cnc * ((1 + np.sqrt(fpp)) / 2.0)**2

        Phi = np.where(np.abs(Cnprime) <= p['C_N1'], 1.0,
                       fpp ** np.minimum(p['D_f']*(np.abs(Cnprime)-p['C_N1']), 1.0))
        Cc = p['eta'] * Cnc * np.tan(alpha_E + p['alpha0']) * np.sqrt(fpp) * Phi
        return Cnf, Cc, fpp

    def _vortex(self, s, fpp, alpha, alpha_E, Cnprime):
        """前缘涡升力。"""
        p = self.p
        N = len(s)
        tauv = self._vortex_time(Cnprime, s, p['C_N1'])
        Cnc = p['C_Nalpha'] * alpha_E
        Kn = (1 + np.sqrt(fpp))**2 / 4.0
        Cv = np.zeros(N); Cnv = np.zeros(N)
        for i in range(1, N):
            ds = s[i] - s[i-1]
            Sa = alpha[i] - alpha[i-1]
            df = fpp[i] - fpp[i-1]
            sig = self._sigmav(tauv[i], p['T_vl'], Sa, df)
            Ds = 0.0
            if abs(Cnprime[i]) < p['C_N1'] and df < 0:
                Ds = 1.0
            elif 0 < tauv[i] <= p['T_vl']:
                Ds = 1.0
            elif Sa > 0 and df > 0:
                Ds = 1.0
            Cv[i] = Ds * Cnc[i] * (1 - Kn[i])
            Ev = np.exp(-sig * ds / p['T_v'])
            Cnv[i] = Cnv[i-1]*Ev + Ds*(Cv[i]-Cv[i-1])*Ev**0.5
        return Cnv

    # ---------------- 主接口 ----------------

    def simulate(self, t, alpha, dalpha=None):
        """仿真 B-L 模型。
        t      : 时间 [s]
        alpha  : 攻角 [rad]
        dalpha : dα/dt [rad/s]（缺省用差分）
        返回 dict: CL, CD, CN, Cc, Cnprime, fpp, Cnv
        """
        alpha = np.asarray(alpha, dtype=float)
        t = np.asarray(t, dtype=float)
        if dalpha is None:
            dalpha = np.gradient(alpha, t)
        dalpha = np.asarray(dalpha, dtype=float)
        V = self.p['M'] * self.p['a']
        q = dalpha * self.p['c'] / V

        Cnp, Cni, alpha_E, Cc_att, s = self._attached(t, alpha, q)
        Cnprime = self._stall_onset(s, Cnp)
        Cnf, Cc, fpp = self._TE_separation(s, Cnprime, Cni, alpha_E, alpha)
        Cnv = self._vortex(s, fpp, alpha, alpha_E, Cnprime)

        CN = Cnf + Cnv
        CL = CN * np.cos(alpha) + Cc * np.sin(alpha)
        CD = CN * np.sin(alpha) - Cc * np.cos(alpha) + self.p['Cd0']
        return {'CL': CL, 'CD': CD, 'CN': CN, 'Cc': Cc,
                'Cnprime': Cnprime, 'fpp': fpp, 'Cnv': Cnv}


if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    m = BLModel()
    f = 0.6
    w = 2*np.pi*f
    t = np.linspace(0, 1/f*2, 400)
    alpha = np.deg2rad(15) + np.deg2rad(10)*np.sin(w*t)
    dalpha = np.gradient(alpha, t)
    r = m.simulate(t, alpha, dalpha)
    print("CL: mean=%.3f min=%.3f max=%.3f" % (r['CL'].mean(), r['CL'].min(), r['CL'].max()))
    print("CD: mean=%.3f min=%.3f max=%.3f" % (r['CD'].mean(), r['CD'].min(), r['CD'].max()))
    print("Cnprime 超 C_N1 点数:", (np.abs(r['Cnprime']) > m.p['C_N1']).sum())
    plt.figure(figsize=(7,5))
    plt.plot(np.rad2deg(alpha), r['CL'], lw=1.5)
    plt.xlabel('alpha (deg)'); plt.ylabel('CL')
    plt.title('B-L self-test: CL vs alpha (15+/-10 deg)')
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              '_bl_selftest.png'), dpi=120, bbox_inches='tight')
    print("自测图已保存")
