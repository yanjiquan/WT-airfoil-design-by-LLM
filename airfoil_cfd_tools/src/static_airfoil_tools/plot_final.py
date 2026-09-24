# -*- coding: utf-8 -*-
"""llm Re1.0M 静态极曲线最终图（0-30 度, 每 1 度）。"""
import json
import glob

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.unicode_minus'] = False

rows = []
for f in glob.glob('run_out/case_llm_S1p0_a*/result_summary.json'):
    try:
        d = json.load(open(f, encoding='utf-8'))
        if d.get('status') == 'done':
            rows.append((d['aoa_deg'], d['cl'], d['cd']))
    except Exception:
        pass
rows.sort()
a = np.array([r[0] for r in rows])
cl = np.array([r[1] for r in rows])
cd = np.array([r[2] for r in rows])

# Cl 线性段斜率（0-8 度附着区拟合）
lin = a <= 8
p = np.polyfit(a[lin], cl[lin], 1)
cl0, slope = float(p[1]), float(p[0])

fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))

# 1. Cl-alpha
axes[0].plot(a, cl, 'o-', color='crimson', lw=1.8, ms=5.5)
axes[0].axline((0, cl0), slope=slope, ls='--', color='0.5', lw=1,
               label='linear fit {:.3f}/deg'.format(slope))
axes[0].set_xlabel('Angle of attack (deg)')
axes[0].set_ylabel(r'$C_l$')
axes[0].set_title('Lift curve')
axes[0].grid(alpha=0.3)
axes[0].legend(fontsize=8, frameon=False)

# 2. Cd-alpha
axes[1].plot(a, cd, 's-', color='navy', lw=1.8, ms=5.5)
axes[1].set_xlabel('Angle of attack (deg)')
axes[1].set_ylabel(r'$C_d$')
axes[1].set_title('Drag curve')
axes[1].grid(alpha=0.3)

# 3. Cl-Cd polar
axes[2].plot(cd, cl, '^-', color='darkgreen', lw=1.8, ms=5.5)
axes[2].set_xlabel(r'$C_d$')
axes[2].set_ylabel(r'$C_l$')
axes[2].set_title('Drag polar')
axes[2].grid(alpha=0.3)

# 标注最大升阻比
with np.errstate(divide='ignore'):
    L_D = np.where(cd > 0, cl / cd, np.nan)
    k = int(np.nanargmax(L_D))
axes[2].plot(cd[k], cl[k], 'o', color='orange', ms=10, zorder=5)
axes[2].annotate('L/D_max={:.1f} @ {:.0f} deg'.format(L_D[k], a[k]),
                 (cd[k], cl[k]), textcoords='offset points', xytext=(10, 10),
                 fontsize=9)

for ax in axes:
    ax.tick_params(labelsize=9)

fig.suptitle('llm airfoil - static polar, Re=1.0M, transition SST (0-30 deg)',
             fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.95])
out = 'run_out/llm_Re1p0_static_polar_final.png'
fig.savefig(out, dpi=150, bbox_inches='tight')
plt.close(fig)

print('已保存:', out)
print('Cl 线性斜率(0-8 deg): {:.4f}/deg, Cl0={:.3f}'.format(slope, cl0))
print('Cl_max = {:.3f} @ {:.0f} deg'.format(cl.max(), a[int(np.argmax(cl))]))
print('L/D_max = {:.1f} @ {:.0f} deg'.format(np.nanmax(L_D), a[k]))
print('Cd_min = {:.4f} @ {:.0f} deg'.format(cd.min(), a[int(np.argmin(cd))]))
