# -*- coding: utf-8 -*-
"""llm vs naca4415 Re1.0M 静态极曲线对比图（预览版，用已完成点）。"""
import json
import glob

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.unicode_minus'] = False

OUT = 'run_out'


def load(af):
    rows = []
    for f in glob.glob(f'{OUT}/case_{af}_S1p0_a*/result_summary.json'):
        try:
            d = json.load(open(f, encoding='utf-8'))
            if d.get('status') == 'done':
                rows.append((d['aoa_deg'], d['cl'], d['cd']))
        except Exception:
            pass
    rows.sort()
    return (np.array([r[0] for r in rows]),
            np.array([r[1] for r in rows]),
            np.array([r[2] for r in rows]))


a_llm, cl_llm, cd_llm = load('llm')
a_nac, cl_nac, cd_nac = load('naca4415')

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Cl-alpha
axes[0].plot(a_llm, cl_llm, 'o-', color='crimson', lw=1.8, ms=5,
             label='llm')
axes[0].plot(a_nac, cl_nac, 's-', color='navy', lw=1.8, ms=6,
             label='naca4415')
axes[0].set_xlabel('Angle of attack (deg)')
axes[0].set_ylabel(r'$C_l$')
axes[0].set_title('Lift curves')
axes[0].grid(alpha=0.3)
axes[0].legend(fontsize=9, frameon=False)

# Cd-alpha
axes[1].plot(a_llm, cd_llm, 'o-', color='crimson', lw=1.8, ms=5,
             label='llm')
axes[1].plot(a_nac, cd_nac, 's-', color='navy', lw=1.8, ms=6,
             label='naca4415')
axes[1].set_xlabel('Angle of attack (deg)')
axes[1].set_ylabel(r'$C_d$')
axes[1].set_title('Drag curves')
axes[1].grid(alpha=0.3)
axes[1].legend(fontsize=9, frameon=False)

for ax in axes:
    ax.tick_params(labelsize=9)

fig.suptitle('Static polar comparison Re=1.0M (llm vs naca4415) '
             '[naca4415 {}/16 done]'.format(len(a_nac)), fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.94])
out = f'{OUT}/llm_vs_naca4415_Re1p0_preview.png'
fig.savefig(out, dpi=150, bbox_inches='tight')
plt.close(fig)
print('已保存:', out)
print('llm 点:', len(a_llm), ' naca4415 点:', len(a_nac))
