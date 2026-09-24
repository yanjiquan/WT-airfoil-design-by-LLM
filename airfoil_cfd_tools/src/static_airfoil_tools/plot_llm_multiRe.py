# -*- coding: utf-8 -*-
"""llm 四 Re 汇总图: Cl-alpha, Cd-alpha 多工况 + 平均线。"""
import json, glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.unicode_minus'] = False

RE_TAGS = ['0p75', '1p0', '1p25', '1p5']
RE_MAG = {'0p75': 0.75, '1p0': 1.00, '1p25': 1.25, '1p5': 1.50}
COLS = {'0p75': '#1f77b4', '1p0': '#ff7f0e', '1p25': '#2ca02c', '1p5': '#d62728'}
rows = {r: [] for r in RE_TAGS}
for f in glob.glob('sst_run/case_llm_S*/result_summary.json'):
    try:
        d = json.load(open(f, encoding='utf-8'))
        if d.get('status') == 'done':
            rows[d['re_tag']].append((int(d['aoa_deg']), d['cl'], d['cd']))
    except Exception: pass

fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
for r in RE_TAGS:
    d = sorted(rows[r])
    a = np.array([x[0] for x in d]); cl = np.array([x[1] for x in d])
    cd = np.array([x[2] for x in d])
    axes[0].plot(a, cl, 'o-', color=COLS[r], lw=1.6, ms=4.5,
                 label='Re={:.2f}M'.format(RE_MAG[r]))
    axes[1].plot(a, cd, 's-', color=COLS[r], lw=1.6, ms=4.5,
                 label='Re={:.2f}M'.format(RE_MAG[r]))

# 平均线
all_a = sorted(set(a for r in rows.values() for a, _, _ in r))
mcl = np.mean([[dict((x[0],x[1]) for x in rows[r]).get(a,np.nan) for a in all_a] for r in RE_TAGS],axis=0)
mcd = np.mean([[dict((x[0],x[2]) for x in rows[r]).get(a,np.nan) for a in all_a] for r in RE_TAGS],axis=0)
axes[0].plot(all_a, mcl, 'k--', lw=2.2, label='4-Re mean')
axes[1].plot(all_a, mcd, 'k--', lw=2.2, label='4-Re mean')

axes[0].set_xlabel('Angle of attack (deg)'); axes[0].set_ylabel(r'$C_l$')
axes[0].set_title('Lift curves - llm'); axes[0].grid(alpha=0.3)
axes[0].legend(fontsize=8, frameon=False)
axes[1].set_xlabel('Angle of attack (deg)'); axes[1].set_ylabel(r'$C_d$')
axes[1].set_title('Drag curves - llm'); axes[1].grid(alpha=0.3)
axes[1].legend(fontsize=8, frameon=False)
for ax in axes: ax.tick_params(labelsize=9)
fig.suptitle('llm airfoil - static polar (SST k-omega), 4 Reynolds numbers', fontsize=13)
fig.tight_layout(rect=[0,0,1,0.94])
out='sst_run/llm_multiRe_summary.png'
fig.savefig(out,dpi=150,bbox_inches='tight')
plt.close(fig)
print('已保存:',out)
