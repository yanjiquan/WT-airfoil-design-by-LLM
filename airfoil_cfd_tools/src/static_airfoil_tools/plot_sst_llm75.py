# -*- coding: utf-8 -*-
"""llm Re0.75M 稳态SST 极曲线。"""
import json, glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.unicode_minus'] = False

rows=[]
for f in glob.glob('sst_run/case_llm_S0p75_a*/result_summary.json'):
    try:
        d=json.load(open(f,encoding='utf-8'))
        if d.get('status')=='done': rows.append((int(d['aoa_deg']),d['cl'],d['cd']))
    except Exception: pass
rows.sort()
a=np.array([r[0] for r in rows]); cl=np.array([r[1] for r in rows]); cd=np.array([r[2] for r in rows])

fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
axes[0].plot(a,cl,'o-',color='crimson',lw=1.8,ms=5)
axes[0].set_xlabel('Angle of attack (deg)'); axes[0].set_ylabel(r'$C_l$')
axes[0].set_title('Lift curve'); axes[0].grid(alpha=0.3)
axes[1].plot(a,cd,'s-',color='navy',lw=1.8,ms=5)
axes[1].set_xlabel('Angle of attack (deg)'); axes[1].set_ylabel(r'$C_d$')
axes[1].set_title('Drag curve'); axes[1].grid(alpha=0.3)
axes[2].plot(cd,cl,'^-',color='darkgreen',lw=1.8,ms=5)
axes[2].set_xlabel(r'$C_d$'); axes[2].set_ylabel(r'$C_l$')
axes[2].set_title('Drag polar'); axes[2].grid(alpha=0.3)
# 标注失速点
k=np.argmax(cl)
axes[0].plot(a[k],cl[k],'o',color='orange',ms=12,zorder=5)
axes[0].annotate('stall {:.2f} @ {:.0f} deg'.format(cl[k],a[k]),(a[k],cl[k]),
                 textcoords='offset points',xytext=(10,-15),fontsize=9)
for ax in axes: ax.tick_params(labelsize=9)
fig.suptitle('llm - static polar Re=0.75M, SST k-omega (steady)', fontsize=13)
fig.tight_layout(rect=[0,0,1,0.94])
out='sst_run/llm_Re0p75_SST.png'
fig.savefig(out,dpi=150,bbox_inches='tight')
plt.close(fig)
print('已保存:',out)
print('Cl_max=%.3f @ %d deg' % (cl.max(), a[np.argmax(cl)]))
