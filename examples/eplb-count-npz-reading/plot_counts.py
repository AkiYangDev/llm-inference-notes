"""Plot the same measured slice with a common linear scale; no smoothing."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

DATA=Path(__file__).resolve().parents[2]/'assets/eplb-count-npz-reading'

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=Path('window-heatmap.png'))
    args=parser.parse_args()
    matrices=[]
    for arm in ('B_community','A_protected'):
        with np.load(DATA/f'{arm}_global_counts.npz',allow_pickle=False) as data:
            matrices.append(data['counts'][:32,0,:].copy())
    fig,axes=plt.subplots(2,1,figsize=(13,6.5),layout='constrained')
    vmax=max(int(x.max()) for x in matrices)
    for ax,values,label in zip(axes,matrices,('B: 16 target + 16 zero draft appends','A: 32 target appends'),strict=True):
        im=ax.imshow(values,aspect='auto',interpolation='none',cmap='YlOrRd',vmin=0,vmax=vmax,
                     extent=(-0.5,271.5,32.5,0.5))
        ax.set_title(label,loc='left',fontsize=12)
        ax.set_ylabel('Append order')
        ax.set_yticks([1,8,16,24,32])
        ax.set_xticks([0,32,64,96,128,160,192,224,271])
        ax.spines[['top','right']].set_visible(False)
    axes[-1].set_xlabel('Global physical expert slot (layer 0; all 16 ranks already summed)')
    fig.colorbar(im,ax=axes,label='Routes per append and physical slot',shrink=0.9)
    fig.suptitle('Measured EPLB counts: first 32 appends, shared scale, no interpolation',fontsize=14)
    fig.savefig(args.output,dpi=150)
    plt.close(fig)
    print('Saved measured heatmap with common scale and all physical slots.')
