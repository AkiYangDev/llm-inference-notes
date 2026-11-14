"""Plot a teaching bound, not a measured accelerator speedup."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=Path('speedup-bound.png'))
    args=parser.parse_args()
    speed=[1+3*i/159 for i in range(160)]
    fig,ax=plt.subplots(figsize=(9,4.8),layout='constrained')
    for p in [.1,.3,.6]:
        gain=[100*(1/(1-p+p/s)-1) for s in speed]
        ax.plot(speed,gain,label=f'Affected fraction = {p:.0%}',linewidth=2)
    ax.set(xlabel='Local speedup of the affected critical-path interval',
           ylabel='Ideal whole-forward rate gain (%)',
           title='Teaching model: local acceleration has a bounded whole-forward gain')
    ax.grid(alpha=.22)
    ax.legend()
    ax.spines[['top','right']].set_visible(False)
    fig.savefig(args.output,dpi=150)
    plt.close(fig)
    print('Saved the teaching critical-path bound.')
