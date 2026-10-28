#!/usr/bin/env python3
# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Render within-pool continuous and cadence-aware selection fronts.

Supply a JSON list with study, period_ms, trial, rmse, continuous, and cadenced
fields. Energy and RMSE values come exclusively from the external input.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
os.environ.setdefault('MPLCONFIGDIR',str(Path(tempfile.gettempdir())/'crest-cs2-selection-mpl'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
BLUE='#1f77b4';ORANGE='#ff7f0e'


def front(rows, metric):
    """Return the nondominated RMSE-energy front in ascending RMSE order.

    Parameters
    ----------
    rows : list of dict
        Candidate records with RMSE and energy values.
    metric : str
        Energy field used for dominance comparisons.

    Returns
    -------
    list of dict
        Original records whose RMSE and chosen energy are nondominated.
    """
    return sorted([a for a in rows if not any(b['rmse']<=a['rmse'] and b[metric]<=a[metric] and (b['rmse']<a['rmse'] or b[metric]<a[metric]) for b in rows)],key=lambda r:r['rmse'])


def load_rows(path):
    """Validate candidate values and the six required pool-period groups.

    Parameters
    ----------
    path : pathlib.Path
        External JSON list of paired candidate observations.

    Returns
    -------
    list of dict
        Validated records, preserving original order.
    """
    rows=json.loads(path.read_text())
    groups={(s,p) for s in ('continuous','cadenced') for p in (100,200,500)}
    if not isinstance(rows,list) or not rows:
        raise ValueError('Expected a nonempty JSON list')
    seen=set()
    for row in rows:
        if not isinstance(row,dict) or not {'study','period_ms','trial','rmse','continuous','cadenced'} <= row.keys():
            raise ValueError('Incomplete candidate record')
        group=(row['study'],row['period_ms'])
        if group not in groups or not isinstance(row['trial'],int) or row['trial']<0:
            raise ValueError('Invalid candidate group or trial')
        key=(*group,row['trial'])
        if key in seen:
            raise ValueError('Duplicate candidate identity within pool and period')
        seen.add(key)
        for field in ('rmse','continuous','cadenced'):
            value=row[field]
            if not isinstance(value,(int,float)) or not math.isfinite(value) or value<0 or (field!='rmse' and value==0):
                raise ValueError('Expected finite nonnegative RMSE and positive energy')
    if {(r['study'],r['period_ms']) for r in rows} != groups:
        raise ValueError('All six pool-period groups are required')
    return rows


def main():
    """Read explicit external input and export the selection-front figure."""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    SOURCE=args.input
    OUT=args.output_dir
    OUT.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10.5,'axes.labelsize':10.5,'axes.titlesize':11,'xtick.labelsize':9.5,'ytick.labelsize':9.5,'axes.linewidth':.7,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none'})
    sizes={}

    def save(fig,stem):
        """Export one figure at the publication canvas size.

        Parameters
        ----------
        fig : matplotlib.figure.Figure
            Completed selection-front figure.
        stem : str
            Output basename without extension.
        """
        for ext in ('png','pdf','svg'):
            fig.savefig(OUT/f'{stem}.{ext}',dpi=400,facecolor='white')
        sizes[stem]={'inches':list(fig.get_size_inches()),'png_pixels':[round(v*400) for v in fig.get_size_inches()]}
        plt.close(fig)
    rows=load_rows(SOURCE)
    fig,axes=plt.subplots(2,3,figsize=(7.05,4.2))
    for row,pool in enumerate(('continuous','cadenced')):
     for col,period in enumerate((100,200,500)):
        ax=axes[row,col];group=[r for r in rows if r['study']==pool and r['period_ms']==period]
        ax.scatter([r['rmse'] for r in group],[r['cadenced'] for r in group],s=8,c='#999999',alpha=.25,linewidths=0,zorder=1)
        for metric,color in [('continuous',BLUE),('cadenced',ORANGE)]:
          f=front(group,metric);ax.plot([r['rmse'] for r in f],[r['cadenced'] for r in f],color=color,marker='o',ms=2.7,lw=1.0,zorder=3)
        ax.set_title(f'({chr(97+row*3+col)}) {period} ms',pad=4)
        ax.set_xlabel('RMSE' if row == 1 else '',labelpad=2);ax.grid(color='#dddddd',linewidth=.4,zorder=0)
        ax.tick_params(length=3,pad=2)
        if col==0:ax.set_ylabel('Cadenced energy\n(mJ / release)',labelpad=3)
    fig.legend(handles=[Line2D([],[],color=BLUE,marker='o',ms=3,lw=1,label='Continuous-energy front'),Line2D([],[],color=ORANGE,marker='o',ms=3,lw=1,label='Cadence-aware front')],loc='upper center',bbox_to_anchor=(.53,1),ncol=2,frameon=False,fontsize=9.5,columnspacing=1,handlelength=1.4)
    fig.text(.55,.918,'Continuous-search pool',ha='center',va='center',fontsize=9.5)
    fig.text(.55,.475,'Cadenced-search pool',ha='center',va='center',fontsize=9.5)
    fig.subplots_adjust(left=.1,right=.99,bottom=.105,top=.85,wspace=.34,hspace=.55)
    save(fig,'case2_selection_fronts')
    (OUT/'figure_manifest.json').write_text(json.dumps({'input_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'figures':sizes,'polylines':'Guide the eye only; no interpolation in selection analysis.'},indent=2)+'\n')


if __name__ == '__main__':
    main()
