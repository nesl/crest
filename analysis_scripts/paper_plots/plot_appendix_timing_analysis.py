#!/usr/bin/env python3
# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Render the publication timing figure from externally supplied reconciled timing data."""
import argparse
import json
from pathlib import Path

import csv
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator


def load_panel(directory):
    """Aggregate one fixed timing panel without modifying its source files.

    Parameters
    ----------
    directory : pathlib.Path
        External directory containing ``breakdown.json``, ``proxy_times.csv``,
        and ``arena_search.json`` for four conditions with ten candidates each
        and 550 training epochs per candidate.

    Returns
    -------
    dict
        Mean host and HIL category durations in minutes and proxy times in seconds.

    Raises
    ------
    ValueError
        If panel sizes, proxy samples, or timing reconciliations are invalid.
    """
    d=json.loads((directory/'breakdown.json').read_text())['conditions']
    if set(d) != {'continuous','cadenced','flops','memory'}:
        raise ValueError('All four conditions are required')
    n=sum(v['trained_candidates'] for v in d.values())
    for condition,record in d.items():
        if record['trained_candidates'] != 10 or record['epochs'] != 550:
            raise ValueError(f'Unexpected campaign size or training budget: {condition}')
        if not math.isclose(sum(record['categories_s'].values()),record['client_process_s'],abs_tol=1e-5):
            raise ValueError(f'Timing categories do not reconcile: {condition}')
    def mean(keys):
        """Return aggregate category time per candidate in minutes.

        Parameters
        ----------
        keys : list of str
            Timing categories to combine across conditions.

        Returns
        -------
        float
            Mean minutes per trained candidate.
        """
        return sum(sum(v['categories_s'].get(k,0) for k in keys) for v in d.values())/n/60
    setup=mean(['loading','model_and_proxy_setup','other_startup_bookkeeping'])
    training=mean(['training_fit']);evaluation=mean(['post_fit_evaluation'])
    with (directory/'proxy_times.csv').open(newline='') as handle:
        rows=list(csv.DictReader(handle))
    proxy=[]
    for stage in ['proxy.flops','proxy.memory_estimate']:
        values=[float(r['inclusive_s']) for r in rows if r['stage']==stage]
        if len(values)!=n or not all(math.isfinite(v) and v>=0 for v in values):
            raise ValueError(f'{stage}: expected {n} finite samples')
        proxy.append(sum(values)/len(values))
    arena=json.loads((directory/'arena_search.json').read_text())['conditions']
    hil_segments={}
    for condition in ('continuous','cadenced'):
        record=d[condition];detail=record['hil_detail_s']
        timeout=arena[condition]['failed_probe_measurement_s']/arena[condition]['candidates']/60
        measurement=detail['runtime_protocol_measurement_wait']/10/60
        if not 0 <= timeout <= measurement:
            raise ValueError('Arena timeout time exceeds measurement category')
        parts={
            'compile_prepare':sum(detail.get(k,0) for k in ('export_stedgeai_analysis_codegen','model_export_and_backend_preparation','firmware_build_signing'))/10/60,
            'flash_boot':detail['programming_and_boot_loading']/10/60,
            'arena_timeouts':timeout,
            'measurement_handshake':measurement-timeout,
        }
        total=record['categories_s']['hil_request']/10/60
        parts['other']=total-sum(parts.values())
        if parts['other'] < -1e-8 or not math.isclose(sum(parts.values()),total,abs_tol=1e-8):
            raise ValueError(f'HIL segments do not reconcile: {condition}')
        hil_segments[condition]=parts
    return {'hil_segments_min':hil_segments,'host_min':setup+training+evaluation,'training_min':training,
            'evaluation_min':evaluation,'setup_min':setup,
            'continuous_hil_min':d['continuous']['categories_s']['hil_request']/10/60,
            'cadenced_hil_min':d['cadenced']['categories_s']['hil_request']/10/60,
            'flops_s':proxy[0],'memory_traffic_s':proxy[1]}



CAPTION = 'Mean time per candidate for the fixed OxIOD panels. Gray bars show shared GPU-host work. Colored bars show the additional continuous or 200 ms cadenced HIL cost, with totals in bold. All values are minutes. The layout shows cost attribution, not execution order.'


def main():
    """Parse external input paths and render the fixed publication figure.

    Notes
    -----
    The input root holds the STM32 panel and a ``portenta_m7`` subdirectory.
    Writes PNG, PDF, SVG, aggregated JSON values, and a caption to the explicit
    output directory. The fixed panel sizes, attribution, and rendering match
    the publication analysis.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--basename', default='appendix_timing_analysis')
    args = parser.parse_args()
    panels = {'STM32 N657': load_panel(args.data_dir),
              'Portenta H7 CM7': load_panel(args.data_dir / 'portenta_m7')}
    plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 14,
                         'axes.labelsize': 15, 'axes.titlesize': 16,
                         'axes.linewidth': 1, 'pdf.fonttype': 42, 'ps.fonttype': 42})
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    segment_colors = [colors[0], colors[1], colors[3], colors[9], colors[4]]
    fig, axes = plt.subplots(2, 1, figsize=(11.5, 6.6), sharex=True)
    for ax, (board, v) in zip(axes, panels.items()):
        host = v['host_min']
        starts = [0, v['setup_min'], v['setup_min'] + v['training_min']]
        widths = [v['setup_min'], v['training_min'], v['evaluation_min']]
        for y, start, width, shade in zip([4, 3, 2], starts, widths, ['0.72', '0.40', '0.60']):
            ax.barh(y, width, left=start, height=.54, color=shade, zorder=3)
            if y == 3:
                ax.text(start + width / 2, y, f'{width:.2f}', ha='center', va='center', color='white')
            elif y == 2:
                ax.text(start - .12, y, f'{width:.2f}', ha='right', va='center')
            else:
                ax.text(start + width + .12, y, f'{width:.2f}', ha='left', va='center')
        for y, condition in [(1, 'continuous'), (0, 'cadenced')]:
            left = host
            for width, color in zip(v['hil_segments_min'][condition].values(), segment_colors):
                ax.barh(y, width, left=left, height=.54, color=color, edgecolor='white', linewidth=.45, zorder=3)
                if width >= .45:
                    ax.text(left + width / 2, y, f'{width:.2f}', ha='center', va='center', color='white', fontsize=11.5)
                left += width
            ax.text(left + .12, y, f"+{v[condition + '_hil_min']:.2f}", va='center', ha='left', fontsize=14, fontweight='bold')
        ax.axvline(host, color='0.35', lw=1.2, ls='--', zorder=2)
        ax.text(host / 2, 4.78, 'GPU host', ha='center', va='center', color='0.25')
        ax.text(host - .12, 4.78, f'{host:.2f}', ha='right', va='center', fontsize=12, color='0.35')
        ax.text((host + 10) / 2, 4.78, 'HIL', ha='center', va='center', color='0.25')
        ax.set_title(board, pad=9)
        ax.set_yticks([4, 3, 2, 1, 0], ['Setup', 'Training', 'Post-fit evaluation', 'Continuous HIL', 'Cadenced HIL'])
        ax.tick_params(axis='y', length=0, pad=8)
        ax.set_xlim(0, 10)
        ax.set_ylim(-.55, 5.18)
        ax.xaxis.set_major_locator(MultipleLocator(1))
        ax.set_axisbelow(True)
        ax.grid(axis='x', color='0.90', linewidth=.75)
        ax.axhline(1.5, color='0.90', linewidth=.6)
    axes[-1].set_xlabel('Cumulative mean time per candidate (min)', labelpad=6)
    labels = ['Compile / prepare', 'Flash / boot', 'Arena timeouts', 'Measurement / handshake', 'Service overhead']
    legend = fig.legend(handles=[Patch(facecolor=c, label=l) for c, l in zip(segment_colors, labels)],
               loc='upper center', bbox_to_anchor=(.50, 1), ncol=5, frameon=False,
               fontsize=14, handlelength=.9, handletextpad=.35, columnspacing=.8)
    fig.subplots_adjust(left=.23, right=.985, bottom=.105, top=.87, hspace=.37)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bounds = [ax.get_tightbbox(renderer) for ax in axes]
    visible_center = (min(b.x0 for b in bounds) + max(b.x1 for b in bounds)) / 2
    center_fraction = fig.transFigure.inverted().transform((visible_center, 0))[0]
    legend.set_bbox_to_anchor((center_fraction, 1), transform=fig.transFigure)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for ext in ['png', 'pdf', 'svg']:
        fig.savefig(args.output_dir / f'{args.basename}.{ext}', dpi=400, bbox_inches='tight', pad_inches=.04)
    (args.output_dir / f'{args.basename}_values.json').write_text(json.dumps(panels, indent=2) + '\n')
    (args.output_dir / f'{args.basename}_caption.txt').write_text(CAPTION + '\n')
    plt.close(fig)


if __name__ == '__main__':
    main()
