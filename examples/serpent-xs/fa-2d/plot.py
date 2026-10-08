#!/usr/bin/env python3
"""Optional report plots; matplotlib is not a required NDgpu dependency."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('report_dir', type=Path)
    args = p.parse_args()
    report = json.loads((args.report_dir/'summary.json').read_text())
    solves = sorted([s for s in report['solves'] if s['device'].startswith('cpu')],
                    key=lambda s: s['refinement'])
    refinement = solves[-1]['refinement']
    with (args.report_dir/f'r{refinement}_cpu_fusion_0.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    ref = np.array([float(r['serpent_power']) for r in rows]).reshape(17,17)
    actual = np.array([float(r['ndgpu_power']) for r in rows]).reshape(17,17)
    fuel = ref > 0
    relative = np.full_like(ref, np.nan)
    relative[fuel] = 100*(actual[fuel]/ref[fuel]-1)
    factor = fuel.sum()  # Average fuel-pin power = one.
    x, y = report['geometry']['x_edges_cm'], report['geometry']['y_edges_cm']
    fig, axes = plt.subplots(1,3, figsize=(13,4.5), layout='constrained')
    maximum = max(ref.max(), actual.max())*factor
    for ax, data, title in zip(axes[:2], (ref*factor, actual*factor),
                               ('Serpent-derived power', f'NDgpu power ({17*refinement}×{17*refinement} mesh)')):
        im = ax.pcolormesh(x,y,data,vmin=0,vmax=maximum,cmap='viridis')
        fig.colorbar(im,ax=ax,label='Power / mean fuel-pin power')
        ax.set_title(title)
    limit = np.nanmax(abs(relative))
    im = axes[2].pcolormesh(x,y,relative,vmin=-limit,vmax=limit,cmap='coolwarm')
    fig.colorbar(im,ax=axes[2],label='NDgpu − Serpent (%)')
    axes[2].set_title('Fuel-pin relative power error')
    for ax in axes:
        ax.set_aspect('equal'); ax.set_xlabel('x (cm)'); ax.set_ylabel('y (cm)')
    fig.savefig(args.report_dir/'power-comparison.png', dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1,2, figsize=(10,4), layout='constrained')
    axes[0].plot([17*s['refinement'] for s in solves], [s['k_eff'] for s in solves], 'o-', label='NDgpu')
    axes[0].axhline(report['reference']['IMP_KEFF'],color='black',label='Serpent IMP_KEFF')
    k = report['reference']['IMP_KEFF']
    err = k*report['reference']['k_relative_standard_error']
    axes[0].axhspan(k-3*err,k+3*err,alpha=.12,color='black',label='Serpent ±3 reference σ')
    axes[0].set_xlabel('Cells per axis'); axes[0].set_ylabel('k-effective'); axes[0].legend()
    g = np.arange(1,5)
    axes[1].bar(g-.18,report['reference']['normalized_spectrum'],width=.36,label='Serpent')
    axes[1].bar(g+.18,solves[-1]['normalized_spectrum'],width=.36,label='NDgpu')
    axes[1].set_xticks(g); axes[1].set_xlabel('Energy group (high → low)')
    axes[1].set_ylabel('Assembly-integrated flux fraction'); axes[1].legend()
    fig.savefig(args.report_dir/'spectrum-and-refinement.png', dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    main()
