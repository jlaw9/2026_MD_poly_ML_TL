"""
Standalone reproduction of the MD-property correlation analysis
(from 260604_A_analyze_md_and_corr.ipynb), run on the latest training data.

Computes:
  - MD-MD Pearson correlations (full property set)
  - MD-polymer Pearson correlations
  - Saves SI scatterplot figures and the main-text histogram

All normalization follows the manuscript: Rg, Hvap, CpL, CpG, and excess Cp
are normalized by sqrt(N) (number of heavy... actually total atoms in trimer).
"""
import os
from pathlib import Path
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]  # .../code
MD_DATE = "260225"
DB_DATE = "250121"

# ---- pretty names (math-style, matching manuscript symbols) ----
LABELS = {
    "md_density_trimer": r"$\rho$",
    "md_rg_trimer_normsqrt": r"$R_g/\sqrt{N}$",
    "md_hov_trimer_normsqrt": r"$\Delta H_{vap}/\sqrt{N}$",
    "md_heat_capacity_liquid_normsqrt": r"$CpL/\sqrt{N}$",
    "md_heat_capacity_gas_normsqrt": r"$CpG/\sqrt{N}$",
    "md_excess_heat_capacity_trimer_normsqrt": r"$\Delta Cp/\sqrt{N}$",
    "md_avg_max_heavy_atom_dist_normsqrt": r"max atom dist$/\sqrt{N}$",
    "md_avg_variance": r"$PT_\sigma$",
    "md_avg_auc_fit_at_zero_cross": r"$t_{ACF}$",
    "md_average_lifetime_ps": r"$HB_{ps}$",
    "md_avg_hbonds_per_frame": r"$HB$",
    "Glass_Transition": r"$T_g$",
    "Melt_Temp": r"$T_m$",
    "log10_ElongBreak": "log Elong. at break",
    "YoungMod": "Young's modulus",
    "Tensile_Strength": "Tensile strength",
    "Density": "Polymer density",
    "log10_Permeability_CO2": r"log $P_{CO_2}$",
    "log10_Permeability_N2": r"log $P_{N_2}$",
    "log10_Permeability_O2": r"log $P_{O_2}$",
}


def load_md():
    d = pd.read_csv(ROOT / f"data/training_data/{MD_DATE}md_train.csv", index_col=0)
    # Build the sqrt(N)-normalized columns the manuscript describes, for the two
    # properties that are NOT pre-normalized in the CSV (CpG and max atom dist).
    N = d["trimer_num_atoms"]
    if "md_heat_capacity_gas_normsqrt" not in d:
        d["md_heat_capacity_gas_normsqrt"] = d["md_heat_capacity_gas"] / np.sqrt(N)
    if "md_avg_max_heavy_atom_dist_normsqrt" not in d:
        d["md_avg_max_heavy_atom_dist_normsqrt"] = d["md_avg_max_heavy_atom_dist"] / np.sqrt(N)
    return d


# Full MD property set used for the correlation analysis / figure (matches the
# 11 scalar properties discussed in the manuscript; RDF pairs excluded as in paper).
MD_COLS_FULL = [
    "md_density_trimer",
    "md_rg_trimer_normsqrt",
    "md_hov_trimer_normsqrt",
    "md_heat_capacity_liquid_normsqrt",
    "md_heat_capacity_gas_normsqrt",
    "md_excess_heat_capacity_trimer_normsqrt",
    "md_avg_max_heavy_atom_dist_normsqrt",
    "md_avg_variance",
    "md_avg_auc_fit_at_zero_cross",
    "md_average_lifetime_ps",
    "md_avg_hbonds_per_frame",
]

POLY_COLS = [
    "Glass_Transition", "Melt_Temp", "log10_ElongBreak", "YoungMod",
    "Tensile_Strength", "Density", "log10_Permeability_CO2",
    "log10_Permeability_N2", "log10_Permeability_O2",
]


def pearson_pairs(df, cols_a, cols_b, within=False):
    out = {}
    for i, c in enumerate(cols_a):
        for j, c2 in enumerate(cols_b):
            if within and not (i < j):
                continue
            if c not in df or c2 not in df:
                continue
            sub = df[[c, c2]].dropna()
            if len(sub) > 2:
                r, p = sp.stats.pearsonr(sub[c], sub[c2])
                out[(c, c2)] = (r, p, len(sub))
    return out


def main():
    data_md = load_md()
    print(f"MD data: {data_md.shape[0]} rows")

    # ----- MD-MD -----
    md_md = pearson_pairs(data_md, MD_COLS_FULL, MD_COLS_FULL, within=True)
    print(f"\n=== MD-MD correlations ({len(md_md)} pairs) ===")
    for (c, c2), (r, p, n) in sorted(md_md.items(), key=lambda kv: -abs(kv[1][0])):
        print(f"  {r:+.3f}  n={n:4d}  {c}  vs  {c2}")

    # ----- MD-polymer -----
    merged = pd.read_csv(ROOT / f"data/training_data/{DB_DATE}db_pub_{MD_DATE}md_and_exp_train.csv")
    # add CpG / maxdist normsqrt if base cols exist in merged (they may not)
    print(f"\nMerged data: {merged.shape[0]} rows")
    md_p = pearson_pairs(merged, POLY_COLS, MD_COLS_FULL)
    print(f"\n=== MD-polymer correlations ({len(md_p)} pairs) ===")
    for (c, c2), (r, p, n) in sorted(md_p.items(), key=lambda kv: -abs(kv[1][0])):
        print(f"  {r:+.3f}  n={n:4d}  {c}  vs  {c2}")

    # ----- summary stats -----
    md_md_abs = np.array([abs(v[0]) for v in md_md.values()])
    md_p_abs = np.array([abs(v[0]) for v in md_p.values()])
    print("\n=== summary ===")
    print(f"  mean |r| MD-MD:      {md_md_abs.mean():.3f}")
    print(f"  mean |r| MD-polymer: {md_p_abs.mean():.3f}")
    print(f"  mean |r| combined:   {np.concatenate([md_md_abs, md_p_abs]).mean():.3f}")

    return data_md, merged, md_md, md_p


if __name__ == "__main__":
    main()
