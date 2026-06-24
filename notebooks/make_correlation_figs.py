"""
Final correlation analysis + figure/table generation for the MD-ML manuscript.

Normalization basis: divide by sqrt(N), N = number of heavy atoms in the trimer
(consistent with the training data and the manuscript text). Run on latest data
(260225 MD, 250121 polymer DB).

Outputs (written to overleaf_paper/graphics/ and code/notebooks/):
  - 260622_fig3_overview_corrs.png : main Fig 3 (panel A schematic + new panel B histogram)
  - panelB_histogram.png           : standalone panel B (for inspection)
  - SI_norm_size_corr.png          : raw vs /sqrtN |r| with trimer size (normalization motivation)
  - SI_corr_scatters.png           : scatterplots of the strongest correlations
  - norm_size_table.tex            : LaTeX table of raw vs /sqrtN size correlations
  - correlation_values.txt         : all numbers, for updating the text
"""
import numpy as np
import pandas as pd
import scipy as sp
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import seaborn as sns
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]            # .../code
GFX = ROOT.parent / "overleaf_paper" / "graphics"     # .../overleaf_paper/graphics
OUT = Path(__file__).resolve().parent                 # .../code/notebooks
MD_DATE, DB_DATE = "260225", "250121"

BLUE, PINK = "#5ac4e2", "#e196d1"

LABELS = {
    "md_density_trimer": r"$\rho$",
    "md_rg_trimer_normsqrt": r"$R_g$",
    "md_hov_trimer_normsqrt": r"$\Delta H_{vap}$",
    "md_heat_capacity_liquid_normsqrt": r"$CpL$",
    "md_heat_capacity_gas_normsqrt": r"$CpG$",
    "md_excess_heat_capacity_trimer_normsqrt": r"$\Delta Cp$",
    "md_avg_max_heavy_atom_dist_normsqrt": "max atom dist",
    "md_avg_variance": r"$PT_\sigma$",
    "md_avg_auc_fit_at_zero_cross": r"$t_{ACF}$",
    "md_average_lifetime_ps": r"$HB_{ps}$",
    "md_avg_hbonds_per_frame": r"$HB$",
    "Glass_Transition": r"$T_g$",
    "Melt_Temp": r"$T_m$",
    "log10_ElongBreak": "log Elong.",
    "YoungMod": "Young's mod.",
    "Tensile_Strength": "Tensile str.",
    "Density": "Polymer $\\rho$",
    "log10_Permeability_CO2": r"log $P_{CO_2}$",
    "log10_Permeability_N2": r"log $P_{N_2}$",
    "log10_Permeability_O2": r"log $P_{O_2}$",
}

MD_COLS = [
    "md_density_trimer", "md_rg_trimer_normsqrt", "md_hov_trimer_normsqrt",
    "md_heat_capacity_liquid_normsqrt", "md_heat_capacity_gas_normsqrt",
    "md_excess_heat_capacity_trimer_normsqrt", "md_avg_max_heavy_atom_dist_normsqrt",
    "md_avg_variance", "md_avg_auc_fit_at_zero_cross", "md_average_lifetime_ps",
    "md_avg_hbonds_per_frame",
]
POLY_COLS = [
    "Glass_Transition", "Melt_Temp", "log10_ElongBreak", "YoungMod",
    "Tensile_Strength", "Density", "log10_Permeability_CO2",
    "log10_Permeability_N2", "log10_Permeability_O2",
]

# raw base columns -> normalized-by-sqrtN column, for the size-motivation table
NORM_MAP = [
    ("md_rg_trimer", "md_rg_trimer_normsqrt", r"$R_g$"),
    ("md_hov_trimer", "md_hov_trimer_normsqrt", r"$\Delta H_{vap}$"),
    ("md_heat_capacity_liquid", "md_heat_capacity_liquid_normsqrt", r"$CpL$"),
    ("md_heat_capacity_gas", "md_heat_capacity_gas_normsqrt", r"$CpG$"),
    ("md_excess_heat_capacity_trimer", "md_excess_heat_capacity_trimer_normsqrt", r"$\Delta Cp$"),
    ("md_avg_max_heavy_atom_dist", "md_avg_max_heavy_atom_dist_normsqrt", "max atom dist"),
]


def load_md():
    d = pd.read_csv(ROOT / f"data/training_data/{MD_DATE}md_train.csv", index_col=0)
    N = d["trimer_num_atoms"]
    d["md_heat_capacity_gas_normsqrt"] = d["md_heat_capacity_gas"] / np.sqrt(N)
    d["md_avg_max_heavy_atom_dist_normsqrt"] = d["md_avg_max_heavy_atom_dist"] / np.sqrt(N)
    return d


def r_of(df, a, b):
    s = df[[a, b]].dropna()
    if len(s) < 3:
        return np.nan, 0
    return sp.stats.pearsonr(s[a], s[b])[0], len(s)


def main():
    d = load_md()
    merged = pd.read_csv(ROOT / f"data/training_data/{DB_DATE}db_pub_{MD_DATE}md_and_exp_train.csv")
    log = []
    def P(*a):
        line = " ".join(str(x) for x in a)
        print(line); log.append(line)

    # ---------- MD-MD ----------
    md_md = {}
    for i, c in enumerate(MD_COLS):
        for j, c2 in enumerate(MD_COLS):
            if i < j:
                r, n = r_of(d, c, c2)
                if n:
                    md_md[(c, c2)] = r
    # ---------- MD-polymer ----------
    md_p = {}
    for c in POLY_COLS:
        for c2 in MD_COLS:
            if c2 in merged.columns:
                r, n = r_of(merged, c, c2)
                if n:
                    md_p[(c, c2)] = r

    P("=== STRONGEST MD-MD (|r|>=0.5) ===")
    for (a, b), r in sorted(md_md.items(), key=lambda kv: -abs(kv[1])):
        if abs(r) >= 0.5:
            P(f"  {r:+.3f}  {LABELS[a]} vs {LABELS[b]}  [{a} | {b}]")
    P("=== STRONGEST MD-polymer (|r|>=0.45) ===")
    for (a, b), r in sorted(md_p.items(), key=lambda kv: -abs(kv[1])):
        if abs(r) >= 0.45:
            P(f"  {r:+.3f}  {LABELS[a]} vs {LABELS[b]}  [{a} | {b}]")

    mdmd_abs = np.abs(list(md_md.values()))
    mdp_abs = np.abs(list(md_p.values()))
    P(f"\nmean |r| MD-MD={mdmd_abs.mean():.3f}  MD-poly={mdp_abs.mean():.3f}  "
      f"combined={np.concatenate([mdmd_abs, mdp_abs]).mean():.3f}")

    # ---------- size-motivation table ----------
    N = d["trimer_num_atoms"]
    P("\n=== RAW vs /sqrtN correlation with trimer size N ===")
    rows = []
    for base, norm, lab in NORM_MAP:
        r_raw, _ = r_of(d.assign(_x=d[base], _N=N), "_x", "_N")
        r_nrm, _ = r_of(d.assign(_x=d[norm], _N=N), "_x", "_N")
        rows.append((lab, r_raw, r_nrm))
        P(f"  {lab:16s} raw={r_raw:+.3f}  /sqrtN={r_nrm:+.3f}")

    # ---------- LaTeX table ----------
    tex = [
        r"\begin{table}[h]", r"\centering",
        r"\caption{Pearson correlation of MD-derived properties with trimer size "
        r"(number of heavy atoms, $N$), before and after normalization by $\sqrt{N}$. "
        r"Length-scale properties decorrelate fully; the heat capacities, being "
        r"extensive, retain partial size dependence.}",
        r"\label{tab:norm_size}",
        r"\begin{tabular}{lcc}", r"\hline",
        r"MD property & raw $r$ with $N$ & $r$ with $N$ after $/\sqrt{N}$ \\", r"\hline",
    ]
    for lab, rr, rn in rows:
        tex.append(f"{lab} & {rr:+.2f} & {rn:+.2f} \\\\")
    tex += [r"\hline", r"\end{tabular}", r"\end{table}"]
    (OUT / "norm_size_table.tex").write_text("\n".join(tex))
    P("\nwrote norm_size_table.tex")

    # ================= FIGURES =================
    sns.set_context("notebook")

    # ----- Panel B histogram -----
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    bins = np.arange(0, 1.01, 0.05)
    ax.hist(mdp_abs, bins=bins, color=BLUE, alpha=0.7, label="MD - Polymer")
    ax.hist(mdmd_abs, bins=bins, color=PINK, alpha=0.7, label="MD - MD")
    ax.legend(frameon=False, fontsize=12)
    ax.set_xlabel("Pearson correlation coefficient")
    ax.set_ylabel("Frequency")
    ax.spines[["right", "top"]].set_visible(False)
    # annotations for the tail bars, stacked at the upper right with leader lines
    ymax = ax.get_ylim()[1]
    def annot(txt, x, ty):
        ax.annotate(txt, xy=(x, 0.7), xytext=(0.99, ty), fontsize=10, ha="right",
                    va="center", arrowprops=dict(arrowstyle="-", lw=0.7, color="0.3"))
    annot(r"$CpL$ & $CpG$ ($r$=0.83)", 0.83, ymax * 0.78)
    annot(r"Density ($r$=0.89)", 0.886, ymax * 0.68)
    annot(r"$R_g$ & max atom dist ($r$=0.96)", 0.96, ymax * 0.58)
    ax.text(-0.06, 1.02, "B", transform=ax.transAxes, fontsize=20, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "panelB_histogram.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    P("wrote panelB_histogram.png")

    # ----- composite with panel A (crop from existing fig) -----
    old = Image.open(GFX / "260511_fig3_overview_corrs.png").convert("RGBA")
    W, H = old.size
    panelA = old.crop((0, 0, int(W * 0.515), H))         # left half = schematic
    newB = Image.open(OUT / "panelB_histogram.png").convert("RGBA")
    # scale B to panel height
    scaleB = H / newB.height
    newB = newB.resize((int(newB.width * scaleB), H))
    comp = Image.new("RGBA", (panelA.width + newB.width, H), (255, 255, 255, 255))
    comp.paste(panelA, (0, 0), panelA)
    comp.paste(newB, (panelA.width, 0), newB)
    comp.convert("RGB").save(GFX / "260622_fig3_overview_corrs.png", dpi=(200, 200))
    P("wrote 260622_fig3_overview_corrs.png", comp.size)

    # ----- SI normalization-motivation bar chart -----
    fig, ax = plt.subplots(figsize=(7, 4))
    labs = [r[0] for r in rows]
    raw = [abs(r[1]) for r in rows]
    nrm = [abs(r[2]) for r in rows]
    x = np.arange(len(labs)); w = 0.38
    ax.bar(x - w/2, raw, w, color="#b0b0b0", label="raw (unnormalized)")
    ax.bar(x + w/2, nrm, w, color=PINK, label=r"normalized by $\sqrt{N}$")
    ax.axhline(0.85, ls="--", lw=0.9, color="k", alpha=0.6)
    ax.text(len(labs)-0.5, 0.86, "0.85", fontsize=9, ha="right")
    ax.set_xticks(x); ax.set_xticklabels(labs, rotation=0)
    ax.set_ylabel(r"$|r|$ with trimer size $N$")
    ax.set_ylim(0, 1.0)
    ax.legend(frameon=False)
    ax.spines[["right", "top"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(GFX / "SI_norm_size_corr.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    P("wrote SI_norm_size_corr.png")

    # ----- SI scatterplot grid of strongest correlations -----
    picks = [
        ("md", "md_rg_trimer_normsqrt", "md_avg_max_heavy_atom_dist_normsqrt"),
        ("md", "md_heat_capacity_liquid_normsqrt", "md_heat_capacity_gas_normsqrt"),
        ("md", "md_heat_capacity_liquid_normsqrt", "md_excess_heat_capacity_trimer_normsqrt"),
        ("mp", "md_density_trimer", "Density"),
        ("mp", "md_avg_auc_fit_at_zero_cross", "Density"),
        ("mp", "md_average_lifetime_ps", "Tensile_Strength"),
        ("mp", "md_average_lifetime_ps", "log10_ElongBreak"),
        ("mp", "md_avg_auc_fit_at_zero_cross", "Glass_Transition"),
        ("mp", "md_density_trimer", "Glass_Transition"),
    ]
    fig, axes = plt.subplots(3, 3, figsize=(11, 10))
    for ax, (kind, xcol, ycol) in zip(axes.ravel(), picks):
        src = d if kind == "md" else merged
        s = src[[xcol, ycol]].dropna()
        r = sp.stats.pearsonr(s[xcol], s[ycol])[0]
        sns.scatterplot(data=s, x=xcol, y=ycol, ax=ax, s=14,
                        color=PINK if kind == "md" else BLUE, alpha=0.55,
                        edgecolor="none")
        ax.set_xlabel(LABELS.get(xcol, xcol))
        ax.set_ylabel(LABELS.get(ycol, ycol))
        ax.set_title(f"r = {r:0.2f}  (n={len(s)})", fontsize=10)
        ax.spines[["right", "top"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(GFX / "SI_corr_scatters.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    P("wrote SI_corr_scatters.png")

    (OUT / "correlation_values.txt").write_text("\n".join(log))


if __name__ == "__main__":
    main()
