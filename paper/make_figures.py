"""Publication figures for the sealed-holdout contamination preprint."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)
SUMM = ROOT.parent / "results" / "summary_results.csv"

COL = 3.45
WIDE = 7.05

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "Times"],
        "font.size": 9,
        "axes.titlesize": 9.5,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7.5,
        "axes.linewidth": 0.7,
        "lines.linewidth": 1.4,
        "lines.markersize": 5,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.grid": True,
        "grid.alpha": 0.28,
        "grid.linewidth": 0.4,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 300,
    }
)

COLORS = {
    "GaussianCopula": "#1f4e79",
    "IndepMarginals": "#c45911",
    "NearDuplicate": "#548235",
    "SeasonalBootstrap": "#c45911",
    "WindowCopula": "#1f4e79",
}
MARKERS = {
    "GaussianCopula": "o",
    "IndepMarginals": "s",
    "NearDuplicate": "^",
    "SeasonalBootstrap": "D",
    "WindowCopula": "o",
}
PRETTY = {
    "GaussianCopula": "Gaussian copula",
    "IndepMarginals": "Indep. marginals",
    "NearDuplicate": "Near-duplicate",
    "SeasonalBootstrap": "Seasonal bootstrap",
    "WindowCopula": "Window copula",
    "BreastCancer": "Breast Cancer",
    "PimaDiabetes": "Pima Diabetes",
    "Wine": "Wine",
    "RandomForest": "Random Forest",
    "LogReg": "Logistic regression",
    "HistGBM": "Hist. GBM",
}


def save(fig, stem: str):
    pdf = FIG / f"{stem}.pdf"
    png = FIG / f"{stem}.png"
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.04, facecolor="white")
    fig.savefig(png, bbox_inches="tight", pad_inches=0.04, facecolor="white")
    plt.close(fig)
    print("wrote", pdf.name)


def fig1():
    fig, ax = plt.subplots(figsize=(COL, 4.55))
    ax.set_xlim(0, 10)
    ax.set_ylim(0.2, 13.4)
    ax.axis("off")
    labels = [
        "1. Real tabular / time-series records",
        "2. Seal a future holdout (never shown\n    to any generator or mixer)",
        "3. Fit generator on the train slice only",
        "4. Replace fraction r of train rows\n    (r = 0, 0.1, 0.3, 0.5, 0.7)",
        "5. Train an ordinary downstream model",
        "6. Score sealed holdout, diversity, ECE",
    ]
    ys = [11.45, 9.25, 7.15, 5.05, 3.15, 1.15]
    h = 1.55
    for y, text in zip(ys, labels):
        ax.add_patch(
            FancyBboxPatch(
                (0.45, y),
                9.1,
                h,
                boxstyle="round,pad=0.02,rounding_size=0.14",
                linewidth=0.9,
                edgecolor="#1f4e79",
                facecolor="#eef3f8",
            )
        )
        ax.text(5, y + h / 2, text, ha="center", va="center", fontsize=8.2, color="#1b1b1b")
    for y_above, y_below in zip(ys[:-1], ys[1:]):
        ax.add_patch(
            FancyArrowPatch(
                (5, y_above - 0.02),
                (5, y_below + h + 0.04),
                arrowstyle="-|>",
                mutation_scale=10,
                lw=0.9,
                color="#1f4e79",
            )
        )
    ax.set_title("Sealed-holdout contamination audit", pad=4)
    save(fig, "fig1_architecture")


def fig2(summ):
    tab = summ[(summ.domain == "tabular") & (summ.model == "RandomForest")]
    names = ["BreastCancer", "PimaDiabetes", "Wine"]
    fig, axes = plt.subplots(1, 3, figsize=(WIDE, 2.55), sharey=False)
    for ax, dname in zip(axes, names):
        g = tab[tab.dataset == dname]
        for gen, gg in g.groupby("generator"):
            gg = gg.sort_values("mix")
            ax.errorbar(
                gg["mix"],
                gg["error_mean"],
                yerr=gg["error_std"].fillna(0),
                marker=MARKERS[gen],
                color=COLORS[gen],
                capsize=2,
                label=PRETTY[gen],
            )
        ax.set_title(PRETTY[dname], loc="left", pad=2)
        ax.set_xticks([0, 0.1, 0.3, 0.5, 0.7])
        ax.set_xlabel("Mix ratio $r$")
    axes[0].set_ylabel("Holdout error ($1 -$ macro-F1)")
    axes[0].legend(frameon=False, loc="upper left")
    fig.tight_layout(w_pad=0.6)
    save(fig, "fig2_tabular_error")


def fig3(summ):
    tab = summ[
        (summ.domain == "tabular")
        & (summ.model == "RandomForest")
        & (summ.generator == "GaussianCopula")
    ]
    fig, ax = plt.subplots(figsize=(COL, 2.35))
    for dname, g in tab.groupby("dataset"):
        g = g.sort_values("mix")
        ax.plot(
            g["mix"],
            g["cri_mean"],
            marker="o",
            color={"BreastCancer": "#1f4e79", "PimaDiabetes": "#c45911", "Wine": "#548235"}[dname],
            label=PRETTY[dname],
        )
    ax.set_xlabel("Mix ratio $r$")
    ax.set_ylabel("CRI")
    ax.set_xticks([0, 0.1, 0.3, 0.5, 0.7])
    ax.legend(frameon=False, loc="upper left")
    ax.set_title("CRI vs. copula mix ratio (Random Forest)", loc="left", pad=3)
    fig.tight_layout()
    save(fig, "fig3_cri")


def fig4(summ):
    tab = summ[(summ.domain == "tabular") & (summ.mix > 0) & (summ.model == "RandomForest")]
    r = float(np.corrcoef(tab["fidelity_mean"], tab["error_mean"])[0, 1])
    fig, ax = plt.subplots(figsize=(COL, 2.45))
    for gen, g in tab.groupby("generator"):
        ax.scatter(
            g["fidelity_mean"],
            g["error_mean"],
            s=32,
            c=COLORS[gen],
            marker=MARKERS[gen],
            label=PRETTY[gen],
            zorder=3,
        )
    ax.set_xlabel("Train-mix KS gap (lower = closer to real train)")
    ax.set_ylabel("Sealed-holdout error")
    ax.legend(frameon=False, loc="best")
    ax.set_title(f"Fidelity vs. holdout risk (Pearson r={r:.2f})", loc="left", pad=3)
    fig.tight_layout()
    save(fig, "fig4_fidelity_vs_risk")


def fig5(summ, y, cut):
    fig, axes = plt.subplots(
        1, 2, figsize=(WIDE, 2.7), gridspec_kw={"width_ratios": [1.25, 1]}
    )
    t = np.arange(len(y))
    axes[0].plot(t[:cut], y[:cut], lw=0.45, color="#1f4e79", label="Train")
    axes[0].plot(t[cut:], y[cut:], lw=0.55, color="#c45911", label="Sealed holdout")
    axes[0].axvline(cut, color="0.45", ls="--", lw=0.7)
    axes[0].set_xlabel("Day index")
    axes[0].set_ylabel("Min. temp. (\N{DEGREE SIGN}C)")
    axes[0].set_title("Melbourne daily minimum temperatures", loc="left", pad=3)
    axes[0].legend(frameon=False, loc="upper right")

    ts = summ[summ.domain == "timeseries"]
    for gen, g in ts.groupby("generator"):
        gg = g.groupby("mix", as_index=False).agg(error_mean=("error_mean", "mean"))
        axes[1].plot(
            gg["mix"],
            gg["error_mean"],
            marker=MARKERS.get(gen, "o"),
            color=COLORS.get(gen, "#333"),
            label=PRETTY.get(gen, gen),
        )
    axes[1].set_xlabel("Mix ratio $r$")
    axes[1].set_ylabel("Normalized MAE")
    axes[1].set_xticks([0, 0.1, 0.3, 0.5, 0.7])
    axes[1].legend(frameon=False, loc="upper left")
    axes[1].set_title("Future-holdout error by generator", loc="left", pad=3)
    fig.tight_layout(w_pad=0.8)
    save(fig, "fig5_timeseries")


def fig6(summ):
    tab = summ[(summ.domain == "tabular") & (summ.generator == "GaussianCopula")]
    p = tab.pivot_table(index="model", columns="mix", values="error_mean", aggfunc="mean")
    p = p.rename(index=PRETTY)
    fig, ax = plt.subplots(figsize=(COL, 2.15))
    im = ax.imshow(p.to_numpy(), aspect="auto", cmap="YlOrRd", vmin=0.10, vmax=0.20)
    ax.set_xticks(range(len(p.columns)))
    ax.set_xticklabels([f"{c:.0%}" for c in p.columns])
    ax.set_yticks(range(len(p.index)))
    ax.set_yticklabels(list(p.index))
    ax.set_xlabel("Synthetic mix ratio")
    ax.set_title("Mean holdout error, Gaussian copula", loc="left", pad=3)
    ax.grid(False)
    for i in range(p.shape[0]):
        for j in range(p.shape[1]):
            val = p.to_numpy()[i, j]
            ax.text(
                j,
                i,
                f"{val:.3f}",
                ha="center",
                va="center",
                fontsize=7.5,
                color="black" if val < 0.16 else "white",
            )
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=7)
    cbar.set_label("Error", fontsize=8)
    fig.tight_layout()
    save(fig, "fig6_heatmap")


def load_series():
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv"
    df = pd.read_csv(url)
    col = df.select_dtypes(include=[np.number]).columns[-1]
    y = df[col].astype(float).to_numpy()
    return y, int(0.7 * len(y))


def main():
    summ = pd.read_csv(SUMM)
    fig1()
    fig2(summ)
    fig3(summ)
    fig4(summ)
    y, cut = load_series()
    fig5(summ, y, cut)
    fig6(summ)


if __name__ == "__main__":
    main()
