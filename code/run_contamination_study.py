"""Sealed-holdout study of synthetic-data contamination (tabular + time series)."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import erf, erfinv
from sklearn.datasets import load_breast_cancer, load_wine, fetch_openml
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import f1_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "figures"
FIG.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(42)
MIX_RATIOS = (0.0, 0.1, 0.3, 0.5, 0.7)
SEEDS = (0, 1, 2)


def gaussian_copula(X: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    n_real, d = X.shape
    ranks = np.empty_like(X)
    for j in range(d):
        col = X[:, j]
        ranks[:, j] = (col.argsort().argsort() + 1.0) / (n_real + 1.0)
    gauss = np.clip(np.sqrt(2) * erfinv(np.clip(2 * ranks - 1, -0.999999, 0.999999)), -4, 4)
    cov = np.cov(gauss, rowvar=False)
    cov = np.atleast_2d(cov) + 1e-3 * np.eye(d)
    z = rng.multivariate_normal(np.zeros(d), cov, size=n)
    u = 0.5 * (1.0 + erf(z / np.sqrt(2)))
    synth = np.empty((n, d))
    for j in range(d):
        synth[:, j] = np.quantile(X[:, j], np.clip(u[:, j], 1e-6, 1 - 1e-6))
    return synth


def independent_marginals(X: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    synth = np.empty((n, X.shape[1]))
    for j in range(X.shape[1]):
        synth[:, j] = rng.choice(X[:, j], size=n, replace=True)
    return synth


def near_duplicate_noise(X: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    idx = rng.integers(0, len(X), size=n)
    scale = np.std(X, axis=0, keepdims=True) + 1e-8
    return X[idx] + 0.05 * scale * rng.normal(size=(n, X.shape[1]))


def expected_calibration_error(y_true, proba, n_bins=10) -> float:
    if proba.ndim == 2:
        if proba.shape[1] == 2:
            conf = proba[:, 1]
            pred = (conf >= 0.5).astype(int)
            y = (y_true == 1).astype(float) if len(np.unique(y_true)) == 2 else (pred == y_true).astype(float)
            # binary ECE on positive class
            acc = y
        else:
            conf = proba.max(axis=1)
            pred = proba.argmax(axis=1)
            acc = (pred == y_true).astype(float)
    else:
        conf = proba
        pred = (conf >= 0.5).astype(int)
        acc = (pred == y_true).astype(float)
        conf = np.where(pred == 1, conf, 1 - conf)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        m = (conf > bins[i]) & (conf <= bins[i + 1]) if i else (conf >= bins[i]) & (conf <= bins[i + 1])
        if m.any():
            ece += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return float(ece)


def diversity(X: np.ndarray, max_n=400) -> float:
    X = np.asarray(X, dtype=float)
    if len(X) > max_n:
        idx = np.linspace(0, len(X) - 1, max_n).astype(int)
        X = X[idx]
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)
    # mean pairwise L2 on a subsample of pairs
    rng = np.random.default_rng(0)
    n = len(X)
    i = rng.integers(0, n, size=min(2000, n * 4))
    j = rng.integers(0, n, size=len(i))
    mask = i != j
    d = np.linalg.norm(X[i[mask]] - X[j[mask]], axis=1)
    return float(d.mean()) if len(d) else 0.0


def ks_mean(real: np.ndarray, synth: np.ndarray) -> float:
    scores = []
    for j in range(real.shape[1]):
        a = np.sort(real[:, j])
        b = np.sort(synth[:, j])
        # empirical KS via quantile grid
        grid = np.linspace(0, 1, 64)
        qa = np.quantile(a, grid)
        qb = np.quantile(b, grid)
        scores.append(np.max(np.abs(qa - qb)) / (np.std(a) + 1e-8))
    return float(np.mean(scores))


def mix_train(X, y, ratio, generator, rng):
    n = len(X)
    n_s = int(round(ratio * n))
    if n_s == 0:
        return X.copy(), y.copy()
    keep = n - n_s
    idx = rng.choice(n, size=keep, replace=False)
    Xs = generator(X, n_s, rng)
    if y is None:
        return np.vstack([X[idx], Xs]), None
    # nearest-label assignment from real train (weak supervision of synth)
    # use random real labels for independent/copula; for classification we sample labels from empirical
    ys = rng.choice(y, size=n_s, replace=True)
    # better: 5-NN labels in original space
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors=min(5, len(y)))
    knn.fit(X, y)
    ys = knn.predict(Xs)
    return np.vstack([X[idx], Xs]), np.concatenate([y[idx], ys])


def classifiers():
    return {
        "LogReg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=400, solver="lbfgs")),
        "RandomForest": RandomForestClassifier(n_estimators=120, random_state=0, n_jobs=-1),
        "HistGBM": HistGradientBoostingClassifier(max_depth=6, learning_rate=0.08, max_iter=120),
    }


def load_tabular():
    data = {}
    bc = load_breast_cancer()
    data["BreastCancer"] = (bc.data.astype(float), bc.target.astype(int))
    wine = load_wine()
    data["Wine"] = (wine.data.astype(float), wine.target.astype(int))
    try:
        diabetes = fetch_openml("diabetes", version=1, as_frame=False, parser="auto")
        X = diabetes.data.astype(float)
        y_raw = np.array(diabetes.target)
        classes, y = np.unique(y_raw, return_inverse=True)
        data["PimaDiabetes"] = (X, y.astype(int))
    except Exception:
        pass
    return data


def make_series(n=1400, seed=0):
    t = np.arange(n)
    rng = np.random.default_rng(seed)
    weekly = 8 * np.sin(2 * np.pi * t / 7)
    yearly = 15 * np.sin(2 * np.pi * t / 365.25)
    trend = 0.01 * t
    noise = rng.normal(0, 2.5, size=n)
    shocks = np.zeros(n)
    shocks[200] = 18
    shocks[800] = -12
    return trend + weekly + yearly + noise + shocks


def try_real_series():
    urls = [
        "https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv",
        "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv",
    ]
    for url in urls:
        try:
            df = pd.read_csv(url)
            col = df.select_dtypes(include=[np.number]).columns[-1]
            y = df[col].astype(float).to_numpy()
            if len(y) >= 200:
                return y, url.split("/")[-1]
        except Exception:
            continue
    return make_series(), "synthetic_seasonal"


def lag_matrix(y, lags=14):
    X, t = [], []
    for i in range(lags, len(y)):
        X.append(y[i - lags : i])
        t.append(y[i])
    return np.asarray(X), np.asarray(t)


def bootstrap_seasonal(y, n, season=7, rng=None):
    rng = rng or np.random.default_rng()
    # seasonal naive residuals
    resid = y[season:] - y[:-season]
    out = np.empty(n)
    out[: min(season, n)] = y[: min(season, n)]
    for i in range(season, n):
        out[i] = out[i - season] + rng.choice(resid)
    return out.reshape(-1, 1)


def copula_windows(windows, n, rng):
    return gaussian_copula(windows, n, rng)


def cri(err, err0, div, div0, ece, ece0, w=(0.5, 0.3, 0.2)):
    # All terms are bounded in [0, 1]; do not divide by a near-zero baseline error
    # (perfect train-split F1 would otherwise explode the index).
    d_err = min(1.0, max(0.0, err - err0))
    d_div = min(1.0, max(0.0, 1.0 - (div / (div0 + 1e-8))))
    d_cal = min(1.0, max(0.0, ece - ece0))
    return float(w[0] * d_err + w[1] * d_div + w[2] * d_cal)


def run_tabular(datasets):
    rows = []
    gens = {
        "GaussianCopula": gaussian_copula,
        "IndepMarginals": independent_marginals,
        "NearDuplicate": near_duplicate_noise,
    }
    for dname, (X, y) in datasets.items():
        for seed in SEEDS:
            Xtr, Xte, ytr, yte = train_test_split(
                X, y, test_size=0.3, random_state=seed, stratify=y
            )
            div0_ref = diversity(Xtr)
            for gname, gen in gens.items():
                for r in MIX_RATIOS:
                    rng = np.random.default_rng(1000 * seed + int(r * 100))
                    Xm, ym = mix_train(Xtr, ytr, r, gen, rng)
                    for mname, clf in classifiers().items():
                        model = clf
                        # new instance each time
                        if mname == "LogReg":
                            model = make_pipeline(
                                StandardScaler(),
                                LogisticRegression(max_iter=400, solver="lbfgs"),
                            )
                        elif mname == "RandomForest":
                            model = RandomForestClassifier(
                                n_estimators=120, random_state=seed, n_jobs=-1
                            )
                        else:
                            model = HistGradientBoostingClassifier(
                                max_depth=6, learning_rate=0.08, max_iter=120, random_state=seed
                            )
                        model.fit(Xm, ym)
                        pred = model.predict(Xte)
                        f1 = f1_score(yte, pred, average="macro")
                        err = 1.0 - f1
                        if hasattr(model, "predict_proba"):
                            ece = expected_calibration_error(yte, model.predict_proba(Xte))
                        else:
                            ece = 0.0
                        div = diversity(Xm)
                        fid = ks_mean(Xtr, Xm) if r > 0 else 0.0
                        rows.append(
                            dict(
                                domain="tabular",
                                dataset=dname,
                                generator=gname,
                                model=mname,
                                mix=r,
                                seed=seed,
                                error=err,
                                f1=f1,
                                ece=ece,
                                diversity=div,
                                diversity0=div0_ref,
                                fidelity_ks=fid,
                            )
                        )
    return pd.DataFrame(rows)


def run_timeseries():
    y, name = try_real_series()
    rows = []
    lags = 14
    n = len(y)
    cut = int(0.7 * n)
    y_train, y_test = y[:cut], y[cut:]
    Xtr, ttr = lag_matrix(y_train, lags)
    # build test with continuity from end of train
    full_X, full_t = lag_matrix(y, lags)
    test_start = cut
    mask = np.arange(lags, n) >= test_start
    Xte, tte = full_X[mask], full_t[mask]

    gens = {
        "SeasonalBootstrap": lambda W, n, rng: bootstrap_seasonal(
            W[:, -1] if False else y_train, n, 7, rng
        ),
        "WindowCopula": copula_windows,
        "NearDuplicate": near_duplicate_noise,
    }

    for seed in SEEDS:
        rng0 = np.random.default_rng(seed)
        for gname, gen in gens.items():
            for r in MIX_RATIOS:
                rng = np.random.default_rng(2000 * seed + int(r * 100))
                n_s = int(round(r * len(Xtr)))
                if n_s == 0:
                    Xm, tm = Xtr.copy(), ttr.copy()
                else:
                    keep = len(Xtr) - n_s
                    idx = rng.choice(len(Xtr), size=keep, replace=False)
                    if gname == "SeasonalBootstrap":
                        series_s = bootstrap_seasonal(y_train, max(n_s + lags, lags + 8), 7, rng)
                        Xs, ts = lag_matrix(series_s.ravel(), lags)
                        take = min(n_s, len(Xs))
                        Xs, ts = Xs[:take], ts[:take]
                    else:
                        Xs = gen(Xtr, n_s, rng)
                        # target = last-lag ridge-free: use mean of window as weak target, better: copy nearest
                        from sklearn.neighbors import KNeighborsRegressor
                        knn = KNeighborsRegressor(n_neighbors=min(5, len(ttr)))
                        knn.fit(Xtr, ttr)
                        ts = knn.predict(Xs)
                    Xm = np.vstack([Xtr[idx], Xs])
                    tm = np.concatenate([ttr[idx], ts])
                model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
                model.fit(Xm, tm)
                pred = model.predict(Xte)
                mae = mean_absolute_error(tte, pred)
                scale = np.mean(np.abs(tte)) + 1e-8
                nmae = mae / scale
                div = diversity(Xm)
                div0 = diversity(Xtr)
                fid = ks_mean(Xtr, Xm) if r > 0 else 0.0
                rows.append(
                    dict(
                        domain="timeseries",
                        dataset=name,
                        generator=gname,
                        model="RidgeLags",
                        mix=r,
                        seed=seed,
                        error=nmae,
                        f1=np.nan,
                        ece=0.0,
                        diversity=div,
                        diversity0=div0,
                        fidelity_ks=fid,
                    )
                )
    return pd.DataFrame(rows), y, name, cut


def attach_cri(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    keys = ["domain", "dataset", "generator", "model", "seed"]
    for key, g in df.groupby(keys):
        g = g.copy()
        base = g.loc[g["mix"] == 0.0]
        if base.empty:
            continue
        err0 = float(base["error"].mean())
        div0 = float(base["diversity"].mean())
        ece0 = float(base["ece"].mean())
        g["cri"] = [
            cri(r.error, err0, r.diversity, div0, r.ece, ece0) for r in g.itertuples()
        ]
        g["err0"] = err0
        out.append(g)
    return pd.concat(out, ignore_index=True)


def summarize(df):
    grp = ["domain", "dataset", "generator", "model", "mix"]
    return (
        df.groupby(grp, as_index=False)
        .agg(
            error_mean=("error", "mean"),
            error_std=("error", "std"),
            f1_mean=("f1", "mean"),
            ece_mean=("ece", "mean"),
            cri_mean=("cri", "mean"),
            cri_std=("cri", "std"),
            fidelity_mean=("fidelity_ks", "mean"),
            diversity_mean=("diversity", "mean"),
        )
        .sort_values(grp)
    )


# ---- figures ----
IEEE = {
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 160,
    "savefig.dpi": 220,
    "axes.grid": True,
    "grid.alpha": 0.25,
}


def fig_architecture():
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")
    boxes = [
        (0.3, 3.6, 2.2, 1.6, "Real data\n(tabular / TS)"),
        (3.0, 3.6, 2.3, 1.6, "Sealed split\ntrain | holdout"),
        (5.8, 4.5, 2.3, 1.2, "Generator\n(never sees holdout)"),
        (5.8, 2.7, 2.3, 1.2, "Mix ratio r\n0–70% synthetic"),
        (8.6, 3.6, 2.4, 1.6, "Train downstream\nmodel on mix"),
        (3.0, 0.5, 2.6, 1.5, "Fidelity (KS)\nlooks fine?"),
        (6.0, 0.5, 2.6, 1.5, "CRI audit\ndiversity + ECE"),
        (9.0, 0.5, 2.6, 1.5, "Holdout risk\nreal future data"),
    ]
    for x, y, w, h, t in boxes:
        rect = plt.Rectangle((x, y), w, h, fill=True, facecolor="#e8eef4", edgecolor="#1f3a5f", lw=1.2)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=8, color="#1f3a5f")
    arrows = [
        ((2.5, 4.4), (3.0, 4.4)),
        ((5.3, 4.4), (5.8, 5.1)),
        ((5.3, 4.4), (5.8, 3.3)),
        ((8.1, 5.1), (8.6, 4.6)),
        ((8.1, 3.3), (8.6, 4.2)),
        ((9.8, 3.6), (10.3, 2.0)),
        ((4.3, 3.6), (4.3, 2.0)),
        ((5.6, 1.25), (6.0, 1.25)),
        ((8.6, 1.25), (9.0, 1.25)),
    ]
    for (x1, y1), (x2, y2) in arrows:
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", color="#1f3a5f", lw=1.1))
    ax.set_title("Sealed-holdout contamination audit pipeline", loc="left", color="#1f3a5f")
    fig.tight_layout()
    fig.savefig(FIG / "fig1_architecture.png", bbox_inches="tight")
    plt.close()


def fig_error_curves(summ):
    tab = summ[(summ.domain == "tabular") & (summ.model == "RandomForest")]
    names = list(tab["dataset"].unique())
    fig, axes = plt.subplots(1, len(names), figsize=(7.4, 2.9), sharey=True)
    if len(names) == 1:
        axes = [axes]
    for ax, dname in zip(axes, names):
        g = tab[tab.dataset == dname]
        for gen, gg in g.groupby("generator"):
            ax.errorbar(
                gg["mix"], gg["error_mean"], yerr=gg["error_std"].fillna(0), marker="o", lw=1.5, label=gen
            )
        ax.set_title(dname)
        ax.set_xlabel("Mix ratio r")
        ax.legend(frameon=False, fontsize=7)
    axes[0].set_ylabel("Holdout error (1 − macro-F1)")
    fig.suptitle("Tabular sealed-holdout error versus synthetic mix", y=1.05, fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "fig2_tabular_error.png", bbox_inches="tight")
    plt.close()


def fig_cri(summ):
    tab = summ[(summ.domain == "tabular") & (summ.model == "RandomForest")]
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    for (dname, gen), g in tab.groupby(["dataset", "generator"]):
        if gen != "GaussianCopula":
            continue
        ax.plot(g["mix"], g["cri_mean"], marker="o", lw=1.6, label=dname)
    ax.set_xlabel("Synthetic mix ratio r")
    ax.set_ylabel("Contamination Risk Index (CRI)")
    ax.set_title("CRI rises with copula contamination on sealed holdout")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "fig3_cri.png", bbox_inches="tight")
    plt.close()


def fig_fidelity_vs_risk(summ):
    tab = summ[(summ.domain == "tabular") & (summ.mix > 0) & (summ.model == "RandomForest")]
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    for gen, g in tab.groupby("generator"):
        ax.scatter(g["fidelity_mean"], g["error_mean"], s=36, alpha=0.85, label=gen)
    ax.set_xlabel("Train-mix fidelity gap (mean KS, lower = closer to real train)")
    ax.set_ylabel("Sealed-holdout error (1 − macro-F1)")
    ax.set_title("Fidelity on the mixed train set does not reliably predict holdout risk")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "fig4_fidelity_vs_risk.png", bbox_inches="tight")
    plt.close()


def fig_timeseries(summ, y, name, cut):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    t = np.arange(len(y))
    axes[0].plot(t[:cut], y[:cut], lw=0.8, label="Train (real)", color="#1f3a5f")
    axes[0].plot(t[cut:], y[cut:], lw=0.9, label="Sealed future holdout", color="#b03a2e")
    axes[0].axvline(cut, color="gray", ls="--", lw=1)
    axes[0].set_title(f"Series split: {name}")
    axes[0].set_xlabel("Time index")
    axes[0].set_ylabel("Value")
    axes[0].legend(frameon=False, loc="upper left")

    ts = summ[summ.domain == "timeseries"]
    for gen, g in ts.groupby("generator"):
        g = g.groupby("mix", as_index=False).agg(error_mean=("error_mean", "mean"), error_std=("error_std", "mean"))
        axes[1].plot(g["mix"], g["error_mean"], marker="o", lw=1.5, label=gen)
    axes[1].set_xlabel("Synthetic mix ratio r")
    axes[1].set_ylabel("Normalized MAE on future holdout")
    axes[1].set_title("Time-series contamination")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "fig5_timeseries.png", bbox_inches="tight")
    plt.close()


def fig_heatmap(summ):
    tab = summ[(summ.domain == "tabular") & (summ.generator == "GaussianCopula")]
    # average over datasets
    p = tab.pivot_table(index="model", columns="mix", values="error_mean", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(7.2, 2.8))
    im = ax.imshow(p.to_numpy(), aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(p.columns)))
    ax.set_xticklabels([f"{c:.0%}" for c in p.columns])
    ax.set_yticks(range(len(p.index)))
    ax.set_yticklabels(list(p.index))
    ax.set_xlabel("Synthetic mix ratio")
    ax.set_title("Mean sealed-holdout error under Gaussian copula (all tabular sets)")
    for i in range(p.shape[0]):
        for j in range(p.shape[1]):
            ax.text(j, i, f"{p.to_numpy()[i, j]:.3f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Error")
    fig.tight_layout()
    fig.savefig(FIG / "fig6_heatmap.png", bbox_inches="tight")
    plt.close()


def main():
    plt.rcParams.update(IEEE)
    raw_path = ROOT / "raw_results.csv"
    if raw_path.exists() and "cri" not in "".join(open(raw_path).readline()):
        pass  # always recompute cri from raw columns
    if raw_path.exists():
        print("Recomputing CRI from cached raw_results.csv")
        raw = pd.read_csv(raw_path)
        raw = raw.drop(columns=[c for c in ["cri", "err0"] if c in raw.columns])
        # drop domain mix-0 cri artifacts
        df = attach_cri(raw)
        y, name = try_real_series()
        cut = int(0.7 * len(y))
        datasets = {k: None for k in df.loc[df.domain == "tabular", "dataset"].unique()}
        tab = df[df.domain == "tabular"]
        ts = df[df.domain == "timeseries"]
    else:
        print("Loading datasets...")
        datasets = load_tabular()
        print("tabular:", list(datasets))
        tab = run_tabular(datasets)
        ts, y, name, cut = run_timeseries()
        df = attach_cri(pd.concat([tab, ts], ignore_index=True))
        datasets = {k: None for k in datasets}
    summ = summarize(df)
    df.to_csv(ROOT / "raw_results.csv", index=False)
    summ.to_csv(ROOT / "summary_results.csv", index=False)

    fig_architecture()
    fig_error_curves(summ)
    fig_cri(summ)
    fig_fidelity_vs_risk(summ)
    fig_timeseries(summ, y, name, cut)
    fig_heatmap(summ)

    # headline numbers for the paper
    def mean_at(domain, mix, generator=None, model=None):
        q = df[df.domain == domain]
        if generator:
            q = q[q.generator == generator]
        if model:
            q = q[q.model == model]
        q = q[q.mix == mix]
        return float(q["error"].mean()), float(q["cri"].mean()), float(q["f1"].mean()) if domain == "tabular" else float("nan")

    headlines = {}
    for r in MIX_RATIOS:
        e, c, f = mean_at("tabular", r, "GaussianCopula", "RandomForest")
        headlines[f"tab_copula_rf_r{int(r*100)}"] = {"error": e, "cri": c, "f1": f}
    e0, _, f0 = mean_at("tabular", 0.0, "GaussianCopula", "RandomForest")
    e70, c70, f70 = mean_at("tabular", 0.7, "GaussianCopula", "RandomForest")
    headlines["tab_rf_copula_f1_drop_pct"] = (f0 - f70) / (abs(f0) + 1e-12) * 100
    headlines["tab_rf_copula_error_rel"] = (e70 - e0) / (abs(e0) + 1e-12) * 100

    # near-duplicate high fidelity but risk
    nd = df[(df.domain == "tabular") & (df.generator == "NearDuplicate") & (df.model == "RandomForest") & (df.mix == 0.7)]
    cop = df[(df.domain == "tabular") & (df.generator == "GaussianCopula") & (df.model == "RandomForest") & (df.mix == 0.7)]
    headlines["nd_fidelity"] = float(nd["fidelity_ks"].mean())
    headlines["cop_fidelity"] = float(cop["fidelity_ks"].mean())
    headlines["nd_error"] = float(nd["error"].mean())
    headlines["cop_error"] = float(cop["error"].mean())

    ts70 = df[(df.domain == "timeseries") & (df.mix == 0.7)]
    ts0 = df[(df.domain == "timeseries") & (df.mix == 0.0)]
    headlines["ts_dataset"] = name
    headlines["ts_nmae_0"] = float(ts0["error"].mean())
    headlines["ts_nmae_70"] = float(ts70["error"].mean())
    headlines["n_tabular_rows"] = int(len(tab))
    headlines["n_ts_rows"] = int(len(ts))
    headlines["datasets"] = list(datasets.keys()) + [name]

    # per-dataset table
    table = []
    for dname in datasets:
        for r in MIX_RATIOS:
            q = df[(df.dataset == dname) & (df.generator == "GaussianCopula") & (df.model == "RandomForest") & (df.mix == r)]
            table.append(
                {
                    "dataset": dname,
                    "mix": r,
                    "f1": float(q["f1"].mean()),
                    "f1_std": float(q["f1"].std()),
                    "error": float(q["error"].mean()),
                    "ece": float(q["ece"].mean()),
                    "cri": float(q["cri"].mean()),
                }
            )
    headlines["table_rf_copula"] = table

    with open(ROOT / "headlines.json", "w") as f:
        json.dump(headlines, f, indent=2)
    print(json.dumps({k: v for k, v in headlines.items() if k != "table_rf_copula"}, indent=2))
    print("Wrote", ROOT)


if __name__ == "__main__":
    main()
