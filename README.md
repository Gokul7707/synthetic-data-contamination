# Detecting Synthetic-Data Contamination in Tabular and Time-Series Training Pipelines

Preprint and reproduction code for a sealed-holdout audit of synthetic rows mixed into tabular classifiers and a univariate forecaster.

**Author:** Gokul Srinivasan Prabu, Department of AI and Data Science, Sri Krishna College of Engineering and Technology, Coimbatore, India ([gokulsrinivasan2020@gmail.com](mailto:gokulsrinivasan2020@gmail.com))

**License:** [CC BY 4.0](LICENSE)

## Paper

- Source: [`paper/ms.tex`](paper/ms.tex)
- Figures: [`paper/figures/`](paper/figures/)
- Compiled PDF (after build): `paper/ms.pdf`

This is a **free preprint**, not a paid journal. The recommended archival venue is [arXiv](https://arxiv.org/) under `cs.LG` (cross-list `stat.ML`, `cs.AI`).

## What the study does

1. Freeze a real holdout that **no generator is allowed to see**.
2. Replace a fraction `r` of the training set with synthetic samples (copula, independent marginals, near-duplicates; seasonal bootstrap on the series).
3. Train ordinary models (logistic regression, Random Forest, HistGBM; ridge-on-lags).
4. Compare sealed-holdout error with train-set KS fidelity, diversity, and expected calibration error.

Headline (Random Forest, Gaussian copula, mean of three public tables): macro-F1 **0.890 → 0.830** at a 70% mix; Wine **1.000 → 0.858**. Seasonal residual bootstrap raised Melbourne temperature nMAE from **0.154 → 0.212**. Train-set KS fidelity correlated only weakly with holdout error (Pearson **r = 0.15**).

## Reproduce

Python 3.12, CPU only, no GPU and no paid API.

```bash
pip install -r code/requirements.txt
python code/run_contamination_study.py
python paper/make_figures.py
```

Cached results used in the paper are in [`results/`](results/).

## arXiv submission (free)

1. Create an account at [arxiv.org](https://arxiv.org/) with your college email if possible.
2. New authors in `cs.LG` need an **endorsement** from someone who has already published in that category. Ask a faculty member.
3. Submit the contents of `paper/` (LaTeX + `figures/`) as the source package. License: **CC BY 4.0**.
4. After the arXiv ID appears, archive this GitHub repo on [Zenodo](https://zenodo.org/) for a DOI.

Do not pay a journal or conference that emails you an invoice before review. That is not how legitimate CS venues work.
