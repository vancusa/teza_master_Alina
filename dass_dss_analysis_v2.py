"""
DASS-21 & DSS-24 Before / After Statistical Analysis
-----------------------------------------------------
Input:   data/dass_before.xlsx, data/dass_after.xlsx
         data/dss_before.xlsx,  data/dss_after.xlsx
         Row 1 = question headers; subsequent rows = numeric responses.
         DASS coded 1–4 → adjusted to 0–3.
         DSS  coded 1–5 → adjusted to 0–4.

NOTE ON SAMPLE SIZES:
  The before files contain 29 participants.
  The after files contain 129 participants total; only the first 29 rows
  are used so that before and after are matched (paired) samples.
  Paired tests (Wilcoxon signed-rank) are used for the before/after
  comparison in Section 9.

Output: console tables + PNG figures saved to output/
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

# ── CONFIG ────────────────────────────────────────────────────────────────────
DATA_DIR   = "data"
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted")

# ── SUBSCALE ITEM INDICES (1-based) ───────────────────────────────────────────
DASS_STRESS     = [1, 6, 8, 11, 12, 14, 18]
DASS_ANXIETY    = [2, 4, 7,  9, 15, 19, 20]
DASS_DEPRESSION = [3, 5, 10, 13, 16, 17, 21]

DSS_AVAILABILITY = [1, 4,  7,  8, 12, 16, 18]
DSS_FOMO         = [5, 10, 13, 21]
DSS_APPROVAL     = [3, 9,  17, 20, 22, 24]
DSS_OVERLOAD     = [2, 6,  11, 14, 15, 19, 23]

# Theoretical maxima (n_items × max_item_score after transformation)
SCALE_MAX = {
    "DASS_Stress":       21,   # 7 × 3
    "DASS_Anxiety":      21,
    "DASS_Depression":   21,
    "DSS_Availability":  28,   # 7 × 4
    "DSS_FOMO":          16,   # 4 × 4
    "DSS_Approval":      24,   # 6 × 4
    "DSS_Overload":      28,   # 7 × 4
}

DASS_COLS = ["DASS_Stress", "DASS_Anxiety", "DASS_Depression"]
DSS_COLS  = ["DSS_Availability", "DSS_FOMO", "DSS_Approval", "DSS_Overload"]
ALL_COLS  = DASS_COLS + DSS_COLS

# Short display labels for axes / radar
SHORT = {
    "DASS_Stress":      "Stress",
    "DASS_Anxiety":     "Anxiety",
    "DASS_Depression":  "Depression",
    "DSS_Availability": "Availability",
    "DSS_FOMO":         "FOMO",
    "DSS_Approval":     "Approval",
    "DSS_Overload":     "Overload",
}

# ── LOAD DATA ─────────────────────────────────────────────────────────────────
def load_survey(path):
    """Load xlsx: header row = question text; data rows = numeric responses."""
    df = pd.read_excel(path, header=0)
    df.columns = range(1, len(df.columns) + 1)   # 1-based integer column names
    return df.apply(pd.to_numeric, errors="coerce").dropna(how="all")

dass_before_raw = load_survey(f"{DATA_DIR}/dass_before.xlsx")
dss_before_raw  = load_survey(f"{DATA_DIR}/dss_before.xlsx")

N_PAIRED = len(dass_before_raw)   # 29 — keep only the matching rows from after

dass_after_raw  = load_survey(f"{DATA_DIR}/dass_after.xlsx").iloc[:N_PAIRED].reset_index(drop=True)
dss_after_raw   = load_survey(f"{DATA_DIR}/dss_after.xlsx").iloc[:N_PAIRED].reset_index(drop=True)

# ── STEP 2: TRANSFORM (-1 so scores start at 0) ───────────────────────────────
dass_before = dass_before_raw - 1   # 1–4 → 0–3
dass_after  = dass_after_raw  - 1
dss_before  = dss_before_raw  - 1   # 1–5 → 0–4
dss_after   = dss_after_raw   - 1

# ── COMPUTE SUBSCALES ──────────────────────────────────────────────────────────
def make_subscales(dass, dss):
    d = pd.DataFrame(index=dass.index)
    d["DASS_Stress"]      = dass[[c for c in DASS_STRESS]].sum(axis=1, min_count=1)
    d["DASS_Anxiety"]     = dass[[c for c in DASS_ANXIETY]].sum(axis=1, min_count=1)
    d["DASS_Depression"]  = dass[[c for c in DASS_DEPRESSION]].sum(axis=1, min_count=1)
    d["DSS_Availability"] = dss[[c for c in DSS_AVAILABILITY]].sum(axis=1, min_count=1)
    d["DSS_FOMO"]         = dss[[c for c in DSS_FOMO]].sum(axis=1, min_count=1)
    d["DSS_Approval"]     = dss[[c for c in DSS_APPROVAL]].sum(axis=1, min_count=1)
    d["DSS_Overload"]     = dss[[c for c in DSS_OVERLOAD]].sum(axis=1, min_count=1)
    return d

scores_before = make_subscales(dass_before, dss_before)
scores_after  = make_subscales(dass_after,  dss_after)

# ── HELPERS ────────────────────────────────────────────────────────────────────
def cronbach_alpha(item_df):
    item_df = item_df.dropna()
    k = item_df.shape[1]
    if k < 2:
        return np.nan
    item_vars = item_df.var(axis=0, ddof=1)
    total_var = item_df.sum(axis=1).var(ddof=1)
    if total_var == 0:
        return np.nan
    return (k / (k - 1)) * (1 - item_vars.sum() / total_var)

def alpha_label(a):
    if np.isnan(a):  return "n/a"
    if a >= 0.9:     return "excellent"
    if a >= 0.8:     return "good"
    if a >= 0.7:     return "acceptable"
    if a >= 0.6:     return "questionable"
    return "poor"

def sep(title=""):
    print("\n" + "=" * 72)
    if title:
        print(title)
        print("=" * 72)

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — DESCRIPTIVE STATISTICS
# ═══════════════════════════════════════════════════════════════════════════════
sep("DESCRIPTIVE STATISTICS")

desc_rows = []
for label, scores in [("Before", scores_before), ("After", scores_after)]:
    for col in ALL_COLS:
        s = scores[col].dropna()
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        desc_rows.append({
            "Timepoint": label,
            "Scale":     col,
            "N":         int(s.count()),
            "Mean":      round(s.mean(), 3),
            "SD":        round(s.std(ddof=1), 3),
            "Median":    round(s.median(), 3),
            "IQR":       f"{q1:.1f}–{q3:.1f}",
            "Min":       round(s.min(), 1),
            "Max":       round(s.max(), 1),
            "Skew":      round(s.skew(), 3),
            "Kurt":      round(s.kurt(), 3),
        })

desc_df = pd.DataFrame(desc_rows).set_index(["Timepoint", "Scale"])
print(desc_df.to_string())

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — INTERNAL CONSISTENCY (Cronbach's α)
# ═══════════════════════════════════════════════════════════════════════════════
sep("INTERNAL CONSISTENCY (Cronbach's α)")

scale_items_map = {
    "DASS Stress":      (dass_before, [c for c in DASS_STRESS],      dass_after, [c for c in DASS_STRESS]),
    "DASS Anxiety":     (dass_before, [c for c in DASS_ANXIETY],     dass_after, [c for c in DASS_ANXIETY]),
    "DASS Depression":  (dass_before, [c for c in DASS_DEPRESSION],  dass_after, [c for c in DASS_DEPRESSION]),
    "DASS-21 Total":    (dass_before, list(range(1, 22)),             dass_after, list(range(1, 22))),
    "DSS Availability": (dss_before,  [c for c in DSS_AVAILABILITY], dss_after,  [c for c in DSS_AVAILABILITY]),
    "DSS FOMO":         (dss_before,  [c for c in DSS_FOMO],         dss_after,  [c for c in DSS_FOMO]),
    "DSS Approval":     (dss_before,  [c for c in DSS_APPROVAL],     dss_after,  [c for c in DSS_APPROVAL]),
    "DSS Overload":     (dss_before,  [c for c in DSS_OVERLOAD],     dss_after,  [c for c in DSS_OVERLOAD]),
    "DSS-24 Total":     (dss_before,  list(range(1, 25)),             dss_after,  list(range(1, 25))),
}

alpha_rows = []
for name, (db, ib, da, ia) in scale_items_map.items():
    ab = cronbach_alpha(db[ib])
    aa = cronbach_alpha(da[ia])
    alpha_rows.append({
        "Scale":               name,
        "α Before":            round(ab, 3),
        "Interpretation":      alpha_label(ab),
        "α After":             round(aa, 3),
        "Interpretation.1":    alpha_label(aa),
    })

alpha_df = pd.DataFrame(alpha_rows).set_index("Scale")
print(alpha_df.to_string())

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — NORMALITY (Shapiro-Wilk)
# ═══════════════════════════════════════════════════════════════════════════════
sep("NORMALITY (Shapiro-Wilk, α=0.05)")

norm_rows = []
for label, scores in [("Before", scores_before), ("After", scores_after)]:
    for col in ALL_COLS:
        s = scores[col].dropna()
        w, p = stats.shapiro(s)
        norm_rows.append({
            "Timepoint": label,
            "Scale":     col,
            "N":         len(s),
            "W":         round(w, 4),
            "p-value":   round(p, 4),
            "Result":    "Normal" if p > 0.05 else "Non-normal",
        })

norm_df = pd.DataFrame(norm_rows).set_index(["Timepoint", "Scale"])
print(norm_df.to_string())

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 01 — Distributions vs Normal distribution
# ═══════════════════════════════════════════════════════════════════════════════
n_cols = len(ALL_COLS)
fig, axes = plt.subplots(2, n_cols, figsize=(3.5 * n_cols, 8))

for row_i, (label, scores) in enumerate([("Before", scores_before), ("After", scores_after)]):
    for col_i, col in enumerate(ALL_COLS):
        ax = axes[row_i, col_i]
        s = scores[col].dropna()

        sns.histplot(s, kde=True, stat="density", ax=ax,
                     color="steelblue", alpha=0.55, linewidth=0)

        mu, sigma = s.mean(), s.std()
        x_range = np.linspace(s.min() - 0.5, s.max() + 0.5, 300)
        ax.plot(x_range, stats.norm.pdf(x_range, mu, sigma),
                "r-", linewidth=1.8, label="Normal fit")
        ax.axvline(mu, color="navy", linestyle="--", linewidth=1, label=f"Mean={mu:.1f}")
        ax.axvline(s.median(), color="green", linestyle=":", linewidth=1,
                   label=f"Median={s.median():.1f}")

        w, p = stats.shapiro(s)
        ax.set_title(f"{SHORT[col]}\n({label}, n={len(s)})\nW={w:.3f}, p={p:.3f}", fontsize=8)
        ax.legend(fontsize=6, loc="upper right")
        ax.set_xlabel("Score", fontsize=8)
        ax.set_ylabel("Density", fontsize=8)
        ax.tick_params(labelsize=7)

plt.suptitle("Distribution of Subscale Scores vs Normal Distribution", fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig01_distributions.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n[Saved] fig01_distributions.png")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — DASS ↔ DSS CORRELATIONS (Spearman)
# ═══════════════════════════════════════════════════════════════════════════════
sep("DASS ↔ DSS CORRELATIONS (Spearman)")

for label, scores in [("Before", scores_before), ("After", scores_after)]:
    print(f"\n  {label} (n={len(scores)}):")
    header = f"  {'':>22}" + "".join(f"  {SHORT[d]:>13}" for d in DSS_COLS)
    print(header)
    print("  " + "-" * (22 + 15 * len(DSS_COLS)))
    for da in DASS_COLS:
        row_str = f"  {SHORT[da]:>22}"
        for ds in DSS_COLS:
            valid = scores[[da, ds]].dropna()
            r, p = stats.spearmanr(valid[da], valid[ds])
            sig = "*" if p < 0.05 else " "
            row_str += f"  {r:>6.3f}{sig}(p={p:.3f})"
        print(row_str)
    print("  * p < 0.05")

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 02 — Correlation heatmaps (Before / After)
# ═══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, (label, scores) in zip(axes, [("Before", scores_before), ("After", scores_after)]):
    corr_mat = pd.DataFrame(index=DASS_COLS, columns=DSS_COLS, dtype=float)
    pval_mat = pd.DataFrame(index=DASS_COLS, columns=DSS_COLS, dtype=float)
    for da in DASS_COLS:
        for ds in DSS_COLS:
            valid = scores[[da, ds]].dropna()
            r, p = stats.spearmanr(valid[da], valid[ds])
            corr_mat.loc[da, ds] = r
            pval_mat.loc[da, ds] = p

    sns.heatmap(
        corr_mat.astype(float), annot=True, fmt=".2f", cmap="RdBu_r",
        vmin=-1, vmax=1, ax=ax, linewidths=0.5,
        xticklabels=[SHORT[c] for c in DSS_COLS],
        yticklabels=[SHORT[c] for c in DASS_COLS],
    )
    for i, da in enumerate(DASS_COLS):
        for j, ds in enumerate(DSS_COLS):
            if pval_mat.loc[da, ds] > 0.05:
                ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False,
                             edgecolor="gray", lw=2, linestyle="--"))
    ax.set_title(
        f"DASS ↔ DSS Spearman Correlations — {label} (n={len(scores)})\n"
        f"(dashed border = p > 0.05)", fontsize=11
    )
    ax.set_xlabel("DSS Subscale")
    ax.set_ylabel("DASS Subscale")

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig02_correlation_heatmaps.png", dpi=150, bbox_inches="tight")
plt.close()
print("[Saved] fig02_correlation_heatmaps.png")

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 03 & 04 — Scatter DASS↔DSS with linearity check (Before / After)
# ═══════════════════════════════════════════════════════════════════════════════
def scatter_grid(scores, label, fig_path):
    """
    3×4 grid of scatter plots (each DASS subscale × each DSS subscale).
    Each panel shows data points, a linear OLS fit, and a LOWESS curve.
    Reports Pearson r (linear) and Spearman ρ (monotonic); when |r| ≈ |ρ| the
    relationship is approximately linear.
    """
    fig, axes = plt.subplots(len(DASS_COLS), len(DSS_COLS),
                             figsize=(4.5 * len(DSS_COLS), 4 * len(DASS_COLS)),
                             squeeze=False)
    for ri, da in enumerate(DASS_COLS):
        for ci, ds in enumerate(DSS_COLS):
            ax = axes[ri, ci]
            valid = scores[[da, ds]].dropna()
            x, y = valid[ds].values, valid[da].values

            ax.scatter(x, y, alpha=0.55, s=30, color="steelblue", zorder=3)

            # Linear fit
            slope, intercept, r_lin, _, _ = stats.linregress(x, y)
            x_line = np.linspace(x.min(), x.max(), 200)
            ax.plot(x_line, slope * x_line + intercept, "r-",
                    linewidth=1.5, label=f"Linear r={r_lin:.2f}")

            # LOWESS
            from statsmodels.nonparametric.smoothers_lowess import lowess
            smooth = lowess(y, x, frac=0.6, return_sorted=True)
            ax.plot(smooth[:, 0], smooth[:, 1], "g--",
                    linewidth=1.5, label="LOWESS")

            rho, p_s = stats.spearmanr(x, y)
            r2 = r_lin ** 2
            lin_note = "≈ linear" if abs(abs(r_lin) - abs(rho)) < 0.08 else "non-linear"
            ax.set_title(
                f"{SHORT[da]} vs {SHORT[ds]}\n"
                f"r={r_lin:.2f}, ρ={rho:.2f} (p={p_s:.3f})\n"
                f"R²={r2:.2f} — {lin_note}",
                fontsize=8,
            )
            ax.set_xlabel(SHORT[ds], fontsize=8)
            ax.set_ylabel(SHORT[da], fontsize=8)
            ax.legend(fontsize=6)
            ax.tick_params(labelsize=7)

    plt.suptitle(
        f"DASS ↔ DSS Scatter Plots with Linearity Check — {label} (n={len(scores)})",
        fontsize=13, y=1.01,
    )
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()

scatter_grid(scores_before, "Before", f"{OUTPUT_DIR}/fig03_scatter_dass_dss_before.png")
print("[Saved] fig03_scatter_dass_dss_before.png")

scatter_grid(scores_after, "After", f"{OUTPUT_DIR}/fig04_scatter_dass_dss_after.png")
print("[Saved] fig04_scatter_dass_dss_after.png")

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 05 & 06 — Before vs After scatter / distribution comparison
# (independent samples — strip plot + box overlay, per subscale)
# ═══════════════════════════════════════════════════════════════════════════════
def before_after_comparison(cols, title_prefix, fig_path):
    """
    Paired scatter: one point per participant (Before on x, After on y).
    Diagonal = no change. Points above = higher after; below = lower after.
    Also shows a linear OLS fit + LOWESS to check linearity of the
    before→after relationship.
    """
    n = len(cols)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 5), squeeze=False)
    axes = axes[0]

    from statsmodels.nonparametric.smoothers_lowess import lowess

    for ax, col in zip(axes, cols):
        b = scores_before[col].values
        a = scores_after[col].values

        # Identity line (no change)
        all_vals = np.concatenate([b, a])
        lo, hi = all_vals.min() - 0.5, all_vals.max() + 0.5
        ax.plot([lo, hi], [lo, hi], "k:", linewidth=1, alpha=0.5, label="No change")

        # Per-participant lines
        for bv, av in zip(b, a):
            color = "#C55A11" if av > bv else "#2E74B5" if av < bv else "gray"
            ax.plot([bv, av], [bv, av], color=color, linewidth=0.5, alpha=0.3)

        ax.scatter(b, a, alpha=0.7, s=40, color="steelblue", zorder=4)

        # Linear fit
        valid_mask = ~(np.isnan(b) | np.isnan(a))
        bv, av = b[valid_mask], a[valid_mask]
        slope, intercept, r_lin, _, _ = stats.linregress(bv, av)
        x_line = np.linspace(lo, hi, 200)
        ax.plot(x_line, slope * x_line + intercept, "r-", linewidth=1.6,
                label=f"Linear r={r_lin:.2f}")

        # LOWESS
        smooth = lowess(av, bv, frac=0.6, return_sorted=True)
        ax.plot(smooth[:, 0], smooth[:, 1], "g--", linewidth=1.6, label="LOWESS")

        rho, p_s = stats.spearmanr(bv, av)
        lin_note = "≈ linear" if abs(abs(r_lin) - abs(rho)) < 0.08 else "non-linear"
        ax.set_title(
            f"{SHORT[col]}\nr={r_lin:.2f}, ρ={rho:.2f} (p={p_s:.3f})\n{lin_note}",
            fontsize=9,
        )
        ax.set_xlabel("Before Score")
        ax.set_ylabel("After Score")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.legend(fontsize=7)

    plt.suptitle(
        f"{title_prefix} — Paired Scatter: Before vs After (n={N_PAIRED})",
        fontsize=13, y=1.01,
    )
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()

before_after_comparison(
    DASS_COLS,
    "DASS Subscales",
    f"{OUTPUT_DIR}/fig05_scatter_dass_before_after.png",
)
print("[Saved] fig05_scatter_dass_before_after.png")

before_after_comparison(
    DSS_COLS,
    "DSS Subscales",
    f"{OUTPUT_DIR}/fig06_scatter_dss_before_after.png",
)
print("[Saved] fig06_scatter_dss_before_after.png")

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURES 07 & 08 — Radar: Before vs After (Mean and Median) — DASS / DSS
# ═══════════════════════════════════════════════════════════════════════════════
def radar_mean_median(cols, title, fig_path):
    """
    Radar chart showing Before/After Mean and Median for each subscale.
    Values are normalised to [0, 1] using theoretical scale maxima.
    """
    n = len(cols)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    labels = [SHORT[c] for c in cols]

    series = {
        "Before Mean":   scores_before[cols].mean().values,
        "Before Median": scores_before[cols].median().values,
        "After Mean":    scores_after[cols].mean().values,
        "After Median":  scores_after[cols].median().values,
    }
    maxvals = np.array([SCALE_MAX[c] for c in cols])
    series_norm = {k: (v / maxvals).tolist() + [(v / maxvals)[0]] for k, v in series.items()}

    colors  = ["#2E74B5", "#9DC3E6", "#C55A11", "#F4B183"]
    lstyles = ["-", "--", "-", "--"]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"polar": True})

    for (name, vals), col, ls in zip(series_norm.items(), colors, lstyles):
        ax.plot(angles, vals, ls, linewidth=2, color=col, label=name)
        ax.fill(angles, vals, alpha=0.06, color=col)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["25 %", "50 %", "75 %", "100 %"], size=7)
    ax.set_title(title, size=13, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=9)

    plt.tight_layout()
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()

radar_mean_median(
    DASS_COLS,
    "DASS Subscales — Before vs After (normalised to scale max)",
    f"{OUTPUT_DIR}/fig07_radar_dass_before_after.png",
)
print("[Saved] fig07_radar_dass_before_after.png")

radar_mean_median(
    DSS_COLS,
    "DSS Subscales — Before vs After (normalised to scale max)",
    f"{OUTPUT_DIR}/fig08_radar_dss_before_after.png",
)
print("[Saved] fig08_radar_dss_before_after.png")

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURES 09 & 10 — Radar: All Participants (Before / After)
#   One line per participant (thin, semi-transparent) + bold mean overlay.
#   Uses all 7 subscales combined, normalised to [0, 1].
# ═══════════════════════════════════════════════════════════════════════════════
def radar_all_participants(scores, label, fig_path):
    cols     = ALL_COLS
    n        = len(cols)
    labels   = [SHORT[c] for c in cols]
    maxvals  = np.array([SCALE_MAX[c] for c in cols])
    angles   = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles  += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True})

    norm_scores = scores[cols].values / maxvals
    mean_norm   = norm_scores.mean(axis=0)
    med_norm    = np.median(norm_scores, axis=0)

    # Individual participant lines
    alpha_ind = max(0.05, min(0.3, 3.0 / len(scores)))
    for row in norm_scores:
        vals = row.tolist() + [row[0]]
        ax.plot(angles, vals, "-", color="steelblue", linewidth=0.7, alpha=alpha_ind)

    # Mean
    mean_vals = mean_norm.tolist() + [mean_norm[0]]
    ax.plot(angles, mean_vals, "-", color="navy", linewidth=2.5, label="Mean")
    ax.fill(angles, mean_vals, alpha=0.15, color="navy")

    # Median
    med_vals = med_norm.tolist() + [med_norm[0]]
    ax.plot(angles, med_vals, "--", color="darkorange", linewidth=2.2, label="Median")

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["25 %", "50 %", "75 %", "100 %"], size=7)
    ax.set_title(
        f"All Participants — {label} (n={len(scores)})\n"
        f"(thin lines = individual profiles, normalised to scale max)",
        size=12, pad=20
    )
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=10)

    plt.tight_layout()
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()

radar_all_participants(
    scores_before, "Before",
    f"{OUTPUT_DIR}/fig09_radar_all_participants_before.png",
)
print("[Saved] fig09_radar_all_participants_before.png")

radar_all_participants(
    scores_after, "After",
    f"{OUTPUT_DIR}/fig10_radar_all_participants_after.png",
)
print("[Saved] fig10_radar_all_participants_after.png")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 9 — BEFORE vs AFTER STATISTICAL TEST (paired, n=29)
#   Wilcoxon signed-rank test (non-parametric paired test, appropriate given
#   non-normal distributions).
#   Effect size: rank-biserial correlation r = Z / sqrt(n).
#   % change computed on medians and means.
# ═══════════════════════════════════════════════════════════════════════════════
sep(f"BEFORE vs AFTER — WILCOXON SIGNED-RANK TEST (paired, n={N_PAIRED})")

test_rows = []
for col in ALL_COLS:
    b = scores_before[col].values
    a = scores_after[col].values

    # Drop pairs where either value is NaN
    mask = ~(np.isnan(b) | np.isnan(a))
    b_clean, a_clean = b[mask], a[mask]
    n = len(b_clean)

    stat, p = stats.wilcoxon(b_clean, a_clean, alternative="two-sided")
    # Approximate effect size: r = Z / sqrt(n), where Z from normal approx
    z = stats.norm.ppf(p / 2)      # two-tailed z (negative)
    r_effect = abs(z) / np.sqrt(n)

    med_b, med_a = np.median(b_clean), np.median(a_clean)
    pct_med = ((med_a - med_b) / med_b * 100) if med_b != 0 else np.nan

    mean_b, mean_a = b_clean.mean(), a_clean.mean()
    pct_mean = ((mean_a - mean_b) / mean_b * 100) if mean_b != 0 else np.nan

    sig = "**" if p < 0.01 else "*" if p < 0.05 else "n.s."

    test_rows.append({
        "Scale":          col,
        "n":              n,
        "Median Before":  round(med_b, 2),
        "Median After":   round(med_a, 2),
        "% Δ Median":     f"{pct_med:+.1f}%" if not np.isnan(pct_med) else "n/a",
        "Mean Before":    round(mean_b, 2),
        "Mean After":     round(mean_a, 2),
        "% Δ Mean":       f"{pct_mean:+.1f}%" if not np.isnan(pct_mean) else "n/a",
        "W":              round(stat, 1),
        "p-value":        round(p, 4),
        "Sig.":           sig,
        "Effect r":       round(r_effect, 3),
    })

test_df = pd.DataFrame(test_rows).set_index("Scale")
print(test_df.to_string())
print("\n  * p < 0.05  ** p < 0.01  n.s. = not significant")
print("  Effect size r (|Z|/√n): < 0.1 negligible, 0.1–0.3 small, 0.3–0.5 medium, > 0.5 large")

print("\n✓  Analysis complete. All figures saved to:", OUTPUT_DIR)
