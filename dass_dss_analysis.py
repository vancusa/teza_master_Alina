"""
DASS-21 & DSS-24 Statistical Analysis
--------------------------------------
Input:  CSV with columns as exported from Google Forms
Output: Console summary + PNG figures saved to working directory
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pingouin as pg
import warnings
warnings.filterwarnings("ignore")

# ── CONFIG ────────────────────────────────────────────────────────────────────
CSV_PATH = "date_alina_in.csv"          # ← change to your actual filename
OUTPUT_DIR = "."               # figures saved here

# DASS-21 subscale item indices (1-based, matching column order in CSV cols 1-21)
DASS_STRESS     = [1, 6, 8, 11, 12, 14, 18]
DASS_ANXIETY    = [2, 4, 7, 9, 15, 19, 20]
DASS_DEPRESSION = [3, 5, 10, 13, 16, 17, 21]

# DSS-24 subscale item indices (1-based, matching column order in CSV cols 22-45)
# Standard DSS-24 subscales:
DSS_AVAILABILITY   = [1, 4, 7, 8, 12, 16, 18]   # Social availability pressure
DSS_FOMO           = [5, 10, 13, 21]              # Fear of missing out
DSS_APPROVAL       = [3, 9, 17, 20, 22, 24]       # Approval anxiety
DSS_OVERLOAD       = [2, 6, 11, 14, 15, 19, 23]   # Information/notification overload

# ── LOAD DATA ─────────────────────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH)
df.columns = df.columns.str.strip()

print("=== DEBUG ===")
print(f"Total CSV columns: {len(df.columns)}")
for i, col in enumerate(df.columns):
    print(f"  [{i}] {col[:60]}")
print("=== END DEBUG ===")

# Identify item columns by position (columns 1-21 = DASS, 22-45 = DSS)
# Column 0 = Timestamp, then 21 DASS items, then 24 DSS items, then Score1, Score2, Hour, Minute
dass_cols = df.columns[1:22].tolist()    # 21 items
dss_cols  = df.columns[22:46].tolist()   # 24 items

# Extract item matrices (0-based indexing into lists)
dass_items = df[dass_cols].apply(pd.to_numeric, errors='coerce')
dss_items  = df[dss_cols].apply(pd.to_numeric, errors='coerce')

# Use pre-computed totals if available, else compute
if 'Score1' in df.columns and 'Score 2' in df.columns:
    df['DASS_total'] = pd.to_numeric(df['Score1'], errors='coerce')
    df['DSS_total']  = pd.to_numeric(df['Score 2'], errors='coerce')
else:
    df['DASS_total'] = dass_items.sum(axis=1)
    df['DSS_total']  = dss_items.sum(axis=1)

dass_item_list = dass_items.columns.tolist()
dss_item_list  = dss_items.columns.tolist()

dass_items_adj = dass_items - 1  # convert 1–4 → 0–3

df['DASS_stress']     = dass_items_adj[[dass_item_list[i-1] for i in DASS_STRESS]].sum(axis=1)
df['DASS_anxiety']    = dass_items_adj[[dass_item_list[i-1] for i in DASS_ANXIETY]].sum(axis=1)
df['DASS_depression'] = dass_items_adj[[dass_item_list[i-1] for i in DASS_DEPRESSION]].sum(axis=1)


# Compute DASS subscales (multiply by 2 per standard DASS-21 scoring)
#df['DASS_stress']      = dass_items[[dass_item_list[i-1] for i in DASS_STRESS]].sum(axis=1)
#df['DASS_anxiety']     = dass_items[[dass_item_list[i-1] for i in DASS_ANXIETY]].sum(axis=1)
#df['DASS_depression']  = dass_items[[dass_item_list[i-1] for i in DASS_DEPRESSION]].sum(axis=1)

# Compute DSS subscales
df['DSS_availability'] = dss_items[[dss_item_list[i-1] for i in DSS_AVAILABILITY]].sum(axis=1)
df['DSS_fomo']         = dss_items[[dss_item_list[i-1] for i in DSS_FOMO]].sum(axis=1)
df['DSS_approval']     = dss_items[[dss_item_list[i-1] for i in DSS_APPROVAL]].sum(axis=1)
df['DSS_overload']     = dss_items[[dss_item_list[i-1] for i in DSS_OVERLOAD]].sum(axis=1)

# Time of day
df['Hour']   = pd.to_numeric(df['Hour'], errors='coerce')
df['Minute'] = pd.to_numeric(df['Minute'], errors='coerce')
df['time_decimal'] = df['Hour'] + df['Minute'] / 60  # e.g., 9.05 for 9:03

# Time block classification
def time_block(h):
    if pd.isna(h): return np.nan
    if 5 <= h < 12:  return 'Morning (5–12)'
    if 12 <= h < 17: return 'Afternoon (12–17)'
    if 17 <= h < 21: return 'Evening (17–21)'
    return 'Night (21–5)'

df['time_block'] = df['Hour'].apply(time_block)

# ── 1. DESCRIPTIVE STATISTICS ─────────────────────────────────────────────────
print("=" * 60)
print("1. DESCRIPTIVE STATISTICS")
print("=" * 60)

desc_cols = ['DASS_total', 'DASS_depression', 'DASS_anxiety', 'DASS_stress',
             'DSS_total', 'DSS_availability', 'DSS_fomo', 'DSS_approval', 'DSS_overload']

desc = df[desc_cols].describe().T[['mean', 'std', '50%', 'min', 'max']]
desc.columns = ['Mean', 'SD', 'Median', 'Min', 'Max']
desc = desc.round(2)
print(desc.to_string())

print(f"\nTime block distribution:\n{df['time_block'].value_counts().to_string()}")

# ── 2. INTERNAL CONSISTENCY (Cronbach's α) ───────────────────────────────────
print("\n" + "=" * 60)
print("2. INTERNAL CONSISTENCY (Cronbach's α)")
print("=" * 60)

def cronbach_alpha(item_df):
    """Compute Cronbach's alpha from item matrix."""
    item_df = item_df.dropna()
    k = item_df.shape[1]
    item_vars = item_df.var(axis=0, ddof=1)
    total_var = item_df.sum(axis=1).var(ddof=1)
    return (k / (k - 1)) * (1 - item_vars.sum() / total_var)

scales = {
    'DASS-21 Total':    dass_items,
    'DASS Stress':      dass_items.iloc[:, [i-1 for i in DASS_STRESS]],
    'DASS Anxiety':     dass_items.iloc[:, [i-1 for i in DASS_ANXIETY]],
    'DASS Depression':  dass_items.iloc[:, [i-1 for i in DASS_DEPRESSION]],
    'DSS-24 Total':     dss_items,
    'DSS Availability': dss_items.iloc[:, [i-1 for i in DSS_AVAILABILITY]],
    'DSS FOMO':         dss_items.iloc[:, [i-1 for i in DSS_FOMO]],
    'DSS Approval':     dss_items.iloc[:, [i-1 for i in DSS_APPROVAL]],
    'DSS Overload':     dss_items.iloc[:, [i-1 for i in DSS_OVERLOAD]],
}

for name, items in scales.items():
    alpha = cronbach_alpha(items)
    interp = "excellent" if alpha >= 0.9 else "good" if alpha >= 0.8 else \
             "acceptable" if alpha >= 0.7 else "questionable" if alpha >= 0.6 else "poor"
    print(f"  {name:<22} α = {alpha:.3f}  ({interp})")

# ── 3. NORMALITY CHECK ────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("3. NORMALITY (Shapiro-Wilk, α=0.05)")
print("=" * 60)

for col in ['DASS_total', 'DSS_total']:
    stat, p = stats.shapiro(df[col].dropna())
    normal = "Normal" if p > 0.05 else "Non-normal"
    print(f"  {col:<15}  W={stat:.3f}, p={p:.4f}  → {normal}")

# ── 4. CIRCULAR CORRELATION: TIME OF DAY vs SCORES ───────────────────────────
print("\n" + "=" * 60)
print("4. TIME-OF-DAY CORRELATION (circular statistics)")
print("=" * 60)
print("  [Time is circular: midnight wraps around, so we use")
print("   sine/cosine components of the 24h cycle]\n")

# Convert time to radians (24h cycle)
df['time_rad'] = df['time_decimal'] * (2 * np.pi / 24)
df['time_sin'] = np.sin(df['time_rad'])
df['time_cos'] = np.cos(df['time_rad'])

score_cols = ['DASS_total', 'DASS_depression', 'DASS_anxiety', 'DASS_stress',
              'DSS_total', 'DSS_availability', 'DSS_fomo', 'DSS_approval', 'DSS_overload']

print(f"  {'Scale':<22} {'rho_sin':>8} {'p_sin':>8} {'rho_cos':>8} {'p_cos':>8}")
print(f"  {'-'*22} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

for col in score_cols:
    valid = df[['time_sin', 'time_cos', col]].dropna()
    r_sin, p_sin = stats.spearmanr(valid['time_sin'], valid[col])
    r_cos, p_cos = stats.spearmanr(valid['time_cos'], valid[col])
    sig_sin = " *" if p_sin < 0.05 else ""
    sig_cos = " *" if p_cos < 0.05 else ""
    print(f"  {col:<22} {r_sin:>8.3f} {p_sin:>8.4f}{sig_sin:<2} {r_cos:>8.3f} {p_cos:>8.4f}{sig_cos}")

print("\n  * p < 0.05")

# ── 5. TIME BLOCK GROUP COMPARISON ───────────────────────────────────────────
print("\n" + "=" * 60)
print("5. SCORE BY TIME BLOCK (Kruskal-Wallis)")
print("=" * 60)

for col in ['DASS_total', 'DSS_total']:
    groups = [g[col].dropna().values for _, g in df.groupby('time_block') if len(g) > 0]
    if len(groups) >= 2:
        stat, p = stats.kruskal(*groups)
        print(f"\n  {col}:")
        print(f"    Kruskal-Wallis H={stat:.3f}, p={p:.4f}")
        for block, g in df.groupby('time_block'):
            vals = g[col].dropna()
            print(f"    {block:<22} n={len(vals)}  median={vals.median():.1f}  IQR={vals.quantile(0.25):.1f}–{vals.quantile(0.75):.1f}")

# ── 6. DASS-21 vs DSS-24 CORRELATION ─────────────────────────────────────────
print("\n" + "=" * 60)
print("6. DASS-21 ↔ DSS-24 CORRELATIONS (Spearman)")
print("=" * 60)

dass_scales = ['DASS_total', 'DASS_depression', 'DASS_anxiety', 'DASS_stress']
dss_scales  = ['DSS_total', 'DSS_availability', 'DSS_fomo', 'DSS_approval', 'DSS_overload']

print(f"\n  {'':>22}", end="")
for d in dss_scales:
    print(f"  {d.replace('DSS_','DSS-'):>14}", end="")
print()

for da in dass_scales:
    print(f"  {da.replace('DASS_','DASS-'):<22}", end="")
    for ds in dss_scales:
        valid = df[[da, ds]].dropna()
        r, p = stats.spearmanr(valid[da], valid[ds])
        sig = "*" if p < 0.05 else " "
        print(f"  {r:>6.3f}{sig} (p={p:.3f})", end="")
    print()

print("\n  * p < 0.05")

# ── 7. OUTLIER DETECTION ──────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("7. OUTLIER DETECTION (Z-score |z| > 2.5 on either scale)")
print("=" * 60)

df['z_dass'] = stats.zscore(df['DASS_total'].fillna(df['DASS_total'].mean()))
df['z_dss']  = stats.zscore(df['DSS_total'].fillna(df['DSS_total'].mean()))
outliers = df[(df['z_dass'].abs() > 2.5) | (df['z_dss'].abs() > 2.5)]

if len(outliers) == 0:
    print("  No outliers detected.")
else:
    print(f"  {len(outliers)} outlier(s) found:")
    for idx, row in outliers.iterrows():
        print(f"  Row {idx}: DASS={row['DASS_total']:.0f} (z={row['z_dass']:.2f}), "
              f"DSS={row['DSS_total']:.0f} (z={row['z_dss']:.2f}), Hour={row['Hour']:.0f}")

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURES
# ═══════════════════════════════════════════════════════════════════════════════

sns.set_theme(style="whitegrid", palette="muted")

# ── FIG 1: Distributions ──────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, col, label in zip(axes, ['DASS_total', 'DSS_total'], ['DASS-21 Total Score', 'DSS-24 Total Score']):
    sns.histplot(df[col].dropna(), kde=True, ax=ax, color='steelblue')
    ax.axvline(df[col].median(), color='red', linestyle='--', label=f'Median={df[col].median():.1f}')
    ax.set_title(label)
    ax.legend()
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig1_distributions.png", dpi=150)
plt.close()
print("\n[Saved] fig1_distributions.png")

# ── FIG 2: Scatter DASS vs DSS with regression line ──────────────────────────
fig, ax = plt.subplots(figsize=(6, 5))
sns.regplot(data=df, x='DSS_total', y='DASS_total', ax=ax,
            scatter_kws={'alpha': 0.6}, line_kws={'color': 'red'})
r, p = stats.spearmanr(df['DSS_total'].dropna(), df['DASS_total'].dropna())
ax.set_title(f'DASS-21 vs DSS-24  (ρ={r:.3f}, p={p:.4f})')
ax.set_xlabel('DSS-24 Total Score')
ax.set_ylabel('DASS-21 Total Score')
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig2_scatter_dass_dss.png", dpi=150)
plt.close()
print("[Saved] fig2_scatter_dass_dss.png")

# ── FIG 3: Subscale correlation heatmap ──────────────────────────────────────
corr_cols = dass_scales + dss_scales
corr_matrix = pd.DataFrame(index=dass_scales, columns=dss_scales, dtype=float)
pval_matrix = pd.DataFrame(index=dass_scales, columns=dss_scales, dtype=float)

for da in dass_scales:
    for ds in dss_scales:
        valid = df[[da, ds]].dropna()
        r, p = stats.spearmanr(valid[da], valid[ds])
        corr_matrix.loc[da, ds] = r
        pval_matrix.loc[da, ds] = p

fig, ax = plt.subplots(figsize=(8, 5))
mask = pval_matrix.astype(float) > 0.05
sns.heatmap(corr_matrix.astype(float), annot=True, fmt='.2f', cmap='RdBu_r',
            vmin=-1, vmax=1, ax=ax, linewidths=0.5,
            xticklabels=[c.replace('DSS_', 'DSS\n') for c in dss_scales],
            yticklabels=[c.replace('DASS_', 'DASS\n') for c in dass_scales])

# Mark non-significant cells
for i, da in enumerate(dass_scales):
    for j, ds in enumerate(dss_scales):
        if pval_matrix.loc[da, ds] > 0.05:
            ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False,
                         edgecolor='gray', lw=2, linestyle='--'))

ax.set_title('Spearman Correlations: DASS-21 vs DSS-24 Subscales\n(dashed border = p > 0.05)')
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig3_subscale_heatmap.png", dpi=150)
plt.close()
print("[Saved] fig3_subscale_heatmap.png")

# ── FIG 4: Scores by time of day (scatter over 24h) ──────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, col, label in zip(axes, ['DASS_total', 'DSS_total'], ['DASS-21', 'DSS-24']):
    valid = df[['time_decimal', col]].dropna()
    ax.scatter(valid['time_decimal'], valid[col], alpha=0.6, color='steelblue')
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel(f'{label} Total Score')
    ax.set_title(f'{label} Score by Time of Day')
    ax.set_xticks(range(0, 25, 3))
    ax.set_xticklabels([f'{h:02d}:00' for h in range(0, 25, 3)], rotation=45)
    # Add smoothed trend
    if len(valid) >= 5:
        from scipy.ndimage import uniform_filter1d
        sorted_v = valid.sort_values('time_decimal')
        smoothed = uniform_filter1d(sorted_v[col].values.astype(float), size=5)
        ax.plot(sorted_v['time_decimal'], smoothed, color='red', alpha=0.6, linewidth=2, label='Trend')
        ax.legend()
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig4_scores_by_time.png", dpi=150)
plt.close()
print("[Saved] fig4_scores_by_time.png")

# ── FIG 5: Radar chart of mean subscale scores ────────────────────────────────
subscale_labels = ['DASS\nDepression', 'DASS\nAnxiety', 'DASS\nStress',
                   'DSS\nAvailability', 'DSS\nFOMO', 'DSS\nApproval', 'DSS\nOverload']
subscale_cols = ['DASS_depression', 'DASS_anxiety', 'DASS_stress',
                 'DSS_availability', 'DSS_fomo', 'DSS_approval', 'DSS_overload']

means = df[subscale_cols].mean().values
# Normalize each to 0-1 range for fair comparison across scales
#maxvals = df[subscale_cols].max().values
#maxvals = np.array([42, 42, 42, 28, 16, 24, 28])  # theoretical maxima per subscale
maxvals = np.array([21, 21, 21, 28, 16, 24, 28])
means_norm = means / maxvals

N = len(subscale_labels)
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]
means_norm_plot = list(means_norm) + [means_norm[0]]

fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
ax.plot(angles, means_norm_plot, 'o-', linewidth=2, color='steelblue')
ax.fill(angles, means_norm_plot, alpha=0.25, color='steelblue')
ax.set_xticks(angles[:-1])
ax.set_xticklabels(subscale_labels, size=9)
ax.set_ylim(0, 1)
ax.set_yticks([0.25, 0.5, 0.75, 1.0])
ax.set_yticklabels(['25%', '50%', '75%', '100%'], size=7)
ax.set_title('Subscale Profile\n(normalized to scale max)', size=12, pad=15)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig5_radar_subscales.png", dpi=150)
plt.close()
print("[Saved] fig5_radar_subscales.png")

print("\n✓ Analysis complete.")
