#!/usr/bin/env python3
"""
Joint + Population-wise Evo2 7B analysis for Jharkhand chicken (4 populations).
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from collections import defaultdict

print("=" * 70)
print("JOINT & POPULATION-WISE EVO2 ANALYSIS")
print("=" * 70)

# ---------------------------------------------------------------------------
# 1. Load population info
# ---------------------------------------------------------------------------
fam = pd.read_csv('ch_gmmlm.fam', sep='\s+', header=None, 
                  names=['FID', 'IID', 'PID', 'MID', 'SEX', 'PHENO'])
pop_map = dict(zip(fam['IID'], fam['FID']))
print(f"Loaded {len(fam)} individuals from 4 populations: {fam['FID'].unique().tolist()}")

# ---------------------------------------------------------------------------
# 2. Load genotypes (.raw file)
# ---------------------------------------------------------------------------
print("\nLoading genotypes...")
raw = pd.read_csv('ch_gmmlm_genes_only.raw', sep='\s+')
# Columns: FID, IID, PAT, MAT, SEX, PHENO, then SNP columns
meta_cols = ['FID', 'IID', 'PAT', 'MAT', 'SEX', 'PHENO']
snp_cols = [c for c in raw.columns if c not in meta_cols]
print(f"Genotypes: {len(raw)} individuals x {len(snp_cols)} SNPs")

# Clean SNP column names: remove trailing _A/C/G/T allele suffix
snp_names = []
for c in snp_cols:
    # Format is "SNP_ID_A" where A is the counted allele
    parts = c.rsplit('_', 1)
    snp_names.append(parts[0])

# ---------------------------------------------------------------------------
# 3. Load Evo2 scores and match
# ---------------------------------------------------------------------------
print("\nLoading Evo2 scores...")
ref = np.load('snps_ref_1kb_evo2_scores.npz')
alt = np.load('snps_alt_1kb_evo2_scores.npz')

# Build Evo2 score dataframe
evo_df = pd.DataFrame({
    'snp_id': [s.decode() if isinstance(s, bytes) else str(s) for s in ref['seq_ids']],
    'ref_ll': ref['log_likelihood'],
    'alt_ll': alt['log_likelihood'],
    'ref_ppl': ref['perplexity'],
    'alt_ppl': alt['perplexity'],
})
evo_df['delta_ll'] = evo_df['ref_ll'] - evo_df['alt_ll']
evo_df['abs_delta_ll'] = evo_df['delta_ll'].abs()

# Parse SNP IDs to get base name
# Format in evo: "NC_006088.3:12608733|GENE|NC_006088:12608733|REF=A|ALT=G"
evo_df['snp_base'] = evo_df['snp_id'].apply(lambda x: x.split('|')[0])

# Match with PLINK SNP names
# PLINK raw columns: "NC_006088.3:12608733_A" -> base "NC_006088.3:12608733"
plink_bases = snp_names

# Keep only SNPs that are in both datasets
matched = []
for i, plink_name in enumerate(plink_bases):
    match = evo_df[evo_df['snp_base'] == plink_name]
    if len(match) == 1:
        matched.append({
            'plink_col': snp_cols[i],
            'plink_name': plink_name,
            'evo_idx': match.index[0],
            'delta_ll': match['delta_ll'].values[0],
            'abs_delta_ll': match['abs_delta_ll'].values[0],
        })

match_df = pd.DataFrame(matched)
print(f"Matched {len(match_df)} SNPs between genotypes and Evo2 scores")

# ---------------------------------------------------------------------------
# 4. Compute per-individual burden scores
# ---------------------------------------------------------------------------
print("\nComputing per-individual burden scores...")

G = raw[match_df['plink_col']].values  # (n_ind, n_matched_snps)
weights = match_df['delta_ll'].values   # (n_matched_snps,)
abs_weights = match_df['abs_delta_ll'].values

# Handle missing values (PLINK raw uses -9 or NA)
G = np.where(G < 0, np.nan, G)

# Burden definitions:
# 1. Total functional burden: sum(genotype * |delta_LL|) / n_snps
# 2. Deleterious burden (alt is bad): sum(genotype * max(0, -delta_LL)) / n_snps
# 3. Beneficial burden (alt is good): sum(genotype * max(0, delta_LL)) / n_snps
# 4. Absolute burden: sum(genotype * |delta_LL|) / n_snps

n_snps_matched = len(match_df)
n_ind = len(raw)

burden_total = np.nansum(G * abs_weights, axis=1) / n_snps_matched
burden_del = np.nansum(G * np.maximum(0, -weights), axis=1) / n_snps_matched
burden_ben = np.nansum(G * np.maximum(0, weights), axis=1) / n_snps_matched

# Count of missing genotypes per individual
missing_count = np.isnan(G).sum(axis=1)

burden_df = pd.DataFrame({
    'FID': raw['FID'],
    'IID': raw['IID'],
    'population': raw['FID'],
    'total_burden': burden_total,
    'del_burden': burden_del,
    'ben_burden': burden_ben,
    'missing_snps': missing_count,
})

print("\nBurden by population (mean ± std):")
for pop in ['JM', 'JR', 'MA', 'RW']:
    sub = burden_df[burden_df['population'] == pop]
    print(f"  {pop} (n={len(sub)}):")
    print(f"    Total burden:  {sub['total_burden'].mean():.4f} ± {sub['total_burden'].std():.4f}")
    print(f"    Del burden:    {sub['del_burden'].mean():.4f} ± {sub['del_burden'].std():.4f}")
    print(f"    Ben burden:    {sub['ben_burden'].mean():.4f} ± {sub['ben_burden'].std():.4f}")

# ANOVA across populations
for burden_type in ['total_burden', 'del_burden', 'ben_burden']:
    groups = [burden_df[burden_df['population'] == pop][burden_type].values for pop in ['JM', 'JR', 'MA', 'RW']]
    f_stat, p_val = stats.f_oneway(*groups)
    print(f"\nANOVA {burden_type}: F={f_stat:.3f}, p={p_val:.4f}")

# ---------------------------------------------------------------------------
# 5. Population-wise per-SNP analysis
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("POPULATION-WISE PER-SNP ANALYSIS")
print("=" * 70)

pop_codes = {'JM': 0, 'JR': 1, 'MA': 2, 'RW': 3}
pop_array = np.array([pop_codes.get(p, -1) for p in raw['FID']])

# For each SNP, test association with population (ANOVA)
anova_results = []
for i in range(len(match_df)):
    g = G[:, i]
    valid = ~np.isnan(g)
    if valid.sum() < 20:
        continue
    
    groups = [g[valid & (pop_array == code)] for code in range(4)]
    # Filter out empty groups
    groups = [g for g in groups if len(g) > 0]
    if len(groups) < 2:
        continue
    
    f_stat, p_val = stats.f_oneway(*groups)
    anova_results.append({
        'plink_name': match_df.iloc[i]['plink_name'],
        'delta_ll': match_df.iloc[i]['delta_ll'],
        'abs_delta_ll': match_df.iloc[i]['abs_delta_ll'],
        'f_stat': f_stat,
        'p_val': p_val,
    })

anova_df = pd.DataFrame(anova_results)
anova_df['fdr'] = stats.false_discovery_rate(anova_df['p_val'].values) if hasattr(stats, 'false_discovery_rate') else anova_df['p_val'] * len(anova_df) / np.arange(1, len(anova_df)+1)
# Simple Bonferroni instead
anova_df['bonf'] = np.minimum(anova_df['p_val'] * len(anova_df), 1.0)

print(f"Tested {len(anova_df)} SNPs for population association")
print(f"Significant at Bonferroni < 0.05: {(anova_df['bonf'] < 0.05).sum()}")
print(f"Significant at p < 0.01: {(anova_df['p_val'] < 0.01).sum()}")

# Correlation between Evo2 effect and population differentiation
r_p, p_p = stats.pearsonr(anova_df['abs_delta_ll'], -np.log10(anova_df['p_val'] + 1e-300))
r_s, p_s = stats.spearmanr(anova_df['abs_delta_ll'], -np.log10(anova_df['p_val'] + 1e-300))
print(f"\nCorrelation |Delta_LL| vs -log10(p_pop):")
print(f"  Pearson r = {r_p:.4f} (p = {p_p:.4f})")
print(f"  Spearman rho = {r_s:.4f} (p = {p_s:.4f})")

# Top differentiated SNPs
print("\nTop 10 most population-differentiated SNPs:")
top_diff = anova_df.nsmallest(10, 'p_val')[['plink_name', 'delta_ll', 'abs_delta_ll', 'f_stat', 'p_val', 'bonf']]
print(top_diff.to_string(index=False))

# ---------------------------------------------------------------------------
# 6. PCA on 536 SNPs
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("PCA ON 536 FUNCTIONAL SNPs")
print("=" * 70)

# Impute missing with column mean
G_imp = G.copy()
for i in range(G_imp.shape[1]):
    col = G_imp[:, i]
    mask = np.isnan(col)
    if mask.any():
        col[mask] = np.nanmean(col)

pca = PCA(n_components=10)
pcs = pca.fit_transform(G_imp)

print(f"Variance explained by first 4 PCs: {pca.explained_variance_ratio_[:4]}")
print(f"Total variance in first 10 PCs: {pca.explained_variance_ratio_[:10].sum():.3f}")

# ---------------------------------------------------------------------------
# 7. Generate comprehensive plots
# ---------------------------------------------------------------------------
print("\nGenerating plots...")

fig = plt.figure(figsize=(20, 14))
gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)

pop_colors = {'JM': '#e41a1c', 'JR': '#377eb8', 'MA': '#4daf4a', 'RW': '#984ea3'}

# Row 0: Burden boxplots
# 0,0: Total burden
ax = fig.add_subplot(gs[0, 0])
data = [burden_df[burden_df['population'] == pop]['total_burden'].values for pop in ['JM', 'JR', 'MA', 'RW']]
bp = ax.boxplot(data, labels=['JM', 'JR', 'MA', 'RW'], patch_artist=True)
for patch, pop in zip(bp['boxes'], ['JM', 'JR', 'MA', 'RW']):
    patch.set_facecolor(pop_colors[pop])
    patch.set_alpha(0.6)
ax.set_ylabel('Total Burden')
ax.set_title('Total Functional Burden\n(weighted by |ΔLL|)')

# 0,1: Deleterious burden
ax = fig.add_subplot(gs[0, 1])
data = [burden_df[burden_df['population'] == pop]['del_burden'].values for pop in ['JM', 'JR', 'MA', 'RW']]
bp = ax.boxplot(data, labels=['JM', 'JR', 'MA', 'RW'], patch_artist=True)
for patch, pop in zip(bp['boxes'], ['JM', 'JR', 'MA', 'RW']):
    patch.set_facecolor(pop_colors[pop])
    patch.set_alpha(0.6)
ax.set_ylabel('Del Burden')
ax.set_title('Deleterious Burden\n(alt disfavored by Evo2)')

# 0,2: Beneficial burden
ax = fig.add_subplot(gs[0, 2])
data = [burden_df[burden_df['population'] == pop]['ben_burden'].values for pop in ['JM', 'JR', 'MA', 'RW']]
bp = ax.boxplot(data, labels=['JM', 'JR', 'MA', 'RW'], patch_artist=True)
for patch, pop in zip(bp['boxes'], ['JM', 'JR', 'MA', 'RW']):
    patch.set_facecolor(pop_colors[pop])
    patch.set_alpha(0.6)
ax.set_ylabel('Ben Burden')
ax.set_title('Beneficial Burden\n(alt favored by Evo2)')

# 0,3: Burden barplot means
ax = fig.add_subplot(gs[0, 3])
pops = ['JM', 'JR', 'MA', 'RW']
total_means = [burden_df[burden_df['population'] == p]['total_burden'].mean() for p in pops]
total_sems = [burden_df[burden_df['population'] == p]['total_burden'].sem() for p in pops]
ax.bar(pops, total_means, yerr=total_sems, color=[pop_colors[p] for p in pops], alpha=0.7, capsize=5)
ax.set_ylabel('Mean Total Burden')
ax.set_title('Mean Burden ± SEM')

# Row 1: PCA
# 1,0: PC1 vs PC2
ax = fig.add_subplot(gs[1, 0])
for pop in pops:
    mask = burden_df['population'] == pop
    ax.scatter(pcs[mask, 0], pcs[mask, 1], c=pop_colors[pop], label=pop, s=60, alpha=0.7, edgecolors='black', linewidth=0.5)
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
ax.set_title('PCA on 536 Functional SNPs')
ax.legend()

# 1,1: PC3 vs PC4
ax = fig.add_subplot(gs[1, 1])
for pop in pops:
    mask = burden_df['population'] == pop
    ax.scatter(pcs[mask, 2], pcs[mask, 3], c=pop_colors[pop], label=pop, s=60, alpha=0.7, edgecolors='black', linewidth=0.5)
ax.set_xlabel(f'PC3 ({pca.explained_variance_ratio_[2]*100:.1f}%)')
ax.set_ylabel(f'PC4 ({pca.explained_variance_ratio_[3]*100:.1f}%)')
ax.set_title('PCA PC3 vs PC4')

# 1,2: PCA colored by total burden
ax = fig.add_subplot(gs[1, 2])
sc = ax.scatter(pcs[:, 0], pcs[:, 1], c=burden_df['total_burden'], cmap='viridis', s=60, alpha=0.8, edgecolors='black', linewidth=0.5)
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
ax.set_title('PCA colored by Total Burden')
plt.colorbar(sc, ax=ax)

# 1,3: Scree plot
ax = fig.add_subplot(gs[1, 3])
ax.bar(range(1, 11), pca.explained_variance_ratio_[:10] * 100, color='steelblue', alpha=0.7)
ax.set_xlabel('PC')
ax.set_ylabel('% Variance')
ax.set_title('Scree Plot')

# Row 2: Population-wise MAF and differentiation
# 2,0: Population-wise MAF spectra of high-effect SNPs
ax = fig.add_subplot(gs[2, 0])
# For each pop, histogram of MAF for SNPs with |delta_LL| > 3
high_effect_mask = match_df['abs_delta_ll'] > 3
for pop in pops:
    pop_idx = raw['FID'] == pop
    pop_g = G[pop_idx][:, high_effect_mask.values]
    # Compute MAF per SNP for this population
    mafs = []
    for j in range(pop_g.shape[1]):
        col = pop_g[:, j]
        col = col[~np.isnan(col)]
        if len(col) > 0:
            maf = col.mean() / 2  # genotype is 0,1,2; allele freq = mean/2
            mafs.append(maf)
    ax.hist(mafs, bins=15, alpha=0.4, label=pop, color=pop_colors[pop])
ax.set_xlabel('MAF')
ax.set_ylabel('Count')
ax.set_title('MAF Spectrum (|ΔLL| > 3 SNPs)')
ax.legend()

# 2,1: Evo2 effect vs population differentiation
ax = fig.add_subplot(gs[2, 1])
ax.scatter(anova_df['abs_delta_ll'], -np.log10(anova_df['p_val'] + 1e-300), c='steelblue', s=25, alpha=0.5)
ax.set_xlabel('|Delta LL| (Evo2 effect)')
ax.set_ylabel('-log10(p) Population ANOVA')
ax.set_title(f'Evo2 Effect vs Population Diff\nSpearman rho = {r_s:.3f}')
ax.axhline(-np.log10(0.05), color='red', linestyle='--', lw=1)

# 2,2: Per-population mean genotype of top deleterious SNPs
ax = fig.add_subplot(gs[2, 2])
top_del_idx = match_df.nsmallest(20, 'delta_ll').index
x_pos = np.arange(20)
width = 0.2
for i, pop in enumerate(pops):
    pop_idx = raw['FID'] == pop
    means = []
    for idx in top_del_idx:
        col = G[pop_idx, idx]
        means.append(np.nanmean(col))
    ax.bar(x_pos + (i - 1.5) * width, means, width, label=pop, color=pop_colors[pop], alpha=0.8)
ax.set_xticks(x_pos)
ax.set_xticklabels([match_df.iloc[i]['plink_name'].split(':')[1] for i in top_del_idx], rotation=90, fontsize=6)
ax.set_ylabel('Mean Genotype')
ax.set_title('Mean Genotype: Top 20 Del SNPs')
ax.legend(fontsize=8)

# 2,3: Population burden violin plot
ax = fig.add_subplot(gs[2, 3])
data = [burden_df[burden_df['population'] == pop]['total_burden'].values for pop in pops]
parts = ax.violinplot(data, positions=range(4), showmeans=True, showmedians=True)
for i, pc in enumerate(parts['bodies']):
    pc.set_facecolor(pop_colors[pops[i]])
    pc.set_alpha(0.6)
ax.set_xticks(range(4))
ax.set_xticklabels(pops)
ax.set_ylabel('Total Burden')
ax.set_title('Burden Distribution by Population')

fig.suptitle('Joint & Population-wise Evo2 7B Analysis - Jharkhand Chicken (4 Pops)', fontsize=16, fontweight='bold', y=0.98)
plt.savefig('evo2_joint_population_analysis.png', dpi=200, bbox_inches='tight')
print("Saved: evo2_joint_population_analysis.png")

# ---------------------------------------------------------------------------
# 8. Population-wise top burden individuals
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("TOP BURDEN INDIVIDUALS PER POPULATION")
print("=" * 70)
for pop in pops:
    sub = burden_df[burden_df['population'] == pop].sort_values('total_burden', ascending=False)
    print(f"\n{pop} - Highest burden:")
    print(sub.head(3)[['IID', 'total_burden', 'del_burden', 'ben_burden']].to_string(index=False))
    print(f"{pop} - Lowest burden:")
    print(sub.tail(3)[['IID', 'total_burden', 'del_burden', 'ben_burden']].to_string(index=False))

# ---------------------------------------------------------------------------
# 9. Save tables
# ---------------------------------------------------------------------------
burden_df.to_csv('evo2_individual_burden.csv', index=False)
anova_df.to_csv('evo2_snp_population_anova.csv', index=False)
print("\nSaved: evo2_individual_burden.csv")
print("Saved: evo2_snp_population_anova.csv")

print("\nAnalysis complete!")
