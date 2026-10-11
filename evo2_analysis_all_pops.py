#!/usr/bin/env python3
"""
Evo2 7B analysis for Jharkhand chicken - ALL 4 populations (JM, JR, MA, RW).
Focus: 536 biallelic SNPs within 94 landscape-genomics enriched genes.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.stats import pearsonr, spearmanr, f_oneway

print("=" * 60)
print("Loading Evo2 scores...")
ref = np.load('snps_ref_1kb_evo2_scores.npz')
alt = np.load('snps_alt_1kb_evo2_scores.npz')

seq_ids = [s.decode() if isinstance(s, bytes) else str(s) for s in ref['seq_ids']]
n_snps = len(seq_ids)

snp_info = []
for sid in seq_ids:
    parts = sid.split('|')
    snp_id = parts[0]
    gene = parts[1]
    chrom_pos = parts[2]
    ref_allele = parts[3].replace('REF=', '')
    alt_allele = parts[4].replace('ALT=', '')
    chrom, pos = chrom_pos.split(':')
    snp_info.append({
        'snp_id': snp_id,
        'gene': gene,
        'chrom': chrom,
        'pos': int(pos),
        'ref': ref_allele,
        'alt': alt_allele,
    })

df = pd.DataFrame(snp_info)
df['ref_ll'] = ref['log_likelihood']
df['alt_ll'] = alt['log_likelihood']
df['ref_ppl'] = ref['perplexity']
df['alt_ppl'] = alt['perplexity']
df['ref_allele_prob'] = ref['snp_allele_prob']
df['alt_allele_prob'] = alt['snp_allele_prob']

df['delta_ll'] = df['ref_ll'] - df['alt_ll']
df['delta_ppl'] = df['ref_ppl'] - df['alt_ppl']
df['delta_allele_prob'] = df['ref_allele_prob'] - df['alt_allele_prob']
df['abs_delta_ll'] = df['delta_ll'].abs()

print(f"Loaded {n_snps} SNPs across {df['gene'].nunique()} genes")

# ---------------------------------------------------------------------------
# Load all 4 population frequencies
# ---------------------------------------------------------------------------
print("\nLoading PLINK frequencies for all populations...")

def parse_frq(path):
    frq = pd.read_csv(path, sep=r'\s+', engine='python')
    frq.columns = [c.upper() for c in frq.columns]
    return frq

pops = {}
for pop in ['jm', 'jr', 'ma', 'rw']:
    pops[pop] = parse_frq(f'{pop}_freq.frq')

# Merge all
freq_all = pops['jm'][['SNP', 'MAF']].rename(columns={'MAF': 'MAF_JM'})
for pop in ['jr', 'ma', 'rw']:
    freq_all = pd.merge(freq_all, pops[pop][['SNP', 'MAF']].rename(columns={'MAF': f'MAF_{pop.upper()}'}), on='SNP')

df = pd.merge(df, freq_all, left_on='snp_id', right_on='SNP', how='left')

# Population differentiation metrics
df['maf_mean'] = df[['MAF_JM', 'MAF_JR', 'MAF_MA', 'MAF_RW']].mean(axis=1)
df['maf_std'] = df[['MAF_JM', 'MAF_JR', 'MAF_MA', 'MAF_RW']].std(axis=1)
df['maf_range'] = df[['MAF_JM', 'MAF_JR', 'MAF_MA', 'MAF_RW']].max(axis=1) - df[['MAF_JM', 'MAF_JR', 'MAF_MA', 'MAF_RW']].min(axis=1)

print(f"Matched frequencies for all 4 pops: {df['MAF_JM'].notna().sum()} SNPs")

# ---------------------------------------------------------------------------
# Summary stats
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("SUMMARY STATISTICS")
print("=" * 60)

print(f"\nDelta Log-Likelihood (Ref - Alt):")
print(f"  Mean: {df['delta_ll'].mean():.3f} | Std: {df['delta_ll'].std():.3f}")
print(f"  Min:  {df['delta_ll'].min():.3f} | Max: {df['delta_ll'].max():.3f}")

# ---------------------------------------------------------------------------
# Population differentiation
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("POPULATION DIFFERENTIATION vs EVO2 EFFECT")
print("=" * 60)

valid = df.dropna(subset=['MAF_JM', 'MAF_JR', 'MAF_MA', 'MAF_RW'])

for metric in ['maf_std', 'maf_range']:
    r_p, p_p = pearsonr(valid['abs_delta_ll'], valid[metric])
    r_s, p_s = spearmanr(valid['abs_delta_ll'], valid[metric])
    print(f"\n|Delta_LL| vs {metric}:")
    print(f"  Pearson r = {r_p:.4f} (p = {p_p:.4f})")
    print(f"  Spearman rho = {r_s:.4f} (p = {p_s:.4f})")

# ---------------------------------------------------------------------------
# Top variants table
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("TOP 10 DELETERIOUS ALT VARIANTS")
print("=" * 60)
print(df.nsmallest(10, 'delta_ll')[['gene', 'chrom', 'pos', 'ref', 'alt', 'delta_ll', 'MAF_JM', 'MAF_JR', 'MAF_MA', 'MAF_RW']].to_string(index=False))

print("\n" + "=" * 60)
print("TOP 10 BENEFICIAL ALT VARIANTS")
print("=" * 60)
print(df.nlargest(10, 'delta_ll')[['gene', 'chrom', 'pos', 'ref', 'alt', 'delta_ll', 'MAF_JM', 'MAF_JR', 'MAF_MA', 'MAF_RW']].to_string(index=False))

# ---------------------------------------------------------------------------
# Gene-level stats
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("GENE-LEVEL STATS (top 15 by max |effect|)")
print("=" * 60)

gene_stats = df.groupby('gene').agg(
    n_snps=('snp_id', 'count'),
    mean_delta_ll=('delta_ll', 'mean'),
    max_abs_delta_ll=('abs_delta_ll', 'max'),
    std_delta_ll=('delta_ll', 'std'),
).sort_values('max_abs_delta_ll', ascending=False)

print(gene_stats.head(15).to_string())

# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Generating plots...")
print("=" * 60)

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle('Evo2 7B Variant Effect Analysis - Jharkhand Chicken (4 Populations)', fontsize=14, fontweight='bold')

# 1. Histogram of delta LL
ax = axes[0, 0]
ax.hist(df['delta_ll'], bins=50, color='steelblue', edgecolor='white', alpha=0.8)
ax.axvline(0, color='red', linestyle='--', linewidth=1)
ax.set_xlabel('Delta LL (Ref - Alt)')
ax.set_ylabel('Count')
ax.set_title('Distribution of Variant Effects')

# 2. Ref vs Alt perplexity
ax = axes[0, 1]
sc = ax.scatter(df['ref_ppl'], df['alt_ppl'], c=df['delta_ll'], cmap='RdBu_r', s=25, alpha=0.6, vmin=-10, vmax=10)
ax.plot([1.5, 4], [1.5, 4], 'k--', lw=1)
ax.set_xlabel('Ref Perplexity')
ax.set_ylabel('Alt Perplexity')
ax.set_title('Ref vs Alt Perplexity')
plt.colorbar(sc, ax=ax, label='Delta LL')

# 3. Top genes by max effect
ax = axes[0, 2]
top_genes = gene_stats.head(15)
colors = ['darkred' if x > 0 else 'darkblue' for x in top_genes['mean_delta_ll']]
ax.barh(range(len(top_genes)), top_genes['max_abs_delta_ll'], color=colors, alpha=0.7)
ax.set_yticks(range(len(top_genes)))
ax.set_yticklabels(top_genes.index, fontsize=8)
ax.set_xlabel('Max |Delta LL|')
ax.set_title('Top Genes by Max Variant Effect')
ax.invert_yaxis()

# 4. MAF across 4 populations for top effect SNPs
ax = axes[1, 0]
top20_idx = df.nlargest(20, 'abs_delta_ll').index
x_pos = np.arange(len(top20_idx))
width = 0.2
ax.bar(x_pos - 1.5*width, df.loc[top20_idx, 'MAF_JM'], width, label='JM', alpha=0.8)
ax.bar(x_pos - 0.5*width, df.loc[top20_idx, 'MAF_JR'], width, label='JR', alpha=0.8)
ax.bar(x_pos + 0.5*width, df.loc[top20_idx, 'MAF_MA'], width, label='MA', alpha=0.8)
ax.bar(x_pos + 1.5*width, df.loc[top20_idx, 'MAF_RW'], width, label='RW', alpha=0.8)
ax.set_xticks(x_pos)
ax.set_xticklabels([df.loc[i, 'gene'] for i in top20_idx], rotation=90, fontsize=7)
ax.set_ylabel('MAF')
ax.set_title('MAF across 4 Pops (Top 20 |Effect| SNPs)')
ax.legend(loc='upper right', fontsize=8)

# 5. Population differentiation vs Evo2 effect
ax = axes[1, 1]
ax.scatter(valid['abs_delta_ll'], valid['maf_std'], c='steelblue', s=25, alpha=0.5, label='MAF std')
ax.set_xlabel('|Delta LL| (Evo2 effect)')
ax.set_ylabel('MAF Standard Deviation (4 pops)')
ax.set_title(f'Pop Differentiation vs Evo2 Effect\nSpearman rho = {r_s:.3f}')

# 6. Boxplot: delta LL by gene (top 10)
ax = axes[1, 2]
top10_genes = gene_stats.head(10).index
gene_data = [df[df['gene'] == g]['delta_ll'].values for g in top10_genes]
bp = ax.boxplot(gene_data, vert=False, patch_artist=True)
for patch in bp['boxes']:
    patch.set_facecolor('steelblue')
    patch.set_alpha(0.6)
ax.set_yticklabels(top10_genes, fontsize=8)
ax.set_xlabel('Delta LL')
ax.set_title('Variant Effect by Gene')
ax.axvline(0, color='red', linestyle='--', lw=1)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('evo2_variant_analysis_4pops.png', dpi=200, bbox_inches='tight')
print("Saved: evo2_variant_analysis_4pops.png")

# ---------------------------------------------------------------------------
# Additional plot: Population structure of top SNPs
# ---------------------------------------------------------------------------
fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
fig2.suptitle('Population Patterns of Evo2-Scored SNPs', fontsize=13, fontweight='bold')

# PCA-like plot: JM vs JR colored by effect
ax = axes2[0]
scatter = ax.scatter(valid['MAF_JM'], valid['MAF_JR'], c=valid['delta_ll'], cmap='RdBu_r', s=30, alpha=0.6, vmin=-10, vmax=10)
ax.plot([0, 0.5], [0, 0.5], 'k--', lw=1)
ax.set_xlabel('MAF JM')
ax.set_ylabel('MAF JR')
ax.set_title('JM vs JR MAF (colored by Delta LL)')
plt.colorbar(scatter, ax=ax, label='Delta LL')

# MA vs RW
ax = axes2[1]
scatter = ax.scatter(valid['MAF_MA'], valid['MAF_RW'], c=valid['delta_ll'], cmap='RdBu_r', s=30, alpha=0.6, vmin=-10, vmax=10)
ax.plot([0, 0.5], [0, 0.5], 'k--', lw=1)
ax.set_xlabel('MAF MA')
ax.set_ylabel('MAF RW')
ax.set_title('MA vs RW MAF (colored by Delta LL)')
plt.colorbar(scatter, ax=ax, label='Delta LL')

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('evo2_population_maf.png', dpi=200, bbox_inches='tight')
print("Saved: evo2_population_maf.png")

# ---------------------------------------------------------------------------
# Save tables
# ---------------------------------------------------------------------------
df.to_csv('evo2_variant_scores_4pops.csv', index=False)
print("Saved: evo2_variant_scores_4pops.csv")

gene_stats.to_csv('evo2_gene_level_stats_4pops.csv')
print("Saved: evo2_gene_level_stats_4pops.csv")

# ---------------------------------------------------------------------------
# Population-specific high-effect variants
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("POPULATION-SPECIFIC PATTERNS")
print("=" * 60)

# SNPs where one population is fixed or nearly fixed for alt, others have ref
for pop in ['JM', 'JR', 'MA', 'RW']:
    maf_col = f'MAF_{pop}'
    # High effect + population-specific (MAF > 0.3 in this pop, < 0.1 in others)
    other_pops = [f'MAF_{p}' for p in ['JM', 'JR', 'MA', 'RW'] if p != pop]
    mask = (df[maf_col] > 0.3) & (df[other_pops].max(axis=1) < 0.15) & (df['abs_delta_ll'] > 3)
    hits = df[mask]
    print(f"\n{pop}-enriched high-effect variants: {len(hits)}")
    if len(hits) > 0:
        print(hits[['gene', 'chrom', 'pos', 'ref', 'alt', 'delta_ll', maf_col] + other_pops].to_string(index=False))

print("\nAnalysis complete!")
