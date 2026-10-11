#!/usr/bin/env python3
"""
Comprehensive Evo2 7B analysis for Jharkhand chicken landscape genomics SNPs.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict

# ---------------------------------------------------------------------------
# 1. Load Evo2 scores
# ---------------------------------------------------------------------------
print("=" * 60)
print("Loading Evo2 scores...")
ref = np.load('snps_ref_1kb_evo2_scores.npz')
alt = np.load('snps_alt_1kb_evo2_scores.npz')

seq_ids = [s.decode() if isinstance(s, bytes) else str(s) for s in ref['seq_ids']]
n_snps = len(seq_ids)

# Parse IDs
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

# Delta scores
df['delta_ll'] = df['ref_ll'] - df['alt_ll']
df['delta_ppl'] = df['ref_ppl'] - df['alt_ppl']
df['delta_allele_prob'] = df['ref_allele_prob'] - df['alt_allele_prob']
df['abs_delta_ll'] = df['delta_ll'].abs()

print(f"Loaded {n_snps} SNPs across {df['gene'].nunique()} genes")

# ---------------------------------------------------------------------------
# 2. Load PLINK allele frequencies
# ---------------------------------------------------------------------------
print("\nLoading PLINK frequencies...")

def parse_frq(path):
    frq = pd.read_csv(path, delim_whitespace=True)
    frq.columns = [c.upper() for c in frq.columns]
    return frq

jm = parse_frq('jm_freq.frq')
jr = parse_frq('jr_freq.frq')

# Merge
freq = pd.merge(
    jm[['SNP', 'A1', 'MAF']], 
    jr[['SNP', 'A1', 'MAF']], 
    on='SNP', suffixes=('_JM', '_JR')
)

# The PLINK .frq files report MAF (minor allele frequency).
# We need to match by SNP ID.
df = pd.merge(df, freq[['SNP', 'MAF_JM', 'MAF_JR']], left_on='snp_id', right_on='SNP', how='left')
df['maf_diff'] = (df['MAF_JM'] - df['MAF_JR']).abs()

print(f"Matched {df['MAF_JM'].notna().sum()} SNPs with frequency data")

# ---------------------------------------------------------------------------
# 3. Summary statistics
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("SUMMARY STATISTICS")
print("=" * 60)

print(f"\nDelta Log-Likelihood (Ref - Alt):")
print(f"  Mean: {df['delta_ll'].mean():.3f}")
print(f"  Std:  {df['delta_ll'].std():.3f}")
print(f"  Min:  {df['delta_ll'].min():.3f}")
print(f"  Max:  {df['delta_ll'].max():.3f}")

print(f"\nPerplexity:")
print(f"  Ref mean: {df['ref_ppl'].mean():.3f}")
print(f"  Alt mean: {df['alt_ppl'].mean():.3f}")

# ---------------------------------------------------------------------------
# 4. Top effect variants
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("TOP VARIANTS BY EVO2 EFFECT")
print("=" * 60)

print("\n--- Most deleterious ALT (Ref >> Alt) ---")
top_del = df.nsmallest(10, 'delta_ll')[['gene', 'chrom', 'pos', 'ref', 'alt', 'delta_ll', 'ref_ppl', 'alt_ppl', 'MAF_JM', 'MAF_JR']]
print(top_del.to_string(index=False))

print("\n--- Most beneficial ALT (Alt >> Ref) ---")
top_ben = df.nlargest(10, 'delta_ll')[['gene', 'chrom', 'pos', 'ref', 'alt', 'delta_ll', 'ref_ppl', 'alt_ppl', 'MAF_JM', 'MAF_JR']]
print(top_ben.to_string(index=False))

# ---------------------------------------------------------------------------
# 5. Gene-level aggregation
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("GENE-LEVEL AGGREGATION")
print("=" * 60)

gene_stats = df.groupby('gene').agg(
    n_snps=('snp_id', 'count'),
    mean_delta_ll=('delta_ll', 'mean'),
    max_abs_delta_ll=('abs_delta_ll', 'max'),
    std_delta_ll=('delta_ll', 'std'),
    mean_ref_ppl=('ref_ppl', 'mean'),
    mean_alt_ppl=('alt_ppl', 'mean'),
).sort_values('max_abs_delta_ll', ascending=False)

print(gene_stats.head(15).to_string())

# ---------------------------------------------------------------------------
# 6. Population differentiation vs Evo2 effect
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("POPULATION DIFFERENTIATION vs EVO2 EFFECT")
print("=" * 60)

# Correlation between MAF difference and Evo2 effect
valid = df.dropna(subset=['MAF_JM', 'MAF_JR'])
from scipy.stats import pearsonr, spearmanr

r_pearson, p_pearson = pearsonr(valid['abs_delta_ll'], valid['maf_diff'])
r_spear, p_spear = spearmanr(valid['abs_delta_ll'], valid['maf_diff'])

print(f"\nCorrelation between |delta_LL| and |MAF_JM - MAF_JR|:")
print(f"  Pearson r = {r_pearson:.4f} (p = {p_pearson:.4f})")
print(f"  Spearman rho = {r_spear:.4f} (p = {p_spear:.4f})")

# ---------------------------------------------------------------------------
# 7. Plots
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Generating plots...")
print("=" * 60)

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Evo2 7B Variant Effect Analysis - Jharkhand Chicken', fontsize=14, fontweight='bold')

# 1. Histogram of delta LL
ax = axes[0, 0]
ax.hist(df['delta_ll'], bins=50, color='steelblue', edgecolor='white', alpha=0.8)
ax.axvline(0, color='red', linestyle='--', linewidth=1)
ax.set_xlabel('Delta LL (Ref - Alt)')
ax.set_ylabel('Count')
ax.set_title('Distribution of Variant Effects')

# 2. Ref vs Alt perplexity
ax = axes[0, 1]
ax.scatter(df['ref_ppl'], df['alt_ppl'], c=df['delta_ll'], cmap='RdBu_r', s=20, alpha=0.6, vmin=-10, vmax=10)
ax.plot([df['ref_ppl'].min(), df['ref_ppl'].max()], [df['ref_ppl'].min(), df['ref_ppl'].max()], 'k--', lw=1)
ax.set_xlabel('Ref Perplexity')
ax.set_ylabel('Alt Perplexity')
ax.set_title('Ref vs Alt Perplexity')
cbar = plt.colorbar(ax.collections[0], ax=ax)
cbar.set_label('Delta LL')

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

# 4. MAF difference vs Evo2 effect
ax = axes[1, 0]
ax.scatter(valid['abs_delta_ll'], valid['maf_diff'], c='steelblue', s=20, alpha=0.5)
ax.set_xlabel('|Delta LL| (Evo2 effect)')
ax.set_ylabel('|MAF_JM - MAF_JR|')
ax.set_title(f'Population Diff vs Evo2 Effect\nPearson r = {r_pearson:.3f}')

# 5. Boxplot of delta LL by gene (top 10 genes)
ax = axes[1, 1]
top10_genes = gene_stats.head(10).index
gene_data = [df[df['gene'] == g]['delta_ll'].values for g in top10_genes]
bp = ax.boxplot(gene_data, vert=False, patch_artist=True)
for patch, gene in zip(bp['boxes'], top10_genes):
    patch.set_facecolor('steelblue')
    patch.set_alpha(0.6)
ax.set_yticklabels(top10_genes, fontsize=8)
ax.set_xlabel('Delta LL')
ax.set_title('Variant Effect Distribution by Gene')
ax.axvline(0, color='red', linestyle='--', lw=1)

# 6. Scatter: delta LL vs delta allele prob
ax = axes[1, 2]
ax.scatter(df['delta_ll'], df['delta_allele_prob'], c='steelblue', s=20, alpha=0.5)
ax.set_xlabel('Delta LL (Ref - Alt)')
ax.set_ylabel('Delta Allele Prob (Ref - Alt)')
ax.set_title('LL Effect vs Allele Probability')
ax.axvline(0, color='red', linestyle='--', lw=1)
ax.axhline(0, color='red', linestyle='--', lw=1)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('evo2_variant_analysis.png', dpi=200, bbox_inches='tight')
print("Saved: evo2_variant_analysis.png")

# ---------------------------------------------------------------------------
# 8. Save results table
# ---------------------------------------------------------------------------
df.to_csv('evo2_variant_scores.csv', index=False)
print("Saved: evo2_variant_scores.csv")

gene_stats.to_csv('evo2_gene_level_stats.csv')
print("Saved: evo2_gene_level_stats.csv")

print("\nAnalysis complete!")
