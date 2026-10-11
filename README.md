# Evo2 SNP Scoring Pipeline - landscape Genomics Candidate Genes in Jharkhand Chicken (*Gallus gallus*)

Scoring of **536 biallelic SNPs** (within 72 landscape-genomics enriched genes) with the
**Evo2 7B** DNA foundation model, using 1 kb reference/alternate allele sequences, and
downstream analysis of **delta log-likelihood (delta LL)**, perplexity, SNP allele
probability, per-individual functional burden, and population differentiation across
four chicken populations (JM, JR, MA, RW).

**Author:** [Kanaka KK](https://scholar.google.com/citations?user=0dQ7Sf8AAAAJ&hl=en)

---

## Overview

For each SNP, 1 kb of flanking sequence was extracted from the *Gallus gallus* GRCg6a
reference genome (500 bp upstream + SNP + 499 bp downstream), once with the reference
allele and once with the alternate allele at the centre. Both sequences were scored by
Evo2 7B, yielding per-sequence:

| Score | Definition |
|---|---|
| Log-likelihood (LL) | `sum_i log P(x_i \| x_<i)` — total probability of the sequence under the model |
| Perplexity (PPL) | `exp(-mean_i log P(x_i \| x_<i))` — model surprise (lower = more conserved) |
| SNP allele probability | `P(x_snp \| x_<snp)` — predicted probability of the observed base at the SNP centre |

The variant effect metric is:

```
delta_LL = LL_ref - LL_alt
```

- **delta_LL > 0** → reference allele preferred (alt putatively deleterious)
- **delta_LL < 0** → alternate allele preferred (alt putatively beneficial)
- **|delta_LL|** → strength of functional impact


## Step-wise Implementation

### Step 0 — Prerequisites

- **Evo2 7B**: model checkpoint `evo2_7b.pt` and the `evo2` Python package
  ([Arc Institute Evo2](https://github.com/ArcInstitute/evo2); Striped Hyena architecture,
  7B parameters, 1M-token context, 512 byte-level tokens).
- **GPU**: NVIDIA GPU with ≥ 32 GB VRAM (analysis run on RTX 5090). ~100 s for all
  1,072 sequences at batch size 4.
- **Python packages**: `torch`, `numpy`, `pandas`, `matplotlib`, `scipy`, `scikit-learn`,
  `biopython`, `tqdm`.
- **PLINK v1.9** for allele frequencies (`--freq --allow-extra-chr`) and genotype
  recoding (`--recode A`).
- **Reference genome**: Gallus_gallus-4.0 (GCF_000002315.3)
- **Note on environment**: a PyTorch/Transformer Engine version conflict was resolved by
  sanitizing `sys.path` (removing `~/.local`) and restoring the conda
  `transformer-engine-torch` shared libraries — this is already handled at the top of
  `extract_evo2_scores.py`.

### Step 1 — Identify SNPs in candidate genes

Landscape-genomics candidate genes (`all_unique_genes.txt`) were mapped to the GRCg6a
GTF (94/103 genes mapped; coordinates in `gene_coords.tsv`). SNPs from the PLINK
`.bim` file were intersected with gene boundaries (chromosome names harmonized by
stripping version suffixes). 651 genic SNPs in 73 genes were found (`snps_in_genes.tsv`);
after keeping only biallelic SNVs, **536 SNPs across 72 genes** remained
(`snp_gene_map_536.txt`).

### Step 2 — Build 1 kb reference/alternate FASTA files

For each of the 536 SNPs, 1 kb of flanking sequence centred on the SNP was extracted
from the reference FASTA (`samtools faidx`). Two sequences per SNP were generated:
the **reference-allele** sequence and the **alternate-allele** sequence (base substituted
at the centre position), written with FASTA headers of the form:

```
>NC_006088.3:12608733|PHTF2|NC_006088:12608733|REF=A|ALT=G
```

Outputs: `data/snps_ref_1kb.fa`, `data/snps_alt_1kb.fa` (536 records each).

### Step 3 — Score sequences with Evo2 7B

```bash
python scripts/extract_evo2_scores.py data/snps_ref_1kb.fa data/snps_ref_1kb_evo2_scores.npz
python scripts/extract_evo2_scores.py data/snps_alt_1kb.fa data/snps_alt_1kb_evo2_scores.npz
```

The script loads Evo2 7B, runs a single forward pass per sequence (batch size 4,
`torch.inference_mode()`), and saves per-sequence log-likelihood, mean log-probability,
perplexity, and SNP allele probability to `.npz`.

### Step 4 — Population allele frequencies (PLINK)

```bash
plink --bfile ch_gmmlm --keep jm.keep --freq --allow-extra-chr --out jm_freq
plink --bfile ch_gmmlm --keep jr.keep --freq --allow-extra-chr --out jr_freq
plink --bfile ch_gmmlm --keep ma.keep --freq --allow-extra-chr --out ma_freq
plink --bfile ch_gmmlm --keep rw.keep --freq --allow-extra-chr --out rw_freq
```

### Step 5 — Delta LL analysis

```bash
# Two populations (JM vs JR)
python scripts/evo2_analysis.py

# All four populations (JM, JR, MA, RW)
python scripts/evo2_analysis_all_pops.py
```

Both scripts compute `delta_LL = LL_ref - LL_alt`, rank the most deleterious/beneficial
ALT variants, aggregate effects per gene, correlate |delta_LL| with population MAF
differentiation (Pearson/Spearman), and produce figures + CSV tables.

### Step 6 — Joint & population-wise analysis

```bash
python scripts/joint_population_analysis.py
```

Uses the additive genotype matrix (`ch_gmmlm_genes_only.raw`) to compute, per individual:

```
Total burden = (1/S) * Σ_s [ g_s * |delta_LL_s| ]
Del  burden = (1/S) * Σ_s [ g_s * max(0, -delta_LL_s) ]
Ben  burden = (1/S) * Σ_s [ g_s * max(0,  delta_LL_s) ]
```

where `g_s ∈ {0,1,2}` is the alt-allele count at SNP `s` and `S = 536`. Population
differences are tested with one-way ANOVA; per-SNP genotype ANOVA across the four
populations is correlated with |delta_LL|; PCA is performed on the 536-SNP genotype matrix.

## Key Outputs

- `results/evo2_variant_scores.csv`, `results/evo2_variant_scores_4pops.csv` — per-SNP Evo2 scores, delta metrics, MAF per population
- `results/evo2_gene_level_stats*.csv` — per-gene aggregated effects
- `results/evo2_individual_burden.csv` — per-individual burden scores (84 individuals)
- `results/evo2_snp_population_anova.csv` — per-SNP population-differentiation ANOVA
- `results/evo2_manuscript_methods_results_discussion.txt` — full Methods/Results/Discussion write-up
- `figures/` — variant effect, population MAF, joint population analysis plots

## Data Provenance

- 84 Jharkhand chickens: JM (n=15), JR (n=26), MA (n=27), RW (n=16); 52,629 genome-wide SNPs (GRCg6a).
- Reference: NCBI Assembly GCF_000002315.6 (GRCg6a), RefSeq annotation release 104.

## Citation

If you use this pipeline, please cite Evo2 (Arc Institute, 2024) and PLINK
(Purcell et al., 2007), and credit **[Kanaka KK](https://scholar.google.com/citations?user=0dQ7Sf8AAAAJ&hl=en)**.

## License

MIT.
