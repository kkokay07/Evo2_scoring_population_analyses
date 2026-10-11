#!/usr/bin/env python3
"""
Extract Evo2 7B scalar scores for ref/alt SNP sequences.
Scores: sequence log-likelihood, perplexity, SNP allele probability at center.
"""
import sys
sys.path = [p for p in sys.path if '.local' not in p]
sys.path.insert(0, '/home/iiab/Documents/DRallele')
import os
os.chdir('/home/iiab/Documents/DRallele/evo2')
import logging
logging.disable(logging.INFO)

import argparse
import numpy as np
import torch
import torch.nn.functional as F
from Bio import SeqIO
from pathlib import Path
from tqdm import tqdm

from evo2 import Evo2


def load_model(local_path='evo2_7b.pt'):
    print("Loading Evo2 7B...")
    model = Evo2('evo2_7b', local_path=local_path)
    model.model.eval()
    device = next(model.model.parameters()).device
    print(f"Model loaded on {device}")
    return model


def tokenize_batch(sequences, tokenizer):
    tokens_list = [tokenizer.tokenize(seq) for seq in sequences]
    max_len = max(len(t) for t in tokens_list)
    batch_size = len(sequences)
    input_ids = torch.full((batch_size, max_len), 0, dtype=torch.long).cuda()
    for i, t in enumerate(tokens_list):
        input_ids[i, :len(t)] = torch.tensor(t, dtype=torch.long)
    return input_ids


def extract_scores_batch(model, sequences, snp_indices):
    """
    sequences: list of DNA strings
    snp_indices: list of 0-based positions where the SNP is located in each sequence
    Returns list of dicts with scores
    """
    input_ids = tokenize_batch(sequences, model.tokenizer)
    batch_size = input_ids.size(0)
    seq_lens = [len(model.tokenizer.tokenize(s)) for s in sequences]
    
    with torch.inference_mode():
        logits_tuple, _ = model.forward(input_ids, return_embeddings=False)
        logits = logits_tuple[0].float()  # (batch, seq_len, vocab_size)
    
    # Next-token prediction: logits[:, i, :] predicts token[i+1]
    pred_logits = logits[:, :-1, :]  # (batch, seq_len-1, vocab_size)
    targets = input_ids[:, 1:]       # (batch, seq_len-1)
    
    log_probs = F.log_softmax(pred_logits, dim=-1)
    token_ll = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)  # (batch, seq_len-1)
    
    # Per-sequence scores
    probs = F.softmax(pred_logits, dim=-1)  # (batch, seq_len-1, vocab_size)
    
    results = []
    for i in range(batch_size):
        valid_len = seq_lens[i] - 1
        ll = token_ll[i, :valid_len].sum().item()
        mean_ll = token_ll[i, :valid_len].mean().item()
        ppl = torch.exp(-token_ll[i, :valid_len].mean()).item()
        
        # SNP allele probability: the model's predicted probability of the ACTUAL allele at SNP position
        # The SNP is at snp_indices[i] in the original sequence.
        # In next-token prediction, position snp_indices[i]-1 predicts token at snp_indices[i]
        snp_pos = snp_indices[i]
        if snp_pos > 0 and snp_pos - 1 < valid_len:
            actual_token = int(input_ids[i, snp_pos].item())
            allele_prob = probs[i, snp_pos - 1, actual_token].item()
        else:
            allele_prob = np.nan
        
        results.append({
            'log_likelihood': ll,
            'mean_log_prob': mean_ll,
            'perplexity': ppl,
            'snp_allele_prob': allele_prob,
        })
    
    return results


def process_fasta(fasta_path, output_npz, batch_size=4):
    model = load_model()
    records = list(SeqIO.parse(fasta_path, 'fasta'))
    n = len(records)
    print(f"Processing {n} sequences from {fasta_path}")
    
    seqs = []
    ids = []
    snp_indices = []
    
    for rec in records:
        seq = str(rec.seq).upper()
        # SNP is in center of 1kb sequence
        snp_idx = len(seq) // 2
        seqs.append(seq)
        ids.append(rec.id)
        snp_indices.append(snp_idx)
    
    results = {
        'seq_ids': ids,
        'log_likelihood': np.zeros(n, dtype=np.float32),
        'mean_log_prob': np.zeros(n, dtype=np.float32),
        'perplexity': np.zeros(n, dtype=np.float32),
        'snp_allele_prob': np.zeros(n, dtype=np.float32),
    }
    
    for i in tqdm(range(0, n, batch_size), desc="Scoring"):
        batch_end = min(i + batch_size, n)
        batch_scores = extract_scores_batch(
            model, 
            seqs[i:batch_end], 
            snp_indices[i:batch_end]
        )
        for j, scores in enumerate(batch_scores):
            idx = i + j
            results['log_likelihood'][idx] = scores['log_likelihood']
            results['mean_log_prob'][idx] = scores['mean_log_prob']
            results['perplexity'][idx] = scores['perplexity']
            results['snp_allele_prob'][idx] = scores['snp_allele_prob']
    
    np.savez_compressed(output_npz, **results)
    print(f"Saved: {output_npz}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('fasta', help='Input FASTA')
    parser.add_argument('output', help='Output .npz')
    parser.add_argument('--batch-size', type=int, default=4)
    args = parser.parse_args()
    process_fasta(args.fasta, args.output, args.batch_size)
