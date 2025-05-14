import os
import shutil
import uuid
import tempfile
import concurrent.futures
import pickle
import math
import random

import numpy as np
import pandas as pd
import scipy
from scipy.sparse import csr_matrix

import pyranges as pr
from pyranges.pyranges_main import PyRanges

from peak_parser import *


def read_fasta(fasta_file):
    with open(fasta_file, 'r') as f:
        name = ''
        sequences = []
        for line in f:
            if line[0] == '>':
                if name:
                    yield name, ''.join(sequences)
                name = line[1:].rstrip()
                sequences = []
                continue
            else:
                sequences.append(line.rstrip())
        yield name, ''.join(sequences)


def dna_2onehot(sequence):
    # Converte DNA sequence into a one-hot matrix, 'N' is represented by 0.25.
    dna_base = ['A', 'C', 'G', 'T', 'N']
    idxseq = [dna_base.index(base) for base in sequence]
    one_hot = np.zeros((len(idxseq), 4))
    for i in range(len(idxseq)):
        if idxseq[i] == 4:
            one_hot[i, :] = 0.25
        else:
            one_hot[i, idxseq[i]] = 1.
    return one_hot


def onehot_2dna(onehot):
    seq_base = np.array(['A', 'C', 'G', 'T', 'N'])
    index = np.argmax(onehot, axis=1)
    idx_N = np.all(onehot == 0.25, axis=1)
    index[idx_N] = 4
    return ''.join(seq_base[index])


def custom_bed(peak_file, reg_file, bed_file, chroms, reg_file_null=None, bed_file_null=None, len_region=40000, len_peak=200, num_null=40000):
    # peak_file: ChIP-peak file of transcrition factor
    # reg_file: region file of epigenome information (EI)
    # bed_file: customed peak file
    # chroms: target chromosomes, []
    # null_file: randomly picked region 
    #
    bed = macs2_narrowpeak(peak_file)
    c1 = len(bed.peak)
    bed.filter_chr(chrs=chroms)
    bed.filter_qval(qval=1e-2)
    bed.filter_len(len_max=800)
    c2 = len(bed.peak)
    # print(peak_file, c1, c2)
    bed_region = custom_peak(bed.peak, length=len_region, write_file=reg_file)
    bed_custom = custom_peak(bed_region, length=len_peak, write_file=bed_file)
    if reg_file_null:
        bed_region_null = bed.generate_nopeak(
            length=len_region, num=num_null, chrom_size=chrom_size, write_file=reg_file_null)
        bed_custom_null = custom_peak(
            bed_region_null, length=len_peak, write_file=bed_file_null)
    return


def extract_fasta(bed_file, fa_file, seq_file):
    # bed_file: bed file of target region
    # fa_file: fasta file of refrence genome
    # seq_file: fasta file of target region
    #
    def tag_fa(region):
        return '>'+region[0]+':'+str(region[1])+'-'+str(region[2])
    bed = pr.read_bed(bed_file)
    tags = bed.df.apply(tag_fa, axis=1)
    seqs = pr.get_sequence(bed, path=fa_file)

    tags_ = [t+'\n' for t in tags]
    seqs_ = [s+'\n' for s in seqs]
    fa = [''.join(x) for x in zip(tags_, seqs_)]
    with open(seq_file, 'w') as f:
        f.writelines(fa)
    return


def read_seq(seq_file):
    seqs = []
    for name, seq in read_fasta(seq_file):
        chr = name.split(':')[0]
        seqs.append((chr, seq))
    return seqs


def read_region(reg_file):
    regs = []
    with open(reg_file, 'r') as f:
        for line in f:
            line = line.rstrip().split()
            regs.append((line[0], int(line[1]), int(line[2])))
    return regs


def peak_2onehot_chrom_whole(peak_file, onehot_peak_file, chrom_size):
    # peak_file: peak file of epigenome infomation, e.g. ATAC-seq
    # onehot_peak_file: onehot matrix of peaks in whole genome, in dictionary structure, the key for chromosome number, and the value is sparse matrix.
    chrom_region = [(k, 0, v) for k, v in chrom_size.items()]  # 0-base

    bed = pr.read_bed(peak_file)
    ohpeak = {}
    for reg in chrom_region:
        oh = peak_2onehot(bed, region=reg, res=1)
        ohpeak[reg[0]] = scipy.sparse.csr_matrix(oh)
    with open(onehot_peak_file, 'wb') as f:
        pickle.dump(ohpeak, f, pickle.HIGHEST_PROTOCOL)
    return


def load_onehot_peak(onehot_peak_file):
    with open(onehot_peak_file, 'rb') as f:
        ohpeak = pickle.load(f)
    ohpeak_ = {k: v.toarray().flatten() for k, v in ohpeak.items()}
    return ohpeak_


def extract_epi_region(regs, target, targets, targets_files, tmpdir):
    # regs: target regions
    # target: EI type, e.g. 'ATAC'
    # targets: list of EI types, e.g. ['ATAC', 'H3K27ac',]
    # targets_files: onehot matrix of target EI in whole genome,
    # tmpdir:
    #
    l = len(regs)
    if target in targets:
        mx_epi = []
        ohpeak = load_onehot_peak(targets_files[target])
        for reg in regs:
            oh = ohpeak[reg[0]][reg[1]:reg[2]]
            #pad the insufficient length with zeros, to 40000 bp
            if len(oh)<40000:
                pad_l=40000-len(oh)
                oh = np.pad(oh, (pad_l//2, pad_l - pad_l//2), 'constant', constant_values=0)
                oh = oh[:40000]
            mx_epi.append(scale_region(oh, res=200).tolist())
        mx_epi = np.array(mx_epi).astype(int).astype(str)
    else:
        mx_epi = np.reshape(
            np.array(['x' for i in range(l*200)]), (l, 200))
    file_name = tmpdir+'/'+uuid.uuid4().hex+'.npy'
    np.save(file_name, mx_epi)
    return file_name


def generate_peak_context(seq_file, reg_file, targets, targets_files, label, out_file, tmpdir):
    # seq_file: sequence file of target peaks 
    # reg_file: corresponding region file
    # targets: list of EI types, e.g. ['ATAC', 'H3K27ac',]
    # targets_files: onehot matrix of target EI in whole genome,
    # label: 1 or 0, denote True peak or null peak separately
    #
    default_targets = ['ATAC', 'H3K27ac', 'H3K27me3',
                       'H3K36me3', 'H3K4me1', 'H3K4me3', 'H3K9me3']
    seqs = read_seq(seq_file)
    regs = read_region(reg_file)

    dna_seq = [s[1] for s in seqs]
    l = len(dna_seq)
    #Use multiprocess to speed up.
    with concurrent.futures.ProcessPoolExecutor(max_workers=7) as executor:
        feature_to_targets = {executor.submit(
            extract_epi_region, regs, target, targets, targets_files, tmpdir): target for target in default_targets}
    # After all processes finish writing, the main program begins reading the file.
    epi_seq = {}
    for future in concurrent.futures.as_completed(feature_to_targets):
        target = feature_to_targets[future]
        epi_seq[target] = np.load(future.result())

    # The order is same as default_targets
    epi_seq_ = epi_seq[default_targets[0]]
    for target in default_targets[1:]:
        epi_seq_ = np.char.add(epi_seq_, epi_seq[target])
    epi_seq_ = epi_seq_.tolist()
    contpeak = [((dna_seq[i], ' '.join(epi_seq_[i])), label) for i in range(l)]
    with open(out_file, 'wb') as f:
        pickle.dump(contpeak, f, pickle.HIGHEST_PROTOCOL)
    return


def replace_char_at_index(s, indexs, new_char='x'):
    for i in indexs:
        if i < 0 or i >= len(s):
            return s
        else:
            s = s[:i] + new_char + s[i+1:]
    return s


def mask_peak_context(contpeak, targets, keep_dna=True):
    # contpeak: peak context, i.e. ((dna_seq,epi_seq),label)
    # targets: EI types to keep, e.g. ['ATAC-seq', 'H3K27ac', ]
    #
    default_targets = ['ATAC', 'H3K27ac', 'H3K27me3',
                       'H3K36me3', 'H3K4me1', 'H3K4me3', 'H3K9me3']
    seqs, label = contpeak
    dna_seq = seqs[0]
    if not keep_dna:
        dna_seq = 'N'*200
    targets_m = [x for x in default_targets if x not in targets]
    idx_m = [default_targets.index(x) for x in targets_m]
    epi_seq_m = seqs[1].split()
    epi_seq_m = [replace_char_at_index(x, idx_m) for x in epi_seq_m]
    return ((dna_seq, ' '.join(epi_seq_m)), label)


def generate_vocabulary(num):
    # num, number of EI types
    # 0-no signal，1-with signal，x-unknown
    combinations = list(itertools.product(['0', '1','x'], repeat=num))
    vocab=[''.join(map(str,x)) for x in combinations]
    return vocab
