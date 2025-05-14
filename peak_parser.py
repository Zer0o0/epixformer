import random
import math

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

import pyranges as pr
from pyranges.pyranges_main import PyRanges

# GRCh38/hg38, https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.chrom.sizes
chrom_size_hs = {'chr1': 248956422, 'chr2': 242193529, 'chr3': 198295559, 'chr4': 190214555, 'chr5': 181538259, 'chr6': 170805979, 'chr7': 159345973, 'chr8': 145138636, 'chr9': 138394717, 'chr10': 133797422, 'chr11': 135086622, 'chr12': 133275309,
              'chr13': 114364328, 'chr14': 107043718, 'chr15': 101991189, 'chr16': 90338345, 'chr17': 83257441, 'chr18': 80373285, 'chr19': 58617616, 'chr20': 64444167, 'chr21': 46709983, 'chr22': 50818468, 'chrX': 156040895, 'chrY': 57227415, }
# GRCm38/mm10, https://hgdownload.soe.ucsc.edu/goldenPath/mm10/bigZips/mm10.chrom.sizes
chrom_size_mm = {'chr1': 195471971, 'chr2': 182113224, 'chr3': 160039680, 'chr4': 156508116, 'chr5': 151834684, 'chr6': 149736546, 'chr7': 145441459, 'chr8': 129401213, 'chr9': 124595110, 'chr10': 130694993, 'chr11': 122082543, 'chr12': 120129022,
              'chr13': 120421639, 'chr14': 124902244, 'chr15': 104043685, 'chr16': 98207768, 'chr17': 94987271, 'chr18': 90702639, 'chr19': 61431566, 'chrX': 171031299, 'chrY': 91744698, }

chrom_size=chrom_size_hs  #

class macs2_narrowpeak():
    def __init__(self, peak_file):
        self.peak_file = peak_file
        self.peak = self._read_peak(peak_file)  # Support gzip file

    def _read_peak(self, peak_file: str):
        # Match each column name to MACS2 narrowpeak.
        names_peak = ['Chromosome', 'Start', 'End', 'Name', '_10log10pvalue', 'Strand',
                      'Foldchange', '_log10pvalue', '_log10qvalue', 'relative_summit_position_to_peak_start']
        peak = pd.read_table(peak_file, header=None, names=names_peak)
        return PyRanges(peak)

    def _get_summit(self):
        return (self.peak.Start+self.peak.End)//2

    def filter_chr(self, chrs):
        self.peak = self.peak[self.peak.Chromosome.isin(chrs)]
        return self.peak

    def filter_pval(self, pval = 0.01):
        threshold = math.log10(pval)*(-1)
        self.peak = self.peak[self.peak._log10pvalue >= threshold]
        return self.peak

    def filter_qval(self, qval= 0.01):
        threshold = math.log10(qval)*(-1)
        self.peak = self.peak[self.peak._log10qvalue >= threshold]
        return self.peak

    def filter_fc(self, fc = 2):
        self.peak = self.peak[self.peak.Foldchange >= fc]
        return self.peak

    def filter_len(self, len_max, len_min = 0):
        lens = self.peak.lengths()
        self.peak = self.peak[pd.Series(
            [len_min <= x <= len_max for x in lens])]
        return self.peak

    def sort(self, n_cpu = 1):
        self.peak = self.peak.sort(nb_cpu=n_cpu)
        return self.peak

    def extract_seq(self, fa_file, fai_file):
        return pr.get_sequence(self.peak, path=fa_file,)

    def generate_nopeak(self, length, num, chrom_size, write_file=None):
        # Randomly select a certain number of regions from the region without peaks as the background.
        def coordinate_peak(peak, chrom_size):
            chrs = set(peak.Chromosome)  # The chromesomes in peak object
            peak_sorted = peak.sort()  # Sort by coordinate
            peak_coord = {}            # chr1:[(2,4),(6,8),...]
            nopeak_coord = {}          # chr1:[(0,2),(4,6),...]
            for chr in chrom_size.keys():
                if chr in chrs:
                    s_peak = list(peak_sorted[chr].Start)
                    e_peak = list(peak_sorted[chr].End)
                    s_nopeak = e_peak.copy()  #
                    e_nopeak = s_peak.copy()
                    s_nopeak.insert(0, 0)
                    e_nopeak.append(chrom_size[chr])
                    peak_coord[chr] = list(zip(s_peak, e_peak))
                    nopeak_coord[chr] = list(zip(s_nopeak, e_nopeak))
                else:
                    # Some peak files may not have a peak on certain chromosomes of the specified chromosomes. Use 'None' to mark the peak of these chromosomes
                    peak_coord[chr] = None  #
                    nopeak_coord[chr] = [(0, chrom_size[chr]),]
            return peak_coord, nopeak_coord

        _, nopeak_coord = coordinate_peak(self.peak, chrom_size)
        coord_keep = {}
        for k, v in nopeak_coord.items():
            # To ensure that the selected region does not overlap with the peak region, only the part whose length is greater than the length of the region to be selected is reserved.
            coord_keep[k] = [x for x in v if x[1]-x[0] > length]
        if not num:
            num = len(self.peak)
        nopeak = {'Chromosome': [],
                  'Start': [],
                  'End': []}
        chrs = np.array(list(chrom_size.keys()))
        for i in range(num):
            ch = chrs[np.random.randint(len(chrs), size=1).item()]
            reg_ch = coord_keep[ch]
            reg = reg_ch[np.random.randint(len(reg_ch), size=1).item()]
            s = np.random.randint(reg[0], reg[1]-length, size=1).item()
            e = s+length
            nopeak['Chromosome'].append(ch)
            nopeak['Start'].append(s)
            nopeak['End'].append(e)
        bed_nopeak = pd.DataFrame(nopeak)
        if write_file:
            bed_nopeak.to_csv(write_file, sep='\t',
                              header=False, index=False)  # Generate bed file
        return PyRanges(bed_nopeak)


def custom_peak(peak, length, anchor = 'summit', orientation = 'both', write_file=None):
    # peak: pyrange object
    # anchor: ['summit','start','end']
    # orientation: ['both','up','down'], for 'up' and 'down' options, need to consider the direction of gene transcription.
    pos_anchor = (peak.Start+peak.End)//2
    pos_start = pos_anchor-length//2
    pos_end = pos_start+length
    ab_end = pd.Series([chrom_size[x] for x in peak.Chromosome])
    ab_end.index = peak.Chromosome.index  # Reset index to stay the same.
    df_peak = pd.DataFrame({'Chromosome': peak.Chromosome,
                            'Start': pos_start,
                            'End': pos_end,
                            'is_start': pos_start,
                            'is_end': pos_end-ab_end,
                            'ab_end': ab_end})
    # Limit the coordinates to the coordinate range of the reference genome, and those beyond will be discarded.
    bed_peak = df_peak[(df_peak['is_start'] >= 0) & (df_peak['is_end'] <= 0)]
    bed_peak = bed_peak[['Chromosome', 'Start', 'End']]
    if write_file:
        bed_peak.to_csv(write_file, sep='\t', header=False, index=False)
    return PyRanges(bed_peak)


def scale_region(nparray, res):
    # The resolution must divide the entire area
    assert len(nparray) % res == 0
    bins = len(nparray)/res
    th = res/2
    array = np.split(nparray, bins)
    array = [x.sum() for x in array]
    # If more than half of the bases in each bin are modified, the bin is considered to be modified and reset to 1., otherwise 0.
    one_hot = [1. if x > th else 0. for x in array]
    one_hot = np.array(one_hot)
    return one_hot

# Converte peak region into a one-hot matrix. The regions with peak is set to 1., and those without peak is set to 0.


def peak_2onehot(peak, region, res = 1):
    # peak: pyrange object
    # region:  the region coordinate, e.g. ('chr1',100,200)
    # res: resolution
    ch, s, e = region
    peak_in_region = peak[ch, s:e]
    len_region = e-s
    one_hot = np.zeros(len_region)
    if len(peak_in_region) > 0:
        # the start site as 0-coordinte.
        coord = peak_in_region.df[['Start', 'End']]-s
        for i in range(coord.shape[0]):
            c = coord.iloc[i].tolist()
            c_s = c[0] if c[0] >= 0 else 0.
            c_e = c[1]+1 if c[1] <= len_region else len_region
            one_hot[c_s:c_e] = 1.
    if res == 1:
        return one_hot
    one_hot = scale_region(one_hot, res)
    return one_hot


def onehot_2peak(region_oh, region):
    # region: (chr,start,end)
    idx = region_oh.indices.tolist()
    ch, s, e = region
    peaks = []
    if not idx:
        return []
    t = [idx[0]]
    for i in range(1, len(idx)):
        if i > e:
            break
        if idx[i]-idx[i-1] > 1:  #
            peaks.append((t[0], t[-1]))
            t = [idx[i]]
        else:
            t.append(idx[i])
    peaks.append((t[0], t[-1]))
    peaks_ = [(ch, x[0]+s, x[1]+s) for x in peaks]
    return peaks_

#
def cut_bed_region(bed_regions,chrom_size,window_size=200,stride_size=100,extend_size=40000):
    #chr1	100036095	100039095
    region_bins={}
    region_regs={}
    def slide_window(bed_region,window_size,stride_size):
        #bed_region: (chr1,100036095,100039095)
        chr,bed_s,bed_e=bed_region
        len_region=int(bed_e)-int(bed_s)+1
        nums=math.ceil((len_region-window_size)/stride_size) +1  #
        bins=[]
        regs=[]
        for i in range(nums):
            start=stride_size*i+int(bed_s)
            end=start+window_size
            end=end if end<=chrom_size[chr] else chrom_size[chr]
            bins.append((chr,start,end))
            
            start_reg=start-(extend_size-window_size)//2
            start_reg=start_reg if start_reg>=0 else 0 
            end_reg=start+(extend_size+window_size)//2  #
            end_reg=end_reg if end_reg<=chrom_size[chr] else chrom_size[chr]
            regs.append((chr,start_reg,end_reg))
        return bins,regs
        
    for bed in bed_regions:
        bed_str='_'.join(list(bed))
        bins,regs=slide_window(bed,window_size=window_size,stride_size=stride_size)
        region_bins[bed_str]=bins
        region_regs[bed_str]=regs
    return region_bins,region_regs
