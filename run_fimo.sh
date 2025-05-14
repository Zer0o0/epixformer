#!/bin/bash

tf_list=(ATF3 BHLHE40 CEBPB CREB1 CTCF EGR1 ELF1 ELK1 FOS FOXK2 FOXM1 GABPA JUND MAFK MAX MAZ MXI1 MYC NRF1 REST RFX5 SP1 SRF TCF12 TCF7L2 TEAD4 USF1 USF2 YY1 ZBTB33 ZBTB40 ZNF24)

#run fimo
for tf in ${tf_list[@]}; do
  echo '###'$tf running ...;
  fimo --bfile data/genome/GRCh38.primary_assembly.genome.fa.bfile --thresh 1e-4 --max-stored-scores 100000 --o data/fimo/${tf} data/fimo/${tf}_*.meme data/genome/GRCh38.primary_assembly.genome.fa &;
  cp data/fimo/${tf}/fimo.gff data/fimo/${tf}.gff;
done

#compare fimo scaning sites to ChIP-seq peaks
ce_list=(GM12878 HCT116 HepG2 IMR-90 K562 MCF-7 SK-N-SH)

for tf in ${tf_list[@]}; do
  echo '###'${tf} `wc -l data/fimo/${tf}.gff`;
  for ce in ${ce_list[@]}; do
    fn=data/training/${ce}/${tf}.bed;
    if [ -f ${fn} ]; then
      mkdir data/training/${ce}/${tf};
      bedtools intersect -a data/fimo/${tf}.gff -b ${fn} -v |awk -F '\t' -v OFS='\t' '{print $1,$4,$5,$9,$6,$7}'> data/training/${ce}/${tf}/fimo_neg.bed;
      sleep 10;
      bedtools intersect -a data/fimo/${tf}.gff -b ${fn} -u |awk -F '\t' -v OFS='\t' '{print $1,$4,$5,$9,$6,$7}'> data/training/${ce}/${tf}/fimo_pos.bed;
      sleep 10;
    fi
  done
done
