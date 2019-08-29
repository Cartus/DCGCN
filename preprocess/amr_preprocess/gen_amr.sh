#!/bin/bash

# CHANGE THIS
REPO_DIR=/home/zhang_yan/work/AGGCN

# CONSTANTS
DATA_DIR=${REPO_DIR}/data
PREPROC_DIR=${DATA_DIR}/tmp_amr
ORIG_AMR_DIR=${DATA_DIR}/LDC2015E86_DEFT_Phase_2_AMR_Annotation_R1/data/alignments/split
FINAL_AMR_DIR=${DATA_DIR}/amr

#####
# CREATE FOLDER STRUCTURE

mkdir -p ${PREPROC_DIR}/train
mkdir -p ${PREPROC_DIR}/dev
mkdir -p ${PREPROC_DIR}/test

mkdir -p ${FINAL_AMR_DIR}/
# mkdir -p ${FINAL_AMR_DIR}/train
# mkdir -p ${FINAL_AMR_DIR}/dev
# mkdir -p ${FINAL_AMR_DIR}/test

#####
# CONCAT ALL SEMBANKS INTO A SINGLE ONE
cat ${ORIG_AMR_DIR}/training/* > ${PREPROC_DIR}/train/raw_amrs.txt
cat ${ORIG_AMR_DIR}/dev/* > ${PREPROC_DIR}/dev/raw_amrs.txt
cat ${ORIG_AMR_DIR}/test/* > ${PREPROC_DIR}/test/raw_amrs.txt

#####
# CONVERT ORIGINAL AMR SEMBANK TO ONELINE FORMAT
for SPLIT in train dev test; do
    python3 split_amr.py ${PREPROC_DIR}/${SPLIT}/raw_amrs.txt ${PREPROC_DIR}/${SPLIT}/surface.txt ${PREPROC_DIR}/${SPLIT}/graphs.txt
    python3 preproc_amr.py ${PREPROC_DIR}/${SPLIT}/graphs.txt ${PREPROC_DIR}/${SPLIT}/surface.txt ${FINAL_AMR_DIR}/${SPLIT}.amr ${FINAL_AMR_DIR}/${SPLIT}_surface.pp.txt --mode LINE_GRAPH --triples-output ${FINAL_AMR_DIR}/${SPLIT}.grh --anon --map-output ${FINAL_AMR_DIR}/${SPLIT}_map.pp.txt --anon-surface ${FINAL_AMR_DIR}/${SPLIT}.snt --nodes-scope ${FINAL_AMR_DIR}/${SPLIT}_nodes.scope.pp.txt --scope
    paste ${FINAL_AMR_DIR}/${SPLIT}.amr ${FINAL_AMR_DIR}/${SPLIT}.grh > ${FINAL_AMR_DIR}/${SPLIT}.amrgrh
done




python3 global_node.py --input_dir ${FINAL_AMR_DIR}/

for SPLIT in train dev test; do
	cp ${FINAL_AMR_DIR}/${SPLIT}.amr_g  ${FINAL_AMR_DIR}/${SPLIT}.amr
	cp ${FINAL_AMR_DIR}/${SPLIT}.grh_g  ${FINAL_AMR_DIR}/${SPLIT}.grh
	cp ${FINAL_AMR_DIR}/${SPLIT}.amrgrh_g  ${FINAL_AMR_DIR}/${SPLIT}.amrgrh
done

rm ${FINAL_AMR_DIR}/*_g

echo '{"d": 1, "r": 2, "s": 3, "g": 4}' > ${FINAL_AMR_DIR}/edge_vocab.json




