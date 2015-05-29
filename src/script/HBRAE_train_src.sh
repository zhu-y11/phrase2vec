#!/bin/bash
if [ $# -ne 1 ]
then
  echo "Usage: $0 coreNum"
  exit -1
fi
N=$1

DIR=`pwd -P`
DIR=$DIR/..
export PYTHONPATH=$DIR

tp_src=Source
tp_trg=Target
src_rt=$DIR/input/source_ruleTable
trg_rt=$DIR/input/target_ruleTable
model=$DIR/output/HBRAE_Trained1-file.model.gz
model_src=$DIR/output/HBRAE_Trained1-file.model.gz_Source
model_trg=$DIR/output/HBRAE_Trained1-file.model.gz_Target
model_la=$DIR/output/HBRAE-trained2-file.model.gz
src_vec=$DIR/input/source_word_vector
trg_vec=$DIR/input/target_word_vector
score=$DIR/output/HBRAE_score
output=$DIR/output/HBRAE_score_filtered

lambda_reg=0.15
lambda_reg_L=1e-2
alpha=0.15
lambda_reg_rec=1e-3
lambda_reg_sem=1e-3
maxiter=200
maxiter_la=20
threshold=0.2

mpirun -n $1 python $PYTHONPATH/nn/HBRAE_Trainer1.py\
  -instances $src_rt\
  -model $model\
  -word_vector $src_vec\
  -lambda_reg $lambda_reg\
  -lambda_reg_L $lambda_reg_L\
  -m $maxiter\
  -v 2\
  -tp $tp_src\
  --checking-grad
