#!/bin/bash
DIR=`pwd -P`
DIR=$DIR/..
export PYTHONPATH=$DIR
src_rt=$DIR/input/source_ruleTable
trg_rt=$DIR/input/target_ruleTable
model=$DIR/output/Trained-file.model.gz
model_la=$DIR/output/LA-trained-file.model.gz
src_vec=$DIR/input/source_word_vector
trg_vec=$DIR/input/target_word_vector
score=$DIR/output/score
output=$DIR/output/score_filtered

alpha=0.15
lambda_reg_rec=1e-3
lambda_reg_sem=1e-3
maxiter=20
maxiter_la=20
threshold=0.2

python $PYTHONPATH/nn/hiero_trainer.py\
  -source_instances $src_rt\
  -target_instances $trg_rt\
  -model $model\
  -model_la $model_la\
  -source_word_vector $src_vec\
  -target_word_vector $trg_vec\
  -lambda_reg $alpha\
  -lambda_reg_rec $lambda_reg_rec\
  -lambda_reg_sem $lambda_reg_sem\
  -m $maxiter\
  -ml $maxiter_la\
  -v 2\
  --checking-grad

comment="
python $PYTHONPATH/nn/rae_cp.py\
  $src_rt\
  $trg_rt\
  $src_vec\
  $trg_vec\
  $model_la\
  $score

python $PYTHONPATH/filter.py\
  -score_file $score\
  -filtered_file $output\
  -threshold $threshold
"

