#!/bin/bash
if [ $# -ne 2 ]
then
  echo "Usage: $0 coreNum1 coreNum2"
  exit -1
fi

DIR=`pwd -P`
DIR=$DIR/..
export PYTHONPATH=$DIR

tp_src=Source
tp_trg=Target
src_rt=$DIR/input/moses_source_ruleTable
trg_rt=$DIR/input/moses_target_ruleTable
model=$DIR/output/BRAE_Trained1-file.model.gz
model_src=$DIR/output/BRAE_Trained1-file.model.gz_Source
model_trg=$DIR/output/BRAE_Trained1-file.model.gz_Target
model_la=$DIR/output/BRAE-trained2-file.model.gz
src_vec=$DIR/input/moses_source_word_vector
trg_vec=$DIR/input/moses_target_word_vector
score=$DIR/output/score
output=$DIR/output/score_filtered

lambda_reg=0.15
lambda_reg_L=1e-2
alpha=0.15
lambda_reg_rec=1e-3
lambda_reg_sem=1e-3
maxiter=60
maxiter_la=20
threshold=0.2

./BRAE_train_src.sh $2 &
./BRAE_train_trg.sh $2 &
wait

mpirun -n $1 python $PYTHONPATH/nn/BRAE_Trainer2.py\
  -source_instances $src_rt\
  -target_instances $trg_rt\
  -model_la $model_la\
  -source_word_vector $src_vec\
  -target_word_vector $trg_vec\
  -src_theta_file $model_src\
  -trg_theta_file $model_trg\
  -alpha $alpha\
  -lambda_reg_L $lambda_reg_L\
  -lambda_reg_rec $lambda_reg_rec\
  -lambda_reg_sem $lambda_reg_sem\
  -ml $maxiter_la\
  -v 2\
  >& log_BRAE

python $PYTHONPATH/nn/BRAE_rae2.py\
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

