#!/bin/bash

DIR=`pwd -P`
export PYTHONPATH=$DIR
python $PYTHONPATH/nn/lbfgstrainer.py\
  -source_instances $DIR/input/source_ruleTable\
  -target_instances $DIR/input/target_ruleTable\
  -model $DIR/output/Trained-file.model.gz\
  -model_la $DIR/output/LA-trained-file.model.gz\
  -source_word_vector $DIR/input/source_word_vector\
  -target_word_vector $DIR/input/target_word_vector\
  -lambda_reg 0.15\
  -m 20\
  -v 2
