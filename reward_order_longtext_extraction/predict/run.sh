#!/usr/bin/bash

basepath=$(cd `dirname $0`; pwd)
cd $basepath/../../
source env.sh
cd $basepath/../
source setting.conf
cd $basepath


# 以下是样例，你可以自定义修改
python predict.py \
    --predict_file_dir=$PREDICT_FILE_DIR \
    --experiment_path=$SAVE_MODEL_DIR \
    --predict_result_file_dir=$PREDICT_RESULT_FILE_DIR \
    --pretain_model="../../model/roberta_zh/"