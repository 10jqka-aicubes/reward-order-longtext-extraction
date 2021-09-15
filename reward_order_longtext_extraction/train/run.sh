#!/usr/bin/bash

basepath=$(cd `dirname $0`; pwd)
cd $basepath/../../
source env.sh
cd $basepath/../
source setting.conf
cd $basepath


# 以下是样例，你可以自定义修改
python train.py \
    --train_file_dir=$TRAIN_FILE_DIR \
    --pretain_model="../../model/roberta_zh/" \
    --experiment_path=$SAVE_MODEL_DIR \
    --batch_size=32