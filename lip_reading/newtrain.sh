#!/bin/bash

# トレーニングスクリプトのパス
TRAIN_SCRIPT="/media/mao/T7/event_based_gaze_tracking/code/new_model/main.py"

# データセットのパス
TRAIN_DATA_PATH="/media/mao/T7/event_based_gaze_tracking/eye_data"

# トレーニングのハイパーパラメータß
LEARNING_RATE=0.001
NUM_EPOCHS=150
BATCH_SIZE=128

# Pythonスクリプトを実行
python $TRAIN_SCRIPT  \
    --train_data_path $TRAIN_DATA_PATH \
    --learning_rate $LEARNING_RATE \
    --num_epochs $NUM_EPOCHS \
    --batch_size $BATCH_SIZE \
    --wandb new_model_project_left
