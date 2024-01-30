#!/bin/bash

onnx_path=ONNX/saves/example_model.onnx
onnx_save_path_rename=ONNX/saves/example_model-rename.onnx
onnx_save_path_retype=ONNX/saves/example_model-rename-retype.onnx

# 修改 [label.tmp_0, score.tmp_0] -> [label, score]
python tools/rename_onnx.py \
       --model $onnx_path \
       --origin_names label.tmp_0 score.tmp_0 \
       --new_names label score \
       --save_file $onnx_save_path_rename

python tools/retype_onnx.py \
       --model-path $onnx_save_path_rename \
       --output-path $onnx_save_path_retype
