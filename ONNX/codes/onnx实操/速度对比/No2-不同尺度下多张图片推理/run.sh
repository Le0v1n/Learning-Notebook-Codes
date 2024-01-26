#!/bin/bash
shape=[18,3,1024,1024]  # 输入图片shape
test_times=50  # 测试次数
warm_up=no  # 是否开启热身

python ONNX/codes/onnx实操/速度对比/No2-不同尺度下多张图片推理/ONNX-fix_dim-multibatch-without_simplified.py \
       --input-shape $shape \
       --test-times $test_times \
       --warm-up $warm_up
python ONNX/codes/onnx实操/速度对比/No2-不同尺度下多张图片推理/ONNX-fix_dim-multibatch-simplified.py \
       --input-shape $shape \
       --test-times $test_times \
       --warm-up $warm_up
python ONNX/codes/onnx实操/速度对比/No2-不同尺度下多张图片推理/ONNX-dynamic_dim-multibatch-without_simplified.py \
       --input-shape $shape \
       --test-times $test_times \
       --test-times $test_times \
       --warm-up $warm_up
python ONNX/codes/onnx实操/速度对比/No2-不同尺度下多张图片推理/ONNX-dynamic_dim-multibatch-simplified.py \
       --input-shape $shape \
       --test-times $test_times \
       --warm-up $warm_up
# python ONNX/codes/onnx实操/速度对比/No2-不同尺度下多张图片推理/PyTorch-dynamic_dim_batch.py \
#        --input-shape $shape \
#        --test-times $test_times \
#        --warm-up $warm_up
