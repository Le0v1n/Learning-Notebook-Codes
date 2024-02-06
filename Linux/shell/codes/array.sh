#!/bin/bash

# 定义数组：括号用来表示数组，数组元素用空格符号分隔开
# 数组名=(value1 value2 ... valueN)
exp_array=( "Hello" "world" "你好" '单引号')

echo "数组为: ${exp_array}"
echo "数组为: ${exp_array[0]}"
echo "数组为: ${exp_array[1]}"
echo "数组为: ${exp_array[2]}"
echo "数组为: ${exp_array[3]}"
echo "数组为: ${exp_array[4]}"

exp_array=( "Hello" "world" "你好" '单引号')

# 使用 @ 符号可以取出数组中所有元素
echo ${exp_array[@]}

exp_array=( "Hello" "world" "你好" '单引号')

# 获取数组的长度
array_len_1=${#exp_array[@]}
array_len_2=${#exp_array[*]}

echo $array_len_1
echo ${array_len_2}

echo "------------"

exp_array=( "Hello" "world" "你好" '单引号')

# 获取数组中某一个元素的长度
elem_len_0=${#exp_array[0]}
elem_len_1=${#exp_array[1]}
elem_len_2=${#exp_array[2]}
elem_len_3=${#exp_array[3]}
elem_len_4=${#exp_array[4]}
echo ${elem_len_0}
echo ${elem_len_1}
echo ${elem_len_2}
echo ${elem_len_3}
echo ${elem_len_4}