#!/bin/bash
# 定义一个变量
my_variable="Hello, World! "

# 输出变量，此时它仅存在于当前 Shell 中
echo $my_variable

# 将变量导出为环境变量
export my_variable

# 此时，启动一个子 Shell
bash -c 'echo $my_variable'