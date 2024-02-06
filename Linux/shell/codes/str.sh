#!/bin/bash


str1="hello world 1"  # 双引号
str2='hello world 2'  # 单引号

# 直接调用
echo $str1
echo $str2

echo "------------"

# 字符串拼接：双引号
name='le0v1n'
name1="hello, $name!"
name2="hello, "$name"!"
name3="hello, '$name'!"
name4="hello, ${name}!"

echo $name1
echo $name2
echo $name3
echo $name4

echo "------------"

# 字符串拼接：单引号
name='le0v1n'

name1='hello, $name!'
name2='hello, '$name'!'
name3='hello, "$name"!'
name4='hello, ${name}!'

echo $name1
echo $name2
echo $name3
echo $name4

echo "------------"

# 字符串长度
email="Le0v1n@163.com"

echo ${email}
echo ${#email}
echo ${email:0:5}
