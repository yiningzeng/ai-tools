#!/bin/bash
#$2 源文件夹路径
#$1 合并的目标文件夹路径
SALVEIFS=$IFS
IFS=$(echo -en "\n\b")

source_pascal_voc_path=$1
for element in `ls ${source_pascal_voc_path}/labels/*.txt`
    do
        echo $element
        #onlyFileName=${element##*/} #该命令的作用是去掉变量dir_or_file从左边算起的最后一个’/‘字符及其左边的内容，返回从左边算起的最后一个’/’（不含该字符）的右边的内容
        #`echo >> ${target_pascal_voc_path}/ImageSets/Main/${onlyFileName}`
        `echo ---------- >> all_in_one.txt` 
        `echo ${element} >> all_in_one.txt` 
        `cat ${element} >> all_in_one.txt` 
        `echo ---------- >> all_in_one.txt` 
    done

IFS=$SAVEIFS