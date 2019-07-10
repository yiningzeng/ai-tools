#!/bin/bash
#$1 源文件夹路径
#$2 合并的目标文件夹路径
SALVEIFS=$IFS
IFS=$(echo -en "\n\b")

source_pascal_voc_path=$1
target_pascal_voc_path=$2
#首先复制annotations
`cp ${source_pascal_voc_path}/Annotations/*.xml ${target_pascal_voc_path}/Annotations`
#复制图片集到目标路径
`cp ${source_pascal_voc_path}/JPEGImages/* ${target_pascal_voc_path}/JPEGImages`
#合并标记框
for element in `ls ${source_pascal_voc_path}/ImageSets/Main/*.txt`
    do
        onlyFileName=${element##*/} #该命令的作用是去掉变量dir_or_file从左边算起的最后一个’/‘字符及其左边的内容，返回从左边算起的最后一个’/’（不含该字符）的右边的内容
        `echo >> ${2}/ImageSets/Main/${onlyFileName}`
        `cat ${source_pascal_voc_path}/ImageSets/Main/${onlyFileName} >> ${2}/ImageSets/Main/${onlyFileName}` 
    done

IFS=$SAVEIFS