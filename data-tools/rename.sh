#!/bin/bash
#$1 需要处理的文件夹路径 一定要以/结束
#$2 需要保存的文件夹路径 一定要以/结束
#$3 文件格式
SALVEIFS=$IFS
IFS=$(echo -en "\n\b")
n=1
function getdir(){
    for element in `ls ${1}`
    do

        dir_or_file=${1}${element}
        if [ -d ${dir_or_file} ]
        then
            getdir ${dir_or_file}
        else
            current=`date "+%Y-%m-%d %H:%M:%S"`
            timeStamp=`date -d "${current}" +%s`
            #echo $timeStamp
            #将current转换为时间戳，精确到毫秒
            #currentTimeStamp=$
