#!/usr/bin/env bash

dd if=/dev/zero of=/tmp/output.tre bs=1 count=$1
hdfs dfs -put /tmp/output.tre /tmp/output.tre
hdfs dfs -setrep 1 /tmp/output.tre &>/dev/null

LENS=$(hdfs fsck "/tmp/output.tre" -files -blocks -locations | grep "len" | awk -F' ' '{print $3}' | awk -F'=' '{print $2}')

sz=0
for len in $LENS; do
    sz=$(($sz + $len))
done

echo $(($1 - $sz))

hdfs dfs -rm /tmp/output.tre
rm /tmp/output.tre
