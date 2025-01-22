#!/usr/bin/env bash

SERVER=$(hdfs fsck -blockId "$1" | grep "HEALTHY" | awk -F' ' '{print $5}'  | awk -F'/' '{print $1}' | head -1)

echo $SERVER:$(ssh hdfsuser@$SERVER "find / -name $1 2>/dev/null")
