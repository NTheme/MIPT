#!/usr/bin/env bash

hdfs fsck "$1" -files -blocks \
| grep "Total blocks"\
| awk '{print $4}'
