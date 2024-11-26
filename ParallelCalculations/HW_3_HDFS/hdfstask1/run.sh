#!/usr/bin/env bash

hdfs fsck "$1" -files -blocks -locations \
| grep "0. B" \
| grep -E -o "([0-9]{1,3}[\.]){3}[0-9]{1,3}" \
| sed -n 2p
