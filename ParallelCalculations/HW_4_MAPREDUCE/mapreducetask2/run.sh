#!/usr/bin/env bash

OUTPUT_DIR="/tmp/someones_task113"
NUM_REDUCERS=8

hadoop fs -mkdir -p $OUTPUT_DIR >/dev/null
hadoop fs -rm -r $OUTPUT_DIR* >/dev/null

yarn jar /opt/cloudera/parcels/CDH/lib/hadoop-mapreduce/hadoop-streaming.jar \
    -D mapreduce.job.name="someones_task113" \
    -D mapreduce.job.reduces=${NUM_REDUCERS} \
    -files mapper.py,reducer.py \
    -mapper mapper.py \
    -reducer reducer.py \
    -input /data/wiki/en_articles_part \
    -output $OUTPUT_DIR >/dev/null

hdfs dfs -cat ${OUTPUT_DIR}/* | sort -k2,2nr -k1,1 | head -10
