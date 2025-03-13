#!/usr/bin/env python3

# --========================================-- #
#   * Author  : NTheme - All rights reserved
#   * Created : 10 December 2024, 9:47 PM
#   * File    : reducer.py
#   * Project : MIPT
# --========================================-- #

import sys

word_count = dict()
for line in sys.stdin:
    word, count = line.strip().split('\t')
    count = int(count)
    
    if count == 1:
        if word not in word_count:
            word_count[word] = 0
        if word_count[word] != -1:
            word_count[word] += count      
    else:
        word_count[word] = -1

sorted_words = sorted(word_count.items(), key=lambda x: (-x[1], x[0]))

for word, count in sorted_words[:10]:
    if count != -1:
        print("%s %d" %(word, count))
