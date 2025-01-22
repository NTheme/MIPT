#!/usr/bin/env python3

# --========================================-- #
#   * Author  : NTheme - All rights reserved
#   * Created : 10 December 2024, 9:47 PM
#   * File    : reducer.py
#   * Project : MIPT
# --========================================-- #

import sys
import random

words = []
for line in sys.stdin:
    add = line.split('\t')[1].strip()
    if len(words) < 40:
        words.append(add)

for i in range(8):
    size = random.randint(1, 5)
    print(*words[i * 5: i * 5 + size], sep=',')
