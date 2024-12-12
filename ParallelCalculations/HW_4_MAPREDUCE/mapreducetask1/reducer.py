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
    words.append(line.split('\t')[1].strip())

for _ in range(8):
    res = []
    for _ in range(random.randint(1, 5)):
        res.append(words[random.randint(0, len(words))])
    print(*res, sep=',')
