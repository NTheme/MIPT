#!/usr/bin/env python3

# --========================================-- #
#   * Author  : NTheme - All rights reserved
#   * Created : 10 December 2024, 9:47 PM
#   * File    : mapper.py
#   * Project : MIPT
# --========================================-- #

import sys
import string

for line in sys.stdin:
    _, *words = line.strip().split('\t')
    words = ' '.join(words).split()

    for word in words:
        word = word.strip().translate(str.maketrans('', '', string.punctuation))
        if 6 <= len(word) <= 9:
            if word[0].isupper() and word[1:].islower():
                print("%s\t%d" % (word.lower(), 1))
            else:
                print("%s\t%d" % (word.lower(), 0))
