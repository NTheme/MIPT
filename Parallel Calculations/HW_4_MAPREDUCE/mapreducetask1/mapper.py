#!/usr/bin/env python3

# --========================================-- #
#   * Author  : NTheme - All rights reserved
#   * Created : 10 December 2024, 9:47 PM
#   * File    : mapper.py
#   * Project : MIPT
# --========================================-- #

import sys
import random

for line in sys.stdin:
    shuffle = int(random.getrandbits(64))
    print("%d\t%s" % (shuffle, line.strip()))
