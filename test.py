#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2020/11/3 9:11
# @Author  : Wang Yuechuan
# @FileName: test.py
# @Software: PyCharm
# @function: 

import time
import numpy as np
from collections import deque

since = time.time()

test = [1,2,34,5]
print(*test)

time_elapsed = time.time() - since
print('Training complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))