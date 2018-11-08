#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 11:30:06 2018

@author: chrismiller
"""

import numpy as np
NUMPOINTS = 1000000

# generate two columns of random numbers uniformly between -1 and 1
data = np.random.uniform(-1, 1, (NUMPOINTS, 2))

# if point is in the cirlce, count
inCircle = 0
for p in range(NUMPOINTS):
    if data[p][0] * data[p][0] + data[p][1] * data[p][1] <= 1:
        inCircle += 1

# put into derived calculation to approximate pi
pi_approx = 4 * inCircle / NUMPOINTS
print(pi_approx)
