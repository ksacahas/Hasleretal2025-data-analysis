import numpy as np
from dist import *


def weightedaverage(pix, p1, p2, slope1, slope2):
    dist1 = dist(pix, p1)
    dist2 = dist(pix, p2)
    
    w1 = 1/(dist1)
    w2 = 1/(dist2)

    if not np.isfinite(slope1):
        slope1 = 100
    if not np.isfinite(slope2):
        slope2 = 100

    if np.abs(slope1) < 1e-5:
        slope1 = 1e-2
    if np.abs(slope2) < 1e-5:
        slope2 = 1e-2

    tot = (w1 * slope1 + w2 * slope2) / (w1 + w2)
    return tot
