import numpy as np

def dist(point1, point2):
    return np.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)

def weightedaverage(pix, p1, p2, slope1, slope2):
    # round slope values to 5 decimal places
    slope1 = round(slope1, 5)
    slope2 = round(slope2, 5)

    # calculate distances between point of interest and neighbors on contour
    # and skeleton
    dist1 = dist(pix, p1)
    dist2 = dist(pix, p2)

    # calculate weights
    w1 = 1/(dist1)
    w2 = 1/(dist2)

    # if both slopes are infinite
    if not np.isfinite(slope1) and not np.isfinite(slope2):
        return 100

   # if both slopes are 0
    elif np.abs(slope1)==0.0 and np.abs(slope2)==0.0:
        return 0

    # if slopes are equal and opposite and have large absolute value
    elif slope1==-slope2 and np.abs(slope1)>1 and dist1==dist2:
        return 100

    # if slopes are equal and opposite and have small absolute value
    elif slope1==-slope2 and np.abs(slope1)<1 and dist1==dist2:
        return 0

    # if slopes are perpendicular
    elif slope1==-0.0 and not np.isfinite(slope2):
        return -1

    elif slope1==0.0 and not np.isfinite(slope2):
        return 1

    elif not np.isfinite(slope1) and slope2==-0.0:
        return -1

    elif not np.isfinite(slope1) and slope2==0.0:
        return 1

    # if one slope is much larger than the other
    elif not np.isfinite(slope1) and np.isfinite(slope2):
        slope1 = 0
        slope2 = -1/slope2
        tot = (w1 * slope1 + w2 * slope2) / (w1 + w2)
        return -1/tot

    elif np.isfinite(slope1) and not np.isfinite(slope2):
        slope1 = -1/slope1
        slope2 = 0
        tot = (w1 * slope1 + w2 * slope2) / (w1 + w2)
        return -1/tot

    # normal case
    else:
        tot = (w1 * slope1 + w2 * slope2) / (w1 + w2)
        return tot


    """
    # Alternative for the case where one slope is much larger
    # But this tends to produce spurious bimodality
    elif not np.isfinite(slope1) and np.isfinite(slope2):
        slope1 = 10
        tot = (w1 * slope1 + w2 * slope2) / (w1 + w2)
        return tot

    elif np.isfinite(slope1) and not np.isfinite(slope2):
        slope2 = 10
        tot = (w1 * slope1 + w2 * slope2) / (w1 + w2)
        return tot
        """
