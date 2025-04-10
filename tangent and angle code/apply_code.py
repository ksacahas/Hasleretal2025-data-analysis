import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.morphology import skeletonize
import random
import os
from skimage import measure
from collections import deque
from find_neighbors import *
from is_point_on_line import *
from find_shortest_path import *
from dist import *
from find_closest_points import *
from find_path import *
from calculate_slope_regression import *
from calculate_angle import *
from get_line import *

# open target image
tg = Image.open('/Users/carolinehasler/Desktop/UC_Berkeley/River images/train_images2/0/4.jpg')

# convert image to array
tg = np.array(tg)

# convert to grayscale
tg_grayscale = np.mean(tg, axis=-1)

# convert to binary
tg_binary = (tg_grayscale > 200).astype(int)

# create a list of all pixels that belong to the river
tg_binary_river = np.argwhere(tg_binary>0)

# find river skeleton
skeleton = skeletonize(tg_binary, method='lee')

# list of pixels that belong to the skeleton
skeleton_list = np.argwhere(skeleton>0).astype(int)

# list of pixels that belong to the contour
contour_list = measure.find_contours(1 - tg_binary, 0)
contour_list = np.vstack(contour_list).astype(int)

# find river contour
contour = np.zeros(np.shape(tg_binary))
for item in contour_list:
    contour[item[0],item[1]]=1


# we want to take the weighted average of the two
# for that, we calculate the distance of ns and nc to pix
# then weights are inverse distance
def weightedaverage(pix, p1, p2, slope1, slope2):
    dist1 = dist(pix, p1)
    dist2 = dist(pix, p2)
    
    w1 = 1/dist1
    w2 = 1/(dist2**2)

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


def plot_line_segment(point, slope, length):
    y1, x1 = point

    slope = 1/slope
    dx = length / (2 * np.sqrt(1 + slope**2))
    dy = slope * dx

    x2 = x1 + dx
    y2 = y1 + dy

    x0 = x1 - dx
    y0 = y1 - dy

    return [x0,x2],[y0,y2]

pix = random.choice(tg_binary_river)
pix = [104, 67]
print("pix", pix)


# maximum search radius
# Adjust if necessary
N = 20


neighbors = find_all_neighbors(tg_binary, pix, N)

# find closest neighbors on skeleton and contour
ns, nc, path = find_closest_points(pix, neighbors, contour, skeleton)

print("closest skeleton point: ", ns)
print("closest contour point: ", nc)

slope_c = None
slope_s = None
angle_s = None
angle_c = None
slope_av = None
angle_av = None

if skeleton[pix[0], pix[1]] > 0 and contour[pix[0], pix[1]] == 0:
    print("pix in skeleton, not in contour")

    # identify first and second neighbors

    ns_neighbors = find_neighbors(ns, 2, 2, 2)
    ns_neighbors = [*ns_neighbors[0], *ns_neighbors[1]]

    # now identify first and second neighbors that are in skeleton_list
    ns_s_n = (ns, *[n for n in ns_neighbors if any(np.array_equal(n, r) for r in skeleton_list)])
    print("neighbors of ns: ", ns_s_n)

    # calculate slope and angle
    slope_s = calculate_slope_regression(ns_s_n)
    print("slope of skeleton tangent: ", slope_s)

    angle_s = calculate_angle(slope_s)
    print("angle skeleton tangent: ", angle_s)

elif contour[pix[0], pix[1]] > 0 and skeleton[pix[0], pix[1]] == 0:
    print("pix in contour, not in skeleton")

    # identify first and second neighbors

    nc_neighbors = get_neighbors(nc, 2, 2, 2)
    nc_neighbors = [*nc_neighbors[0], *nc_neighbors[1]]

    # now identify first and second neighbors that are in skeleton_list
    nc_c_n = (nc, *[n for n in nc_neighbors if any(np.array_equal(n, r) for r in contour_list)])
    print("neighbors of nc: ", nc_c_n)

    # calculate slope and angle
    slope_c = calculate_slope_regression(nc_c_n)
    print("slope of contour tangent: ", slope_c)

    angle_c = calculate_angle(slope_c)
    print("angle contour tangent: ", angle_c)

else:
    print("pix not in contour or skeleton")
    # identify first and second neighbors

    ns_neighbors = get_neighbors(ns, 2, 2, 2)
    ns_neighbors = [*ns_neighbors[0], *ns_neighbors[1]]

    # now identify first and second neighbors that are in skeleton_list
    ns_s_n = (ns, *[n for n in ns_neighbors if any(np.array_equal(n, r) for r in skeleton_list)])
    # print("neighbors of ns: ", ns_s_n)

    # calculate slope 
    slope_s = calculate_slope_regression(ns_s_n)
    print("slope of skeleton tangent: ", slope_s)


    # identify first and second neighbors

    nc_neighbors = get_neighbors(nc, 2, 2, 2)
    nc_neighbors = [*nc_neighbors[0], *nc_neighbors[1]]

    # now identify first and second neighbors that are in skeleton_list
    nc_c_n = (nc, *[n for n in nc_neighbors if any(np.array_equal(n, r) for r in contour_list)])
    # print("neighbors of nc: ", nc_c_n)

    # calculate slope
    slope_c = calculate_slope_regression(nc_c_n)
    print("slope of contour tangent: ", slope_c)


if slope_s is not None and slope_c is not None:
    slope_av = weightedaverage(pix, nc, ns, slope_c, slope_s)
    angle_av = calculate_angle(slope_av)
    print("average slope: ", slope_av)

if slope_c is None and slope_s is not None:
    slope_av = slope_s
    angle_av = calculate_angle(slope_s)

if slope_s is None and slope_c is not None:
    slope_av = slope_c
    angle_av = calculate_angle(slope_c)

# Plotting
colors = plt.cm.rainbow(np.linspace(0, 1, N))
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,8))
ax.imshow(tg_binary, cmap=plt.cm.gray)
# Loop through each level of neighbors and plot with a different color
for i, neighbor_level in enumerate(neighbors):
    color = colors[i]
    ax.scatter(neighbor_level[:, 1], neighbor_level[:, 0], color=color, marker="s",
               s=12, alpha = 0.5)
ax.scatter(pix[1], pix[0], color='r', marker="s", s=5)

ax.scatter(contour_list[:,1], contour_list[:,0], marker = '.',
           color='r', s = 5)
ax.scatter(skeleton_list[:,1], skeleton_list[:,0],marker = '.',
           color='r', s = 5)

    
if slope_s is not None:
    if slope_s >= 1e5:
        ax.vlines(ns[1], ymin=ns[0]-5, ymax=ns[0]+6,color="r",linewidth=1.0)
    else:
        x_line, y_line = plot_line_segment(ns, slope_s, 5)
        ax.plot(x_line, y_line, color='r',linewidth=1.0)

if slope_c is not None:
    if slope_c >= 1e5:
        ax.vlines(nc[1], ymin=nc[0]-5, ymax=nc[0]+6,color="r",linewidth=1.0)
    else:
        x_line, y_line = plot_line_segment(nc, slope_c, 5)
        ax.plot(x_line, y_line, color='r',linewidth=1.0)
    
if slope_av >= 1e5:
        ax.vlines(y, ymin=x-2, ymax=x+3,color='b',linewidth=1.0)
        
else:
        x_line, y_line = plot_line_segment(pix, slope_av, 5)
        ax.plot(x_line, y_line, color='r',linewidth=1.0)

plt.ylim(127,0)
plt.xlim(0,127)

if nc is not None:
    ax.scatter(nc[1], nc[0], color = 'black',marker = '.', s=15)
if ns is not None:
    ax.scatter(ns[1], ns[0], color = 'black',marker = '.', s=15)

plt.show()


