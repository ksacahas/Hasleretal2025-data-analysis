import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.morphology import skeletonize
import random
import pandas as pd
from skimage import measure
from find_closest_points import *
from find_angles import *
from find_neighbors import *
from plot_line_segment import *

# open target image
tg = Image.open('/Users/carolinehasler/Downloads/19_filtered_mask.png')

# convert image to array
tg_array = np.array(tg)

tg_binary = (tg_array > 200).astype(int)

# get image dimensions
h, w = np.shape(tg_binary)

# create a list of all pixels that belong to the river
tg_binary_river = np.argwhere(tg_binary>0)

# find river skeleton
skeleton = skeletonize(tg_binary)

# list of pixels that belong to the skeleton
skeleton_list = np.argwhere(skeleton>0).astype(int)

# list of pixels that belong to the contour
contour_list = measure.find_contours(1 - tg_binary, 0)
contour_list = np.vstack(contour_list).astype(int)

# find river contour
contour = np.zeros([h,w])
for item in contour_list:
    contour[item[0],item[1]]=1

# choose K random pixels
K = 200

pixels = []
for i in range(K):
    pix = random.choice(tg_binary_river)
    # reject points near the edge of the image 
    while not ((pix[0] >= 20) & (pix[0] < h-20) & (pix[1] >= 20) & (pix[1] < w-20)):
        print("point rejected")
        pix = random.choice(tg_binary_river)
    pixels.append(pix)

pixels_array = np.array(pixels)
slopes = []
angles = []
ncs = []
nss = []
slopes_nc = []
slopes_ns = []

for i in range(K):
    pix = pixels[i]
    print("starting with point pix = ", pix)
     
    # maximum search radius for finding neighbors
    # Adjust if necessary
    N = 30

    neighbors = find_all_neighbors(tg_binary, pix, N)

    """    
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
    plt.show()
    """

    # find closest neighbors on skeleton and contour
    print("finding closest neighbors")
    ns, nc, path = find_closest_points(pix, neighbors, contour, skeleton)
    
    while nc is None or ns is None:
        print("no neighbors")
        print("replacing point")
        pix = random.choice(tg_binary_river)
        print("proposed pixel: ", pix)
        # reject points near the edge of the image
        while not ((pix[0] >= 20) & (pix[0] < h-20) & (pix[1] >= 20) & (pix[1] < w-20)):
            print("pixel is near the edge of the image")
            print("replacing point again")
            pix = random.choice(tg_binary_river)
        pixels[i] = pix
        neighbors = find_all_neighbors(tg_binary, pix, N)

        # find closest neighbors on skeleton and contour
        ns, nc, path = find_closest_points(pix, neighbors, contour, skeleton)
        
        
    # print pixel
    print("i = ", i)
    print("pix", pix)
    y, x = pix[0], pix[1]
    print("closest skeleton point: ", ns)
    print("closest contour point: ", nc)

    slope_nc, slope_ns, slope_av, angle_av = find_angles(tg_binary, skeleton, contour, pix, ns, nc)

    print("average slope (dy/dx): ", slope_av)
    print("average angle: ", angle_av, "°")
    print("\n")
    ncs.append(nc)
    nss.append(ns)
    
    slopes_nc.append(float(slope_nc))
    slopes_ns.append(float(slope_ns))
    slopes.append(float(slope_av))
    angles.append(float(angle_av))

# Plotting
colors = plt.cm.rainbow(np.linspace(0, 1, N))
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,8))
ax.imshow(tg_binary, cmap=plt.cm.gray)
ax.set_xlabel("y")
ax.set_ylabel("x")
#ax.scatter(contour_list[:,1], contour_list[:,0], color = "gray",
#           marker = ".")
#ax.scatter(skeleton_list[:,1], skeleton_list[:,0], color = "gray",
#           marker = ".")

#ax.scatter(contour_list[:,1], contour_list[:,0], marker = '.',
#           color='r', s = 5)
#ax.scatter(skeleton_list[:,1], skeleton_list[:,0],marker = '.',
#           color='r', s = 5)

for i in range(K):
    pix = pixels[i]
    slope_av = slopes[i] 
    if np.abs(slope_av) >= 1e5:
        continue       
    else:
        x_line, y_line = plot_line_segment(pix, slope_av, 5)
        ax.plot(x_line, y_line, color='b',linewidth=1.0)
        nc = ncs[i]
        ns = nss[i]
        ax.scatter(nc[1], nc[0], color='b')
        ax.scatter(ns[1], ns[0], color='b')
        ax.scatter(pix[1], pix[0], color="deeppink")
    #ax.set_ylim(127,0)
    #ax.set_xlim(0,127)

plt.show()

my_dict = {'pixel': pixels,
        'closest contour point': ncs,
        'closest skeleton point': nss,
        'slope at nc': slopes_nc,
        'slope at ns': slopes_ns,
        'average slope': slopes,
        'average angle': angles}


# Create DataFrame
df = pd.DataFrame(my_dict)

# Use the custom converter when saving to CSV
df.to_csv('image_19_sampled_points.csv')


print("saved csv file")

