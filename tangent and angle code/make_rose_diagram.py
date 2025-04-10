from rose_diagram import *
import pandas as pd
from PIL import Image
from skimage.morphology import skeletonize
from find_closest_points import *
from find_angles import *
from plot_line_segment import *
from rose_diagram import *
import random

def parse_list_of_strings(input_list):
    result = []
    for item in input_list:
        # Remove brackets and split the string into numbers
        numbers = [int(num) for num in item.strip('[]').split()]
        # Append the list of numbers to the result
        result.append(numbers)
    return result

def sampled_rose_diagram(data_path, im_path, n_samples, rsc=False, rsc_size=0):
    # Input:
    # data_path
    # im_path
    # n_samples: number of samples in the data file
    # rsc: rescale image?
    # rsc_size: height of rescaled image if rsc == True
    data = pd.read_csv(data_path).values

    random_indices = random.sample(range(len(data)), n_samples)
    data = data[random_indices]

    pixels = parse_list_of_strings(data[:,1])
    slopes = data[:,6]
    angles = data[:,7]

    theta, hist = rose_diagram(angles, 5)

    # open target image
    tg = Image.open(im_path)# convert image to array

    # rescale image if necessary
    if rsc==True:
        
        # get image dimensions
        w, h = tg.size

        # get rescaled image dimensions
        h_rsc = rsc_size

        print("rescaling image to height=", rsc_size)

        # rescale image
        # we need to rescale the image because we rescaled it to process the sine-generated river curve
        # so the point coordinates saved in the files refer to the rescaled image
        scale_factor = int(h / h_rsc)
        scaled_width = int(w / scale_factor)

        tg = tg.resize((scaled_width, h_rsc))

    tg_array = np.array(tg)

    # convert to binary
    # tg_binary = (tg_array > 200).astype(int)

    # alternative conversion to binary
    # convert to grayscale
    tg_binary = tg.convert('L')

    # convert to array
    tg_binary = np.array(tg_binary)
    tg_binary[tg_binary < 128] = 0  
    tg_binary[tg_binary >= 128] = 1


    # get image dimensions
    h, w = np.shape(tg_binary)
    # create a list of all pixels that belong to the river
    tg_binary_river = np.argwhere(tg_binary>0)

    fig, [ax1, ax2] = plt.subplots(nrows=1, ncols=2, figsize=(12,6))
    ax1.imshow(tg_binary, cmap=plt.cm.gray)
    K = len(pixels)
    for i in range(K):
        pix = pixels[i]
        slope_av = slopes[i]
        if np.abs(slope_av) >= 1e5:
            continue       
        else:
            x_line, y_line = plot_line_segment(pix, slope_av, 5)
            ax1.plot(x_line, y_line, color='black',linewidth=1.0)

    ax2 = plt.subplot(1, 2, 2, polar=True)
    bin_width = 10
    xticks = np.linspace(0.0, 2*np.pi, 36, endpoint=False)
    xticklabels = ['0°','10°','20°','30°','40°','50°','60°','70°','80°','90°',
                   '100°','110°','120°','130°','140°','150°','160°','170°',
                   '180°','190°','200°','210°','220°','230°','240°','250°','260°',
                   '270°','280°','290°','300°','310°','320°','330°','340°','350°'
                     ]
    ax2.bar(theta, hist, width=np.radians(1*bin_width),
           align='edge', color='black', alpha=1.0)
    # Set the direction of the plot to clockwise
    ax2.set_theta_direction(-1)
    # Set 0 degrees to be at the top of the plot
    ax2.set_theta_offset(np.radians(90))
    ax2.set_xticks(xticks)
    ax2.set_xticklabels(xticklabels)
    ax2.set_yticklabels([])
    #plt.savefig("w2_1_rosediagram.png")
    plt.show()

    

