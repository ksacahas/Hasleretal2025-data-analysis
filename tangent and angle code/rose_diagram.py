import numpy as np
import matplotlib.pyplot as plt
from math import pi

# we need the angles in radians

def rose_diagram(angles, bin_width):

    # Define the bins
    bins = np.arange(-90, 91, bin_width)  # Bins of width 1 from -90 to 90

    # Create the histogram
    hist, edges = np.histogram(angles, bins=bins)

    # Convert bin edges to radians for polar plot
    theta_original = np.radians(edges[:-1])

    # Duplicate the data for angles greater than 90 degrees and less than -90 degrees
    theta_symmetrical = np.concatenate([theta_original, theta_original + np.pi])

    # Duplicate the histogram values
    hist_symmetrical = np.concatenate([hist, hist])
    """
    xticks = np.linspace(0.0, 2*np.pi, 36, endpoint=False)
    xticklabels = ['0°','10°','20°','30°','40°','50°','60°','70°','80°','90°',
                   '100°','110°','120°','130°','140°','150°','160°','170°',
                   '180°','190°','200°','210°','220°','230°','240°','250°','260°',
                   '270°','280°','290°','300°','310°','320°','330°','340°','350°'
                     ]
    
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.bar(theta_symmetrical, hist_symmetrical, width=np.radians(1*bin_width),
           align='edge', color='blue', alpha=1.0)
    # Set the direction of the plot to clockwise
    ax.set_theta_direction(-1)
    # Set 0 degrees to be at the top of the plot
    ax.set_theta_offset(np.radians(90))
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    ax.set_yticklabels([])
    """
    
    return theta_symmetrical, hist_symmetrical
    
