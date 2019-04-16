import matplotlib.pyplot as plt
import numpy as np

def draw_bounding_boxes(img, loc, glimpse_size):
    import matplotlib.patches as patches
    h, w = img.shape
    # draw bounding box fist    
    loc = loc + 1 # making them in the range of [0, 2]
    x_center, y_center = h / 2 * loc[0], w / 2 * loc[1]
    x, y = x_center - glimpse_size/2, y_center - glimpse_size/2
    return patches.Rectangle((x,y),glimpse_size,glimpse_size,linewidth=1,edgecolor='r',facecolor='none')

def plot_prediction_path(image, locs, n_glimpses, glimpse_size):
    image_h, image_w, image_c = image.shape
    img = image.reshape(image_h, image_w)
    
    max_size = glimpse_size * (2 ** (n_glimpses - 1))
    padding = max_size - glimpse_size
    image = np.pad(img, padding, mode='constant', constant_values=0)
    figs={}
    axs={}
    for i, loc in enumerate(locs):
        figs[i]=plt.figure()
        axs[i]=figs[i].add_subplot(111)
        for j in range(n_glimpses):
            current_size = int(glimpse_size * (2 ** (j)))
            
            axs[i].imshow(image)
            axs[i].set_title(str(i) + ":" + str(loc))
            
            box = draw_bounding_boxes(img, loc, current_size)
            axs[i].add_patch(box)
    plt.show()