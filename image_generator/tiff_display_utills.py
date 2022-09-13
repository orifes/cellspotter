import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tifffile


def get_tiff_with_meta_data(tiff_file):
    tiff = tifffile.TiffFile(tiff_file)
    tiff_im = tiff.asarray()
    ranges, luts = tiff.imagej_metadata['Ranges'], tiff.imagej_metadata['LUTs']
    return tiff_im, ranges, luts


def display_im_with_cmap(im, lut, range):
    # vmin, vmax = range
    # cmap = matplotlib.colors.ListedColormap((lut / 256).T.reshape(-1, 3))
    # mn = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    # fig = plt.figure(frameon=False)
    # ax = plt.Axes(fig, [0., 0., 1., 1.])
    # ax.set_axis_off()
    # fig.add_axes(ax)
    # ax.imshow(im, norm=mn, cmap=cmap)
    # return fig
    return apply_lut_on_grayscale_im(im, lut)

import cv2


def apply_lut_on_grayscale_im(im, lut):
    norm_im = cv2.normalize(im, None, 0, 255, cv2.NORM_MINMAX).astype(int)
    rgb_im = np.zeros((im.shape[0], im.shape[1], 3))
    print(lut.shape)
    rgb_im[:, :, 0] = lut[0][norm_im]
    rgb_im[:, :, 1] = lut[1][norm_im]
    rgb_im[:, :, 2] = lut[2][norm_im]
    print(rgb_im)
    return rgb_im/255
