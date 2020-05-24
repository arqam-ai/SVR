# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os

def image_grid(
    images,
    title,
    rows=None,
    cols=None,
    fill: bool = True,
    show_axes: bool = False,
    rgb: bool = True,
):
    """A util function for plotting a grid of images.

    Params:
    ------------------
        images: (N, H, W, 4) array of RGBA images
        rows: number of rows in the grid
        cols: number of columns in the grid
        fill: boolean indicating if the space between images should be filled
        show_axes: boolean indicating if the axes of the plots should be visible
        rgb: boolean, If True, only RGB channels are plotted.
            If False, only the alpha channel is plotted.

    Returns:
    ------------------
        None
    """
    if (rows is None) != (cols is None):
        raise ValueError("Specify either both rows and cols or neither.")

    if rows is None:
        rows = len(images)
        cols = 1

    gridspec_kw = {"wspace": 0.0, "hspace": 0.0} if fill else {}
    fig, axarr = plt.subplots(
        rows, cols, gridspec_kw=gridspec_kw, figsize=(25, 19)
    )
    
    bleed = 0
    fig.subplots_adjust(
        left=bleed, bottom=bleed, right=(1 - bleed), top=(1 - bleed)
    )
    fig.suptitle(title, fontsize=16)
    for ax, im in zip(axarr.ravel(), images):
        if rgb:
            # only render RGB channels
            ax.imshow(im[..., :3])
        else:
            # only render Alpha channel
            ax.imshow(im[..., 3])
        if not show_axes:
            ax.set_axis_off()
    



def visuaize_pts(
    ptcloud,
    title, 
    rows=None,
    cols=None,
    fill: bool = True,
    show_axes: bool = False,
    rgb: bool = True,
):
    """A util function for plotting a grid of ptcloud.

    Params:
    ----------------------------------
        ptcloud: (N, ptnum, 3) array of RGBA images
        rows: number of rows in the grid
        cols: number of columns in the grid
        fill: boolean indicating if the space between images should be filled
        show_axes: boolean indicating if the axes of the plots should be visible
        rgb: boolean, If True, only RGB channels are plotted.
            If False, only the alpha channel is plotted.

    Returns:
    -----------------------------------
        None
    """
    if (rows is None) != (cols is None):
        raise ValueError("Specify either both rows and cols or neither.")

    if rows is None:
        rows = 1
        cols = ptcloud.shape[0]

    fig = plt.figure(figsize=(5 * cols,5 * rows))
    fig.suptitle(title, fontsize=16)
    for row in range(rows):
        for col in range(cols):
            idx = row * cols + col
            idx = min(idx, ptcloud.shape[0]-1)
            sample = ptcloud[idx]
            ax = fig.add_subplot(rows, cols, row * cols + col+1, projection='3d')
            ax.scatter(sample[:,2], sample[:,0], sample[:,1], s= 10)
            ax.set_xlim([0,1.0])
            ax.set_ylim([0,1.0])
            ax.set_zlim([0,1.0])
            ax.set_xlabel('Z')
            ax.set_ylabel('X')
            ax.set_zlabel('Y')
            ax.view_init(30, 135)


class NumpytoPNG(object):
    def __init__(self, vis_dir):
        self.counter = 0
        self.vis_dir = vis_dir
    def save(self, batch):
        print(batch.shape)
        batch = batch.transpose(0, 2, 3, 1)
        for i in range(batch.shape[0]):
            image = batch[i]
            rescaled = (255.0 * image).astype(np.uint8)
            im = Image.fromarray(rescaled)
            im.save(os.path.join(self.vis_dir, 'image_{:04}.png'.format(self.counter)))
            self.counter += 1


            