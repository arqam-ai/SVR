import sys
import numpy as np
import pandas as pd
from pyntcloud import PyntCloud
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt



def plot_cords(cords, title=None, save_file=None):
    fig = plt.figure(figsize=(10,4))
    ax = fig.add_subplot(121, projection='3d')
    ax.scatter(cords[0,:], cords[1,:], cords[2,:], s = 2)
    ax.set_xlim([0,128])
    ax.set_ylim([0,128])
    ax.set_zlim([0,128])
    ax.view_init(30, 135)
    ax = fig.add_subplot(122, projection='3d')
    ax.scatter(cords[0,:], cords[1,:], cords[2,:], s = 2)
    ax.set_xlim([0,128])
    ax.set_ylim([0,128])
    ax.set_zlim([0,128])
    ax.view_init(30, 90)
    plt.title(title)
    if save_file is None:
        plt.show()
        plt.close()
    else:
        plt.savefig(save_file)
        plt.close()

def convert_array_to_dataframe(data):
    columns = ['x', 'y', 'z']
    df = pd.DataFrame(data=data, columns=columns)
    return df

def convert_ptc_to_voxel(ptc, n_x=128, n_y=128, n_z=128):
    cloud = PyntCloud(ptc)
    voxelgrid_id = cloud.add_structure("voxelgrid", n_x=n_x, n_y=n_y, n_z=n_z)
    voxelgrid = cloud.structures[voxelgrid_id]

    x_cords = voxelgrid.voxel_x
    y_cords = voxelgrid.voxel_y
    z_cords = voxelgrid.voxel_z

    voxel = np.zeros((n_x, n_y, n_z)).astype(np.bool)
    for x, y, z in zip(x_cords, y_cords, z_cords):
        voxel[x][y][z] = True

    x_cords = np.expand_dims(x_cords, axis=0)
    y_cords = np.expand_dims(y_cords, axis=0)
    z_cords = np.expand_dims(z_cords, axis=0)
    cords = np.concatenate((x_cords, y_cords, z_cords), axis=0)
    
    return cords, voxel

def convert_cords_to_bool(cords, n_x=128, n_y=128, n_z=128):
    """convert coordnate array to boolean voxel

    Params:
    ----------
    cords:   numpy.array int (n, 3)

    Returns:
    ----------
    voxel    numpy.array boolean (n_x, n_y, n_z)
    """
    voxel = np.zeros((n_x, n_y, n_z)).astype(np.bool) 
    for index in range(cords.shape[0]):
        voxel[cords[index][0]][cords[index][1]][cords[index][2]] = True
    
    return voxel


def plot_cords_compare(cords1, cords2, title=None, save_file=None):
    fig = plt.figure(figsize=(10,4))
    ax = fig.add_subplot(121, projection='3d')
    ax.scatter(cords1[0,:], cords1[1,:], cords1[2,:], s = 2)
    ax.set_xlim([0,128])
    ax.set_ylim([0,128])
    ax.set_zlim([0,128])
    ax.view_init(30, 135)
    ax = fig.add_subplot(122, projection='3d')
    ax.scatter(cords2[0,:], cords2[1,:], cords2[2,:], s = 2)
    ax.set_xlim([0,128])
    ax.set_ylim([0,128])
    ax.set_zlim([0,128])
    ax.view_init(30, 135)
    plt.title(title)
    if save_file is None:
        plt.show()
        plt.close()
    else:
        plt.savefig(save_file)
        plt.close()
