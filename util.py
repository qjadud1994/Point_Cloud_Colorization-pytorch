import torch
import io
import matplotlib.pyplot as plt
import numpy as np
import h5py


def load_h5(path, *kwd):
    """ Read h5 format files """
    
    f = h5py.File(path)
    load_list = []
    for item in kwd:
        load_list.append(f[item][:])
        
    return load_list


def pc_visualize(pt, color, title):
    """ Point cloud visualization on tensorboard """
    
    from mpl_toolkits.mplot3d import Axes3D
    import PIL.Image
    from torchvision.transforms import ToTensor

    pt = pt.transpose(1,0)
    color = color.transpose(1,0) # -1~1

    color = (color + 1.) / 2.  # 0~1

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    xs = pt[ :, 0]
    ys = pt[ :, 1]
    zs = pt[ :, 2]
    s = 10

    ax.scatter(xs, ys, zs, s=s, c=color, marker='o', zdir='y')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.title.set_text(title)

    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg')
    buf.seek(0)
    
    image = PIL.Image.open(buf)
    image = ToTensor()(image) #.unsqueeze(0)

    return image
    
    
def rotate_pc(pc):
    """
    point cloud data rotation augmentation
    Args:
         pc: size n x 3
    Returns:
         rotated_pc: size n x 3
    """
    pc = pc.transpose(1, 0)

    angle = np.random.uniform() * 2 * np.pi
    cosval = np.cos(angle)
    sinval = np.sin(angle)
    
    #rotation_axis = np.random.randint(0, 3)
    rotation_axis = 1  # test -> only y axis rotation

    if rotation_axis == 0:      # x
        rotation_matrix = np.array([[1, 0, 0], [0, cosval, -sinval], [0, sinval, cosval]])
    elif rotation_axis == 1:    # y
        rotation_matrix = np.array([[cosval, 0, sinval], [0, 1, 0], [-sinval, 0, cosval]])
    else:                       # z
        rotation_matrix = np.array([[cosval, -sinval, 0], [sinval, cosval, 0], [0, 0, 1]])
        
    rotated_pc = np.dot(pc, rotation_matrix).astype(np.float32)
    rotated_pc = rotated_pc.transpose(1, 0)

    return rotated_pc
