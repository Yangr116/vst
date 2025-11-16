"""
Most of code are borrowed from Omni3D
"""
import cv2
import torch
import numpy as np
from PIL import Image
from matplotlib.path import Path as pltPath


def to_float_tensor(input):

    data_type = type(input)

    if data_type != torch.Tensor:
        input = torch.tensor(input)
    
    return input.float()


def get_cuboid_verts_faces(box3d=None, R=None):
    """
    Computes vertices and faces from a 3D cuboid representation.
    Args:
        bbox3d (flexible): [[X Y Z W H L]]
        R (flexible): [np.array(3x3)]
    Returns:
        verts: the 3D vertices of the cuboid in camera space
        faces: the vertex indices per face
    """
    if box3d is None:
        box3d = [0, 0, 0, 1, 1, 1]

    # make sure types are correct
    box3d = to_float_tensor(box3d)
    
    if R is not None:
        R = to_float_tensor(R)

    squeeze = len(box3d.shape) == 1
    
    if squeeze:    
        box3d = box3d.unsqueeze(0)
        if R is not None:
            R = R.unsqueeze(0)
    
    n = len(box3d)

    x3d = box3d[:, 0].unsqueeze(1)
    y3d = box3d[:, 1].unsqueeze(1)
    z3d = box3d[:, 2].unsqueeze(1)
    xsize3d = box3d[:, 3].unsqueeze(1)
    ysize3d = box3d[:, 4].unsqueeze(1)
    zsize3d = box3d[:, 5].unsqueeze(1)

    '''
                    v4_____________________v5
                    /|                    /|
                   / |                   / |
                  /  |                  /  |
                 /___|_________________/   |
              v0|    |                 |v1 |
                |    |                 |   |
                |    |                 |   |
                |    |                 |   |
                |    |_________________|___|
                |   / v7               |   /v6
                |  /                   |  /
                | /                    | /
                |/_____________________|/
                v3                     v2
    '''

    verts = to_float_tensor(torch.zeros([n, 3, 8], device=box3d.device)) # shape=(3, 8), verts[0, :] 表示 8 个点的 x 坐标

    # 这里的 setup 和他们的坐标系是一致的 X right, Y down, Z toword screen
    # setup X
    verts[:, 0, [0, 3, 4, 7]] = -xsize3d / 2
    verts[:, 0, [1, 2, 5, 6]] = xsize3d / 2

    # setup Y
    verts[:, 1, [0, 1, 4, 5]] = -ysize3d / 2
    verts[:, 1, [2, 3, 6, 7]] = ysize3d / 2

    # setup Z
    verts[:, 2, [0, 1, 2, 3]] = -zsize3d / 2
    verts[:, 2, [4, 5, 6, 7]] = zsize3d / 2

    if R is not None:

        # rotate
        verts = R @ verts
    
    # translate
    verts[:, 0, :] += x3d
    verts[:, 1, :] += y3d
    verts[:, 2, :] += z3d

    verts = verts.transpose(1, 2)

    faces = torch.tensor([
        [0, 1, 2], # front TR
        [2, 3, 0], # front BL

        [1, 5, 6], # right TR
        [6, 2, 1], # right BL

        [4, 0, 3], # left TR
        [3, 7, 4], # left BL

        [5, 4, 7], # back TR
        [7, 6, 5], # back BL

        [4, 5, 1], # top TR
        [1, 0, 4], # top BL

        [3, 2, 6], # bottom TR
        [6, 7, 3], # bottom BL
    ]).float().unsqueeze(0).repeat([n, 1, 1])

    if squeeze:
        verts = verts.squeeze()
        faces = faces.squeeze()

    return verts, faces.to(verts.device)


def draw_3d_box_from_verts(im, K, verts3d, color=(0, 200, 200), thickness=1, draw_back=False, draw_top=False, zplane=0.05, eps=1e-4):
    """
    Draws a scene from multiple different modes. 
    Args:
        im (array): the image to draw onto
        K (array): the 3x3 matrix for projection to camera to screen
        verts3d (array): the 8x3 matrix of vertices in camera space
        color (tuple): color in RGB scaled [0, 255)
        thickness (float): the line thickness for opencv lines
        draw_back (bool): whether a backface should be highlighted
        draw_top (bool): whether the top face should be highlighted
        zplane (float): a plane of depth to solve intersection when
            vertex points project behind the camera plane. 
    """

    if isinstance(K, torch.Tensor):
        K = K.detach().cpu().numpy()

    if isinstance(verts3d, torch.Tensor):
        verts3d = verts3d.detach().cpu().numpy()

    # reorder
    bb3d_lines_verts = [[0, 1], [1, 2], [2, 3], [3, 0], [1, 5], [5, 6], [6, 2], [4, 5], [4, 7], [6, 7], [0, 4], [3, 7]]
    
    # define back and top vetice planes
    back_idxs = [4, 0, 3, 7]
    top_idxs = [4, 0, 1, 5]
    
    for (i, j) in bb3d_lines_verts:
        v0 = verts3d[i] # 取出对应的点
        v1 = verts3d[j]

        z0, z1 = v0[-1], v1[-1]

        if (z0 >= zplane or z1 >= zplane):
            
            # computer intersection of v0, v1 and zplane
            s = (zplane - z0) / max((z1 - z0), eps)
            new_v = v0 + s * (v1 - v0)

            if (z0 < zplane) and (z1 >= zplane):
                # i0 vertex is behind the plane
                v0 = new_v
            elif (z0 >= zplane) and (z1 < zplane):
                # i1 vertex is behind the plane
                v1 = new_v

            v0_proj = (K @ v0)/max(v0[-1], eps)  # 相机坐标系转换到 screen 坐标系
            v1_proj = (K @ v1)/max(v1[-1], eps)

            # project vertices
            cv2.line(im, 
                (int(v0_proj[0]), int(v0_proj[1])), 
                (int(v1_proj[0]), int(v1_proj[1])), 
                color, thickness
            )

    # dont draw  the planes if a vertex is out of bounds
    draw_back &= np.all(verts3d[back_idxs, -1] >= zplane)
    draw_top &= np.all(verts3d[top_idxs, -1] >= zplane)

    if draw_back or draw_top:
        
        # project to image
        verts2d = (K @ verts3d.T).T
        verts2d /= verts2d[:, -1][:, np.newaxis]
        
        if type(verts2d) == torch.Tensor:
            verts2d = verts2d.detach().cpu().numpy()

        if draw_back:
            draw_transparent_polygon(im, verts2d[back_idxs, :2], blend=0.5, color=color)

        if draw_top:
            draw_transparent_polygon(im, verts2d[top_idxs, :2], blend=0.5, color=color)


def get_polygon_grid(im, poly_verts):

    nx = im.shape[1]
    ny = im.shape[0]

    x, y = np.meshgrid(np.arange(nx), np.arange(ny))
    x, y = x.flatten(), y.flatten()

    points = np.vstack((x, y)).T

    path = pltPath(poly_verts)
    grid = path.contains_points(points)
    grid = grid.reshape((ny, nx))

    return grid


def draw_circle(im, pos, radius=5, thickness=1, color=(250, 100, 100), fill=True):

    if fill: thickness = -1

    cv2.circle(im, (int(pos[0]), int(pos[1])), radius, color=color, thickness=thickness)


def draw_transparent_polygon(im, verts, blend=0.5, color=(0, 255, 255)):

    mask = get_polygon_grid(im, verts[:4, :])

    im[mask, 0] = im[mask, 0] * blend + (1 - blend) * color[0]
    im[mask, 1] = im[mask, 1] * blend + (1 - blend) * color[1]
    im[mask, 2] = im[mask, 2] * blend + (1 - blend) * color[2]
