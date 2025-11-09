# Copyright 2025 [Visual Spatial Tuning] Authors
import cv2
import torch
import numpy as np
from PIL import Image
from matplotlib.path import Path as pltPath


def get_cuboid_verts_faces(box3d=None, R=None):
    """
    Computes vertices and faces for one or more 3D cuboids.
    This function generates the 8 vertices and 12 triangular faces for cuboids
    defined by their center, dimensions, and an optional rotation.
    Args:
        box3d (tensor-like, optional): A tensor of shape (N, 6) or (6,).
            Each row represents a cuboid as [cx, cy, cz, w, h, l] (center and size).
            If None, a default 1x1x1 cuboid at the origin is used.
            Defaults to None.
        R (tensor-like, optional): A tensor of shape (N, 3, 3) or (3, 3).
            Rotation matrix for each cuboid. If None, no rotation is applied.
            Defaults to None.
    Returns:
        tuple[torch.Tensor, torch.Tensor]:
        - verts (torch.Tensor): A tensor of shape (N, 8, 3) containing the
          cuboid vertices in 3D space. If the input was for a single cuboid,
          the shape is (8, 3).
        - faces (torch.Tensor): A tensor of shape (N, 12, 3) containing the
          vertex indices for each of the 12 triangular faces. If the input
          was for a single cuboid, the shape is (12, 3).
    """
    if box3d is None:
        box3d = [0, 0, 0, 1, 1, 1] # Default unit cube at origin
    # --- 1. Input Handling and Normalization ---
    box3d = to_float_tensor(box3d)
    is_single_box = box3d.dim() == 1
    if is_single_box:
        box3d = box3d.unsqueeze(0)
    device = box3d.device
    num_boxes = box3d.shape[0]
    # --- 2. Create Vertices for an Axis-Aligned Unit Cube ---
    # Create a 2x2x2 grid of points ranging from -0.5 to 0.5
    # This generates the 8 corners of a unit cube centered at the origin.
    # The `itertools.product` logic is vectorized using meshgrid.
    corners = torch.tensor(
        [[-0.5, -0.5, -0.5],
         [-0.5, -0.5,  0.5],
         [-0.5,  0.5, -0.5],
         [-0.5,  0.5,  0.5],
         [ 0.5, -0.5, -0.5],
         [ 0.5, -0.5,  0.5],
         [ 0.5,  0.5, -0.5],
         [ 0.5,  0.5,  0.5]],
        dtype=torch.float32, device=device
    ) # Shape: (8, 3)
    # --- 3. Scale, Rotate, and Translate Vertices ---
    # Extract center and size from box3d
    center = box3d[:, :3] # Shape: (N, 3)
    size = box3d[:, 3:]   # Shape: (N, 3)
    # Unsqueeze to enable broadcasting for batch operations
    # corners: (8, 3) -> (1, 8, 3)
    # size:    (N, 3) -> (N, 1, 3)
    verts = corners.unsqueeze(0) * size.unsqueeze(1) # Scale the cube
    # Apply rotation if provided
    if R is not None:
        R = to_float_tensor(R).to(device)
        if is_single_box:
            R = R.unsqueeze(0)
        # (N, 3, 3) @ (N, 3, 8) -> (N, 3, 8)
        verts = torch.bmm(R, verts.transpose(1, 2)).transpose(1, 2)
    # Translate the vertices to their final positions
    # center: (N, 3) -> (N, 1, 3)
    verts += center.unsqueeze(1)
    # --- 4. Define Faces ---
    # The face indices are constant for all cuboids.
    # We define them once and then expand for the batch.
    faces = torch.tensor(
        [[0, 2, 1], [1, 2, 3],  # Bottom face
         [4, 5, 6], [5, 7, 6],  # Top face
         [0, 1, 4], [1, 5, 4],  # Back face
         [2, 6, 3], [3, 6, 7],  # Front face
         [0, 4, 2], [2, 4, 6],  # Left face
         [1, 3, 5], [3, 7, 5]], # Right face
        dtype=torch.long, device=device
    ) # Shape: (12, 3)
    faces = faces.unsqueeze(0).expand(num_boxes, -1, -1) # Shape: (N, 12, 3)
    # --- 5. Final Output Formatting ---
    if is_single_box:
        verts = verts.squeeze(0)
        faces = faces.squeeze(0)
    return verts, faces


def to_float_tensor(input):
    
    if type(input) != torch.Tensor:
        input = torch.tensor(input)
    
    return input.float()


def _to_numpy(tensor):
    """Helper to convert a PyTorch tensor to a NumPy array."""
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    return tensor


def draw_3d_box_from_verts(im, K, verts3d, color=(0, 200, 200), thickness=1, draw_back=False, draw_top=False, zplane=0.05, eps=1e-4):
    """
    Draws a 3D bounding box on an image by projecting its 3D vertices.
    This function handles clipping for vertices that are behind the camera's near plane
    and can optionally highlight the top and back faces of the box.
    Args:
        im (np.ndarray): The image to draw onto.
        K (array-like): The 3x3 camera intrinsics matrix for projection.
        verts3d (array-like): The 8x3 matrix of the box's vertices in camera space.
        color (tuple): BGR color for the lines, scaled [0, 255].
        thickness (int): The line thickness for drawing.
        draw_back (bool): If True, highlights the back face of the box.
        draw_top (bool): If True, highlights the top face of the box.
        zplane (float): The near clipping plane depth. Any part of a line behind
                        this plane will be clipped.
        eps (float): A small epsilon value to prevent division by zero.
    """
    # --- 1. Input Conversion and Setup ---
    K = _to_numpy(K)
    verts3d = _to_numpy(verts3d)
    # Define the 12 lines (edges) of the cuboid by vertex indices
    edges = np.array([
        [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom face
        [4, 5], [5, 6], [6, 7], [7, 4],  # Top face
        [0, 4], [1, 5], [2, 6], [3, 7]   # Connecting edges
    ])
    # --- 2. Clip Lines Against the Near Plane and Draw ---
    for i, j in edges:
        v0, v1 = verts3d[i], verts3d[j]
        z0, z1 = v0[2], v1[2]
        # Skip drawing if both vertices are behind the near plane
        if z0 < zplane and z1 < zplane:
            continue
        # If one vertex is behind the plane, clip the line segment
        # This is an implementation of the Sutherland-Hodgman algorithm for a single clipping plane
        if z0 < zplane:
            # v0 is behind, clip it to the plane
            s = (zplane - z0) / (z1 - z0 + eps)
            v0 = v0 + s * (v1 - v0)
        elif z1 < zplane:
            # v1 is behind, clip it to the plane
            s = (zplane - z1) / (z0 - z1 + eps)
            v1 = v1 + s * (v0 - v1)
        # --- 3. Project Clipped Vertices and Draw Line ---
        # Project to 2D screen space: p' = K @ P / Z
        p0 = (K @ v0) / (v0[2] + eps)
        p1 = (K @ v1) / (v1[2] + eps)
        cv2.line(
            im,
            (int(p0[0]), int(p0[1])),
            (int(p1[0]), int(p1[1])),
            color,
            thickness
        )
    # --- 4. Draw Highlighted Faces (Optional) ---
    face_indices = {
        'back': [4, 7, 3, 0], # Using a consistent winding order
        'top': [4, 5, 1, 0]
    }
    # Check if all vertices of a face are visible before drawing
    can_draw_back = draw_back and np.all(verts3d[face_indices['back'], 2] >= zplane)
    can_draw_top = draw_top and np.all(verts3d[face_indices['top'], 2] >= zplane)
    if can_draw_back or can_draw_top:
        # Project all 8 vertices at once for efficiency
        # verts_proj = (K @ verts3d.T).T / (verts3d[:, 2, np.newaxis] + eps)
        # The above line is a vectorized version, but since we already projected
        # the vertices needed for the lines, we can just project all 8 here.
        verts2d = (K @ verts3d.T).T
        verts2d /= (verts2d[:, 2, np.newaxis] + eps)
        verts2d = _to_numpy(verts2d)
        if can_draw_back:
            draw_transparent_polygon(im, verts2d[face_indices['back'], :2], blend=0.5, color=color)
        if can_draw_top:
            draw_transparent_polygon(im, verts2d[face_indices['top'], :2], blend=0.5, color=color)


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
