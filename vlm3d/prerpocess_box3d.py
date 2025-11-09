# Copyright 2025 [Visual Spatial Tuning] Authors
import re
import math
from typing import Optional, List, Dict, Tuple
import numpy as np
import copy
from loguru import logger
from scipy.spatial.transform import Rotation
from vlm3d.utils.box3d_utils import decode_bbox3d_dict_from_json, encode_json, convert_degree_range


def get_camera_params(input_string: str):
    width_match = re.search(r'image width=(\d+)', input_string)
    height_match = re.search(r'height=(\d+)', input_string)
    fx_match = re.search(r'Focal length f_x=(\d+\.\d+)', input_string)
    fy_match = re.search(r'f_y=(\d+\.\d+)', input_string)
    cx_match = re.search(r'c_x=(\d+\.\d+)', input_string)
    cy_match = re.search(r'cy=(\d+\.\d+)', input_string)
    if cy_match is None:
        cy_match = re.search(r'c_y=([\d.]+)', input_string)
    camera_params = [width_match, height_match, fx_match, fy_match, cx_match, cy_match]
    if any([x is None for x in camera_params]):
        return [None] * 6
    return [float(x.group(1)) for x in camera_params]


def scale_image_to_new_fov(original_width, original_height, original_focal_length, new_focal_length):
    """
    在保持原始 HFOV 和 VFOV 的前提下，计算缩放后的图像尺寸，使其适配新的焦距。
    
    参数:
        original_width (float): 原图像宽度（像素）
        original_height (float): 原图像高度（像素）
        original_focal_length (float): 原焦距（像素）
        new_focal_length (float): 目标焦距（像素）

    返回:
        (new_width, new_height): 缩放后的图像尺寸（像素）
    """
    # 计算原图的 FOV
    hfov_rad = 2 * math.atan(original_width / (2 * original_focal_length))
    vfov_rad = 2 * math.atan(original_height / (2 * original_focal_length))

    # 使用新的焦距和相同的 FOV 计算新的图像尺寸
    new_width = 2 * new_focal_length * math.tan(hfov_rad / 2)
    new_height = 2 * new_focal_length * math.tan(vfov_rad / 2)

    return new_width, new_height


def convert_prefix_prompt_to_fov(ori_prompt_str, ori_size=None, new_focal_length=529.5, no_resize=False):

    # 提取 width, height, fx, fy
    camera_params = get_camera_params(ori_prompt_str)

    if all([x is not None for x in camera_params]):
        original_width, original_height, fx, fy = camera_params[:4]
        if ori_size is not None:
            original_width, original_height = ori_size
        # 假设新的焦距与原焦距相同
        new_width, new_height = scale_image_to_new_fov(original_width, original_height, fx, new_focal_length)
        new_width = int(new_width)
        new_height = int(new_height)

        # 计算 hfov 和 vfov
        hfov_rad = 2 * math.atan(original_width / (2 * fx))
        vfov_rad = 2 * math.atan(original_height / (2 * fx))
        hfov = math.degrees(hfov_rad)
        vfov = math.degrees(vfov_rad)

        # print(f"original_width, original_height = {original_width, original_height}, new_width, new_height={new_width, new_height}")
        # 生成新的提示字符串
        if no_resize:
            new_prompt = f"Here are the detailed camera parameters for the image.\n"
            new_prompt += f"Camera intrinsic parameters: Horizontal fov, hfov={hfov:.2f}, and vertical fov, vfov={vfov:.2f}. Image width={original_width} and height={original_height}. We do not consider distortion parameters here. \n"
            new_prompt += "Camera coordinate: X-axis points rightward, Y-axis points downward, and Z-axis points forward. The origin point is the camera location.\n"
            new_prompt += "We take the camera coordinate system as the world coordinate system."
            # 保留原提示字符串中后面的部分
            remaining_prompt = ori_prompt_str.split("We take the camera coordinate system as the world coordinate system.")[1]
            new_prompt += remaining_prompt
            return new_prompt, (original_width, original_height), camera_params
        else:
            new_prompt = f"Here are the detailed camera parameters for the image.\n"
            new_prompt += f"Camera intrinsic parameters: Horizontal fov, hfov={hfov:.2f}, and vertical fov, vfov={vfov:.2f}. Image width={new_width} and height={new_height}. We do not consider distortion parameters here. \n"
            new_prompt += "Camera coordinate: X-axis points rightward, Y-axis points downward, and Z-axis points forward. The origin point is the camera location.\n"
            new_prompt += "We take the camera coordinate system as the world coordinate system."
            # 保留原提示字符串中后面的部分
            remaining_prompt = ori_prompt_str.split("We take the camera coordinate system as the world coordinate system.")[1]
            new_prompt += remaining_prompt
            return new_prompt, (new_width, new_height), camera_params
    else:
        print("fail to convert prefix prompt to fov")
        print(ori_prompt_str)
        return ori_prompt_str, None, camera_params


xyz_prompt = """
3D bounding box format: [x_center, y_center, z_center, x_size, y_size, z_size, pitch, yaw, roll]
* x_center, y_center, z_center: the center of the object in the camera coordinate, in meters. z_center is the depth of the object in space.
* x_size, y_size, z_size: The dimensions of the object along the ( XYZ ) axes, in meters, when the rotation angles are zero.
* pitch, yaw, roll: Euler angles representing rotations around the X, Y, and Z axes, respectively. Each angle is normalized to the range of (-1, 1) and is multiplied by 180 to convert it into degrees.
"""

uvd_prompt = """
3D bounding box format: [u_center, v_center, depth, x_size, y_size, z_size, pitch, yaw, roll]
* u_center, v_center: the center of object in the image coordinate, in pixels. The origin of the image coordinate system is at the top-left corner.
* depth: the depth of the object in the camera coordinate, in meters.
* x_size, y_size, z_size: The dimensions of the object along the ( XYZ ) axes, in meters, when the rotation angles are zero.
* pitch, yaw, roll: Euler angles representing rotations around the X, Y, and Z axes, respectively. Each angle is normalized to the range of (-1, 1) and is multiplied by 180 to convert it into degrees.
"""

quat_prompt = """
3D bounding box format: [x_center, y_center, z_center, x_size, y_size, z_size, qx, qy, qz, qw]
* x_center, y_center, z_center: the center of the object in the camera coordinate, in meters. z_center is the depth of the object in space.
* x_size, y_size, z_size: The dimensions of the object along the ( XYZ ) axes, in meters, when the rotation angles are zero.
* qx, qy, qz, qw: a unit quaternion representing the rotation. qw is the scalar (real) part. qx, qy and qz are the vector (imaginary) components of the quaternion.
"""

def _convert_bbox3d_to_uvd(conversations: List[Dict], camera_params: Optional[List], new_size: Optional[Tuple]):
    # image = conversations['image']
    # conversations = conversations['conversations']
    if camera_params is None:
        camera_params = get_camera_params(conversations[0]['value'])
    if any([x is None for x in camera_params]):
        return conversations
    # get camera intrinsics
    original_width, original_height, fx, fy, cx, cy = camera_params
    scale_factor_w, scale_factor_h = 1.0, 1.0
    if new_size is not None:
        new_width, new_height = new_size
        scale_factor_w, scale_factor_h = new_width / original_width, new_height / original_height
    K = np.array([
        [fx * scale_factor_w, 0, cx * scale_factor_w], 
        [0, fy * scale_factor_h, cy * scale_factor_h], 
        [0, 0, 1]])
    # replace the prefix prompt
    if xyz_prompt in conversations[0]['value']:
        conversations[0]['value'] = conversations[0]['value'].replace(xyz_prompt, uvd_prompt)
    for idx, conv in enumerate(conversations):
        if conv['from'] == 'gpt':
            value = conv['value']
            bbox3d_dict_list = decode_bbox3d_dict_from_json(value)
            for bbox3d_dict in bbox3d_dict_list:
                bbox3d = bbox3d_dict['bbox_3d']
                uvd = K @ np.array(bbox3d[:3]) # [3, 3] @ [3]
                depth = max(uvd[2], 1e-5)
                uv = (uvd[:2] / depth).astype(int).tolist()
                bbox3d[:2] = uv
                bbox3d_dict['bbox_3d'] = copy.deepcopy(bbox3d)
            new_value = encode_json(bbox3d_dict_list)
            conv['value'] = re.sub(r'```json.*?```', new_value, value, flags=re.DOTALL)
    return conversations


def convert_bbox3d_to_uvd(conversations: List[Dict], camera_params: Optional[List], new_size: Optional[Tuple]):
    try:
        return _convert_bbox3d_to_uvd(conversations, camera_params, new_size)
    except Exception as e:
        logger.error(f"{conversations}\n{e}")
        return conversations


def convert_bbox3d_to_quat(conversations: List[Dict], camera_params: Optional[List], new_size: Optional[Tuple]):
    try:
        return _convert_bbox3d_to_quat(conversations, camera_params, new_size)
    except Exception as e:
        logger.error(f"{conversations}\n{e}")
        return conversations


def _convert_bbox3d_to_quat(conversations: List[Dict], camera_params: Optional[List], new_size: Optional[Tuple]):
    # replace the prefix prompt
    if xyz_prompt in conversations[0]['value']:
        conversations[0]['value'] = conversations[0]['value'].replace(xyz_prompt, quat_prompt)
    for idx, conv in enumerate(conversations):
        if conv['from'] == 'gpt':
            value = conv['value']
            bbox3d_dict_list = decode_bbox3d_dict_from_json(value)
            for bbox3d_dict in bbox3d_dict_list:
                bbox3d = bbox3d_dict['bbox_3d']
                bbox3d = convert_degree_range(bbox3d, from_range='1', to_range='180')
                euler_angle = bbox3d[-3:]
                rotation = Rotation.from_euler(angles=euler_angle, seq='xyz', degrees=True)
                quat = rotation.as_quat()
                quat = [round(x, 2) for x in quat.tolist()]
                bbox3d = bbox3d[:-3] + quat
                bbox3d_dict['bbox_3d'] = copy.deepcopy(bbox3d)
            new_value = encode_json(bbox3d_dict_list)
            conv['value'] = re.sub(r'```json.*?```', new_value, value, flags=re.DOTALL)
    return conversations


FOV_FLAG = "Camera coordinate: X-axis points rightward, Y-axis points downward, and Z-axis points forward. The origin point is the camera location."

def func_remove_intrinsics(conversations: List[Dict]):
    original_prompt = conversations[0]['value']
    if FOV_FLAG in original_prompt:
        splited_prompt = original_prompt.split(FOV_FLAG)
        new_prompt = FOV_FLAG + splited_prompt[1]
        conversations[0]['value'] = new_prompt
    return conversations


prompt_predict_fov = """Here are the camera definition.
Camera coordinate: X-axis points rightward, Y-axis points downward, and Z-axis points forward. The origin point is the camera location.
We take the camera coordinate system as the world coordinate system.

3D bounding box format: [x_center, y_center, z_center, x_size, y_size, z_size, pitch, yaw, roll]
* x_center, y_center, z_center: the center of the object in the camera coordinate, in meters. z_center is the depth of the object in space.
* x_size, y_size, z_size: The dimensions of the object along the ( XYZ ) axes, in meters, when the rotation angles are zero.
* pitch, yaw, roll: Euler angles representing rotations around the X, Y, and Z axes, respectively. Each angle is normalized to the range of (-1, 1) and is multiplied by 180 to convert it into degrees.

For the 3D bounding boxes, output a json list where each entry contains the object name in "label" and its 3D bounding box in "bbox_3d".
"""

predict_fov_question_template = "Predict the hfov (horizontal field of view) of this image and {question}"
predict_fov_answer_template = "hfov={hfov:.2f} degree.\nThe 3D bounding boxes are:\n{original_answer}"


def func_get_hfov(human_value):
    camera_params = get_camera_params(human_value)
    original_width, original_height, fx, fy = camera_params[:4]
    hfov_rad = 2 * math.atan(original_width / (2 * fx))
    hfov = math.degrees(hfov_rad)
    return hfov


def convert_conv_predict_fov(conversations):
    human_value = conversations[0]['value']
    human_value_split = human_value.split('Output a json list where each entry contains the object name in "label" and its 3D bounding box in "bbox_3d"')
    question = human_value_split[-1].strip()
    question = question[0].lower() + question[1:]
    question = predict_fov_question_template.format(question=question)

    gpt_value = conversations[1]['value']
    hfov = func_get_hfov(human_value)
    answer = predict_fov_answer_template.format(hfov=hfov, original_answer=gpt_value)

    conversations[0]['value'] = prompt_predict_fov + question
    conversations[1]['value'] = answer
    return conversations


def debug_draw(image, point):
    from PIL import Image, ImageDraw
    draw = ImageDraw.Draw(image)
    x, y = point
    draw.ellipse((x-2, y-2, x+2, y+2), fill="red")
    image.save("points_visualization.png")
