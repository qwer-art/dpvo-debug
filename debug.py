import os
from multiprocessing import Process, Queue
from pathlib import Path

import cv2
import numpy as np
import torch
from evo.core.trajectory import PoseTrajectory3D
from evo.tools import file_interface
from evo.tools.user import prompt_val

from dpvo.config import cfg
from dpvo.dpvo import DPVO
from dpvo.plot_utils import plot_trajectory, save_output_for_COLMAP, save_ply
from dpvo.stream import image_stream, video_stream
from dpvo.utils import Timer
from dpvo.lietorch import SE3
from dpvo.debug_utils import *

def debug():
    # 将位姿数据转换为张量，并指定设备和数据类型
    frame_pose1 = torch.tensor([0.8208, 0.5111, -3.25, -0.0827, -0.0335, 0.0105, 0.996], device="cuda",
                               dtype=torch.float32)
    frame_pose2 = torch.tensor([1.0285, 0.5172, -3.3163, -0.0849, -0.0706, 0.0053, 0.9939], device="cuda",
                               dtype=torch.float32)

    # 现在用张量创建 SE3 对象
    P1 = SE3(frame_pose1)
    P2 = SE3(frame_pose2)
    x2 = (P1 * P2.inv()).log()
    print(f"x2: {x2}")

    P = SE3.exp(x2) * P1
    x1 = (P * P1.inv()).log()
    print(f"x1: {x1}")

if __name__ == '__main__':
    debug()