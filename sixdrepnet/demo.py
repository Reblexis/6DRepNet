import time
import math
import re
import sys
import os
import argparse

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.backends import cudnn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from face_detection import RetinaFace
import matplotlib
from matplotlib import pyplot as plt
from PIL import Image
matplotlib.use('TkAgg')

from model import SixDRepNet
import utils

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description='Head pose estimation using the 6DRepNet.')
    parser.add_argument('--gpu',
                        dest='gpu_id', help='GPU device id to use [0], set -1 to use CPU',
                        default=0, type=int)
    parser.add_argument('--cam',
                        dest='cam_id', help='Camera device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--snapshot',
                        dest='snapshot', help='Name of model snapshot.',
                        default='', type=str)
    parser.add_argument('--save_viz',
                        dest='save_viz', help='Save images with pose cube.',
                        default=False, type=bool)

    args = parser.parse_args()
    return args


transformations = transforms.Compose([transforms.Resize(224),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


class HeadPredictor:
    def __init__(self, snapshot_path: str):
        cudnn.enabled = True
        self.model = SixDRepNet(backbone_name='RepVGG-B1g2',
                        backbone_file='',
                        deploy=True,
                        pretrained=False)


        self.detector = RetinaFace(gpu_id=0)

        # Load snapshot
        saved_state_dict = torch.load(os.path.join(
            snapshot_path), map_location='cpu')

        if 'model_state_dict' in saved_state_dict:
            self.model.load_state_dict(saved_state_dict['model_state_dict'])
        else:
            self.model.load_state_dict(saved_state_dict)
        
        self.device = "cuda:0"
        self.model.to(self.device)

        # Test the Model
        self.model.eval()  # 

    def run(self, frame: np.ndarray) -> dict[str, float]:
        with torch.no_grad():
            faces = self.detector(frame)

            box, landmarks, score = next(((box, landmarks, score) for box, landmarks, score in faces if score > 0.95), (None, None, None))
            if box is None:
                return None

            x_min = int(box[0])
            y_min = int(box[1])
            x_max = int(box[2])
            y_max = int(box[3])
            bbox_width = abs(x_max - x_min)
            bbox_height = abs(y_max - y_min)

            x_min = max(0, x_min-int(0.2*bbox_height))
            y_min = max(0, y_min-int(0.2*bbox_width))
            x_max = x_max+int(0.2*bbox_height)
            y_max = y_max+int(0.2*bbox_width)

            img = frame[y_min:y_max, x_min:x_max]
            img = Image.fromarray(img)
            img = img.convert('RGB')
            img = transformations(img)

            img = torch.Tensor(img[None, :]).to(self.device)

            R_pred = self.model(img)

            euler = utils.compute_euler_angles_from_rotation_matrices(
                R_pred)*180/np.pi
            p_pred_deg = euler[:, 0].item()
            y_pred_deg = euler[:, 1].item()
            r_pred_deg = euler[:, 2].item()

            return {
                'yaw_pred_deg': y_pred_deg,
                'pitch_pred_deg': p_pred_deg,
                'roll_pred_deg': r_pred_deg,
                'x_min': x_min,
                'y_min': y_min,
                'x_max': x_max,
                'y_max': y_max,
                'bbox_width': bbox_width,
            }

    def annotate_frame(self, frame: np.ndarray, info: dict[str, float]):
        if info is None:
            return
        utils.plot_pose_cube(frame,  info['yaw_pred_deg'], info['pitch_pred_deg'], info['roll_pred_deg'], info['x_min'] + int(.5*(
            info['x_max']-info['x_min'])), info['y_min'] + int(.5*(info['y_max']-info['y_min'])), size=info['bbox_width'])


if __name__ == '__main__':
    args = parse_args()
    cudnn.enabled = True
    gpu = args.gpu_id
    if (gpu < 0):
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:%d' % gpu)
    cam = args.cam_id
    snapshot_path = args.snapshot
    model = SixDRepNet(backbone_name='RepVGG-B1g2',
                       backbone_file='',
                       deploy=True,
                       pretrained=False)

    print('Loading data.')

    detector = RetinaFace(gpu_id=gpu)

    # Load snapshot
    saved_state_dict = torch.load(os.path.join(
        snapshot_path), map_location='cpu')

    if 'model_state_dict' in saved_state_dict:
        model.load_state_dict(saved_state_dict['model_state_dict'])
    else:
        model.load_state_dict(saved_state_dict)
    model.to(device)

    # Test the Model
    model.eval()  # Change model to 'eval' mode (BN uses moving mean/var).

    cap = cv2.VideoCapture(cam)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    # Check if the webcam is opened correctly
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    with torch.no_grad():
        while True:
            ret, frame = cap.read()

            faces = detector(frame)

            for box, landmarks, score in faces:

                # Print the location of each face in this image
                if score < .95:
                    continue
                x_min = int(box[0])
                y_min = int(box[1])
                x_max = int(box[2])
                y_max = int(box[3])
                bbox_width = abs(x_max - x_min)
                bbox_height = abs(y_max - y_min)

                x_min = max(0, x_min-int(0.2*bbox_height))
                y_min = max(0, y_min-int(0.2*bbox_width))
                x_max = x_max+int(0.2*bbox_height)
                y_max = y_max+int(0.2*bbox_width)

                img = frame[y_min:y_max, x_min:x_max]
                img = Image.fromarray(img)
                img = img.convert('RGB')
                img = transformations(img)

                img = torch.Tensor(img[None, :]).to(device)

                c = cv2.waitKey(1)
                if c == 27:
                    break

                start = time.time()
                R_pred = model(img)
                end = time.time()
                print('Head pose estimation: %2f ms' % ((end - start)*1000.))

                euler = utils.compute_euler_angles_from_rotation_matrices(
                    R_pred)*180/np.pi
                p_pred_deg = euler[:, 0].cpu()
                y_pred_deg = euler[:, 1].cpu()
                r_pred_deg = euler[:, 2].cpu()

                #utils.draw_axis(frame, y_pred_deg, p_pred_deg, r_pred_deg, left+int(.5*(right-left)), top, size=100)
                utils.plot_pose_cube(frame,  y_pred_deg, p_pred_deg, r_pred_deg, x_min + int(.5*(
                    x_max-x_min)), y_min + int(.5*(y_max-y_min)), size=bbox_width)

            cv2.imshow("Demo", frame)
            cv2.waitKey(5)
