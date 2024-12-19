import os, sys; sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from sixdrepnet.regressor import SixDRepNet_Detector as SixDRepNet
from sixdrepnet import backbone
from sixdrepnet import loss
from sixdrepnet import model
from sixdrepnet import utils


"""
6DRepNet.

Accurate and unconstrained head pose estimation.
"""

__version__ = "0.1.6"
__author__ = 'Thorsten Hempel'
