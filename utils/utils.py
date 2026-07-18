import json
import torch
import random
import os
import numpy as np
from scipy.io import loadmat
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from .pyExt import Dict2Obj

def getDatasetInfo(dataset):
    with open("datasets/dataset_config.json", "r") as f:
        info = json.load(f)[dataset]

    return Dict2Obj(info)

def getDataByInfo(info):
    dataset_path = os.path.join('./datasets', info.path)

    if info.type is None:
        data = loadmat(os.path.join(dataset_path, info.file_name))[info.mat_name].astype(np.float32)
    elif info.type == 'npy':
        data = np.load(os.path.join(dataset_path, info.file_name)).astype(np.float32)

    return data

def getGTByInfo(info):
    dataset_path = os.path.join('./datasets', info.path)

    if info.type is None:
        gt = loadmat(os.path.join(dataset_path, info.gt_file_name))[info.gt_mat_name].astype(np.int64)
    elif info.type == 'npy':
        gt = np.load(os.path.join(dataset_path, info.gt_file_name)).astype(np.int64)
    
    return gt

def seed_torch(seed):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True

def getDevice(device=None):
    if device is None:
        if torch.cuda.is_available():
            return torch.device('cuda')
        else:
            return torch.device('cpu')
    elif device == -1:
        return torch.device('cpu')
    else:
        return torch.device(f'cuda:{device}')



def extended_confusion_matrix(y_true, y_pred, true_labels=None, pred_labels=None):
    if not true_labels:
        true_labels = sorted(list(set(list(y_true))))
    true_label_to_id = {x: i for (i, x) in enumerate(true_labels)}
    if not pred_labels:
        pred_labels = true_labels
    pred_label_to_id = {x: i for (i, x) in enumerate(pred_labels)}
    confusion_matrix = np.zeros([len(true_labels), len(pred_labels)])
    for (true, pred) in zip(y_true, y_pred):
        confusion_matrix[true_label_to_id[true]][pred_label_to_id[pred]] += 1.0
    return confusion_matrix