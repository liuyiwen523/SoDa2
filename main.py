from __future__ import print_function
import os

import torch.optim

os.environ["OMP_NUM_THREADS"] = "1"

from torch.autograd import Variable
import argparse
import time
from utils.utils import getDevice
import models.SODA2 as SODA
from mmd_mill import *
from utils.dataLoader import getDataLoader
from utils.utils import getDatasetInfo
from sklearn.mixture import GaussianMixture
import random

def get_args():
    parser = argparse.ArgumentParser(description='SODA2')
    parser.add_argument('--source_dataset', choices=['Houston13_7gt', 'PaviaU_7gt', 'Yancheng_ZY'],default='PaviaU_7gt')
    parser.add_argument('--target_dataset', choices=['Houston18_OS', 'PaviaC_OS', 'Yancheng_GF'],default='Yancheng_GF')
    parser.add_argument('--patch', type=int, default=7)
    parser.add_argument('--seed', type=int, default=179, help='random seed (default: 0)')
    parser.add_argument('--train_num', type=int, default=180)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--bottle_neck_dim', type=int, default=256)
    parser.add_argument('--bottle_neck_dim2', type=int, default=500)
    parser.add_argument('--training_iter', type=int, default=50)

    parser.add_argument('--a',  default=1)
    parser.add_argument('--b',  default=10)
    parser.add_argument('--c',  default=0)
    parser.add_argument('--d',  default=1)
    parser.add_argument('--e',  default=1)
    parser.add_argument('--k',  default=2)

    parser.add_argument('--ls_eps', type=float, default=0.001)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    parser.add_argument('--log_interval',  default=10)

    parser.add_argument('--is_save_pt', default=False)
    parser.add_argument('--is_save_jpg', default=False)
    parser.add_argument('--save_root', type=str, default='result//')
    parser.add_argument('--set_gpu', type=int, default=0, help='gpu setting 0 or 1')

    try:
        args = parser.parse_args()
    except:
        args, _ = parser.parse_known_args()

    args.device = getDevice(args.set_gpu)

    if args.source_dataset == 'PaviaU_7gt' and args.target_dataset == 'PaviaC_OS':
        args.source_known_classes = [1,2,3,4,5,6,7]
        args.target_known_classes = [1,2,3,4,5,6,7]
        args.target_unknown_classes = [9]
        args.chan_num = 102
        args.out_dim = 7
    elif args.source_dataset == 'Houston13_7gt' and args.target_dataset == 'Houston18_OS':
        args.source_known_classes = [1,2,3,4,5,6,7]
        args.target_known_classes = [1,2,3,4,5,6,7]
        args.target_unknown_classes = [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
        args.chan_num = 48
        args.out_dim = 7
    elif args.source_dataset == 'Yancheng_ZY' and args.target_dataset == 'Yancheng_GF':
        args.source_known_classes = [1, 6, 7]
        args.target_known_classes = [1, 6, 7]
        args.target_unknown_classes = [2, 3, 4, 5]
        args.chan_num = 147
        args.out_dim = 3

    return args

def train(model):
    print('Train Starts')

    src_train_loader = data_loader['source']['train']
    target_train_loader = data_loader['target']['train']
    optimizer = torch.optim.SGD(model.parameters(), lr=args.ls_eps, weight_decay=args.weight_decay, momentum = args.momentum)
    for epoch in range(1, args.training_iter):
        joint_loader = zip(src_train_loader, target_train_loader)
        alpha = float((float(2) / (1 + np.exp(-10 * float((float(epoch) / float(args.training_iter)))))) - 1)
        for batch_idx, ((img_s,label_s), (img_t, label_t)) in enumerate(joint_loader):
            model.train()

            img_s, label_s = img_s.to(args.device), label_s.to(args.device)
            img_t = img_t.to(args.device)
            img_s, label_s = Variable(img_s), Variable(label_s)
            img_t = Variable(img_t)

            loss = model(args, img_s, label_s, img_t)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    acc_hos = test(model,args.training_iter)
    return acc_hos

def to_np(x):
    return x.squeeze().cpu().detach().numpy()

def test(model,epoch):
    device = args.device
    model.eval()

    all_diff = []
    all_pred_t = []
    all_label_t = []

    with torch.no_grad():
        for batch_idx, (img_t, label_t) in enumerate(data_loader['target']['test']):
            img_t = img_t.to(device)
            label_t = label_t.to(device)

            out_t, diff = model(args, img_t)
            pred_t = out_t.data.max(1)[1]

            all_diff.append(diff.cpu())
            all_pred_t.append(pred_t.cpu())
            all_label_t.append(label_t.cpu())

    total_diff = torch.cat(all_diff, dim=0)
    y_pred = torch.cat(all_pred_t, dim=0)
    y_true = torch.cat(all_label_t, dim=0)

    diff_reshaped = total_diff.cpu().numpy()[:, None]
    gmm = GaussianMixture(n_components=args.k, covariance_type='full').fit(diff_reshaped)
    unk_cluster = np.argmax(gmm.means_)
    gmm_index = gmm.predict(diff_reshaped)
    gmm_index_tensor = torch.tensor(gmm_index)
    unk_mask = (gmm_index_tensor == unk_cluster)
    unknown_class_value = len(args.source_known_classes)

    y_pred[unk_mask] = unknown_class_value

    y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().numpy()

    num_class = len(args.source_known_classes) + 1
    known_num_class = num_class - 1
    max_target_label = int(np.max(y_pred) + 1)

    class_correct = np.zeros(num_class)
    class_total = np.zeros(num_class)

    for pred, true in zip(y_pred, y_true):
        class_total[true] += 1
        if pred == true:
            class_correct[true] += 1

    classes_acc = [class_correct[i] / class_total[i] if class_total[i] > 0 else 0.0
                   for i in range(known_num_class)]

    m = extended_confusion_matrix(y_true, y_pred, true_labels=list(range(max_target_label)),
                                         pred_labels=list(range(num_class)))
    cm = m
    cm = cm.astype(np.float64) / np.sum(cm, axis=1, keepdims=True)
    acc_os_star = sum([cm[i][i] for i in range(known_num_class)]) / known_num_class
    acc_unknown = sum(
        [cm[i][known_num_class] for i in range(known_num_class, int(np.max(y_true) + 1))]) / (
                          max_target_label - known_num_class)
    acc_hos = (2 * acc_os_star * acc_unknown) / (acc_os_star + acc_unknown)

    # -------------------------- 5. 输出结果 --------------------------
    print("\n===== 评价指标结果 =====")
    print("epoch    OS*    unkn    HOS")
    print(f"{epoch}    {acc_os_star:.3f}    {acc_unknown:.3f}    {acc_hos:.3f}")
    print(" " + ", ".join([f"{acc:.3f}" for acc in classes_acc]))
    print("======================================")
    return acc_hos

def draw(model,name):
    device = args.device
    model.eval()

    prediction_list = []

    with torch.no_grad():
        for batch_idx, img_t in enumerate(data_loader['target']['all']):
            img_t = img_t.to(device)

            out_t, diff = model(args, img_t)
            pred_t = out_t.data.max(1)[1]

            # gmm
            diff_reshaped = diff.cpu().numpy()[:, None]
            gmm = GaussianMixture(n_components=2, covariance_type='full').fit(diff_reshaped)
            unk_cluster = np.argmax(gmm.means_)
            gmm_index = gmm.predict(diff_reshaped)
            gmm_index_tensor = torch.tensor(gmm_index)
            unk_mask = (gmm_index_tensor == unk_cluster)
            unknown_class_value = len(args.source_known_classes)

            pred_t[unk_mask] = unknown_class_value
            prediction_list.append(pred_t.cpu())


    from utils.draw import drawPredictionMap,drawGTMap

    drawPredictionMap(prediction_list, name, target_info,
                      args.target_known_classes, args.target_unknown_classes, False)
    drawGTMap(args.target_dataset, path = 'map', known_classes = args.target_known_classes, unknown_classes = args.target_unknown_classes, unknown_color = [255, 255, 255])

    return

def setup_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    args = get_args()
    setup_seed(args.seed)

    source_info = getDatasetInfo(args.source_dataset)
    target_info = getDatasetInfo(args.target_dataset)
    data_loader: dict = getDataLoader(args, source_info, target_info)
    model = OSDA.OSDA(args.chan_num, args.patch, in_dim=288, out_dim=args.out_dim,
                      bottle_neck_dim=args.bottle_neck_dim).to(args.device)
    save_path = args.save_root + '/{}'.format(args.source_dataset + args.target_dataset)
    state_dict = torch.load(save_path, map_location=args.device)
    model.load_state_dict(state_dict)
    test(model, args.training_iter)
    name = f'{args.source_dataset} {args.target_dataset}'
    if args.is_save_jpg:
        draw(model,name)















