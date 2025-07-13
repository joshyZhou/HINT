from ast import arg
import numpy as np
import os
import argparse
from tqdm import tqdm
import cv2

import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import utils

from natsort import natsorted
from glob import glob
from skimage import img_as_ubyte
from pdb import set_trace as stx
from skimage import metrics

from basicsr.models import create_model
from basicsr.utils.options import dict2str, parse

parser = argparse.ArgumentParser(
    description='Image Enhancement using HINT')

parser.add_argument('--input_dir', default='./dataset/LOLv2/Real_captured/Test/Low',
                    type=str, help='Directory of validation images')
parser.add_argument('--result_dir', default='./results/HINT/',
                    type=str, help='Directory for results')
parser.add_argument(
    '--opt', type=str, default='./Options/HINT_LOL_v2_real.yml', help='Path to option YAML file.')
parser.add_argument('--weights', default='./models/lolv2Real.pth',
                    type=str, help='Path to weights')
parser.add_argument('--dataset', default='LOL_v2_real', type=str,
                    help='Test Dataset') 
parser.add_argument('--gpus', type=str, default="1", help='GPU devices.')
parser.add_argument('--GT_mean', action='store_true', help='Use the mean of GT to rectify the output of the model')

args = parser.parse_args()

def splitimage(imgtensor, crop_size=128, overlap_size=64):
    _, C, H, W = imgtensor.shape
    hstarts = [x for x in range(0, H, crop_size - overlap_size)]
    while hstarts and hstarts[-1] + crop_size >= H:
        hstarts.pop()
    hstarts.append(H - crop_size)
    wstarts = [x for x in range(0, W, crop_size - overlap_size)]
    while wstarts and wstarts[-1] + crop_size >= W:
        wstarts.pop()
    wstarts.append(W - crop_size)
    starts = []
    split_data = []
    for hs in hstarts:
        for ws in wstarts:
            cimgdata = imgtensor[:, :, hs:hs + crop_size, ws:ws + crop_size]
            starts.append((hs, ws))
            split_data.append(cimgdata)
    return split_data, starts

def get_scoremap(H, W, C, B=1, is_mean=True):
    center_h = H / 2
    center_w = W / 2

    score = torch.ones((B, C, H, W))
    if not is_mean:
        for h in range(H):
            for w in range(W):
                score[:, :, h, w] = 1.0 / (math.sqrt((h - center_h) ** 2 + (w - center_w) ** 2 + 1e-6))
    return score

def mergeimage(split_data, starts, crop_size = 128, resolution=(1, 3, 128, 128)):
    B, C, H, W = resolution[0], resolution[1], resolution[2], resolution[3]
    tot_score = torch.zeros((B, C, H, W))
    merge_img = torch.zeros((B, C, H, W))
    scoremap = get_scoremap(crop_size, crop_size, C, B=B, is_mean=True)
    for simg, cstart in zip(split_data, starts):
        hs, ws = cstart
        merge_img[:, :, hs:hs + crop_size, ws:ws + crop_size] += scoremap * simg
        tot_score[:, :, hs:hs + crop_size, ws:ws + crop_size] += scoremap
    merge_img = merge_img / tot_score
    return merge_img


# 指定 gpu
gpu_list = ','.join(str(x) for x in args.gpus)
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
print('export CUDA_VISIBLE_DEVICES=' + gpu_list)

####### Load yaml #######
yaml_file = args.opt
weights = args.weights
print(f"dataset {args.dataset}")

import yaml

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

opt = parse(args.opt, is_train=False)
opt['dist'] = False


x = yaml.load(open(args.opt, mode='r'), Loader=Loader)
s = x['network_g'].pop('type')
##########################


model_restoration = create_model(opt).net_g

# 加载模型
checkpoint = torch.load(weights)

try:
    model_restoration.load_state_dict(checkpoint['params'])
except:
    new_checkpoint = {}
    for k in checkpoint['params']:
        new_checkpoint['module.' + k] = checkpoint['params'][k]
    model_restoration.load_state_dict(new_checkpoint)

print("===>Testing using weights: ", weights)
model_restoration.cuda()
model_restoration = nn.DataParallel(model_restoration)
model_restoration.eval()

# 生成输出结果的文件
factor = 4
dataset = args.dataset
config = os.path.basename(args.opt).split('.')[0]
checkpoint_name = os.path.basename(args.weights).split('.')[0]
result_dir = os.path.join(args.result_dir, dataset, config, checkpoint_name)
result_dir_input = os.path.join(args.result_dir, dataset, 'input')
result_dir_gt = os.path.join(args.result_dir, dataset, 'gt')
# stx()
os.makedirs(result_dir, exist_ok=True)

psnr = []
ssim = []
if dataset in ['SID', 'SMID', 'SDSD_indoor', 'SDSD_outdoor']:
    os.makedirs(result_dir_input, exist_ok=True)
    os.makedirs(result_dir_gt, exist_ok=True)
    if dataset == 'SID':
        from basicsr.data.SID_image_dataset import Dataset_SIDImage as Dataset
    elif dataset == 'SMID':
        from basicsr.data.paired_image_dataset import Dataset_SMIDImage as Dataset
    else:
        from basicsr.data.SDSD_image_dataset import Dataset_SDSDImage as Dataset
    opt = opt['datasets']['val']
    opt['phase'] = 'test'
    if opt.get('scale') is None:
        opt['scale'] = 1
    if '~' in opt['dataroot_gt']:
        opt['dataroot_gt'] = os.path.expanduser('~') + opt['dataroot_gt'][1:]
    if '~' in opt['dataroot_lq']:
        opt['dataroot_lq'] = os.path.expanduser('~') + opt['dataroot_lq'][1:]
    dataset = Dataset(opt)
    print(f'test dataset length: {len(dataset)}')
    dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)
    with torch.inference_mode():
        for data_batch in tqdm(dataloader):
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()

            input_ = data_batch['lq']
            input_save = data_batch['lq'].cpu().permute(
                0, 2, 3, 1).squeeze(0).numpy()
            target = data_batch['gt'].cpu().permute(
                0, 2, 3, 1).squeeze(0).numpy()
            inp_path = data_batch['lq_path'][0]

            B, C, H, W = input_.shape
            corp_size_arg = 400
            overlap_size_arg = 384
            split_data, starts = splitimage(input_, crop_size=corp_size_arg, overlap_size=overlap_size_arg)
            for i, data in enumerate(split_data):
                split_data[i] = model_restoration(data).cpu()
            restored = mergeimage(split_data, starts, crop_size=corp_size_arg, resolution=(B, C, H, W))


            restored = torch.clamp(restored, 0, 1).cpu(
            ).detach().permute(0, 2, 3, 1).squeeze(0).numpy()

            if args.GT_mean:
                # This test setting is the same as KinD, LLFlow, and recent diffusion models
                # Please refer to Line 73 (https://github.com/zhangyhuaee/KinD/blob/master/evaluate_LOLdataset.py)
                mean_restored = cv2.cvtColor(restored.astype(np.float32), cv2.COLOR_BGR2GRAY).mean()
                mean_target = cv2.cvtColor(target.astype(np.float32), cv2.COLOR_BGR2GRAY).mean()
                restored = np.clip(restored * (mean_target / mean_restored), 0, 1)

            psnr.append(utils.PSNR(target, restored))
            ssim.append(utils.calculate_ssim(
                img_as_ubyte(target), img_as_ubyte(restored)))
            type_id = os.path.dirname(inp_path).split('/')[-1]
            os.makedirs(os.path.join(result_dir, type_id), exist_ok=True)
            os.makedirs(os.path.join(result_dir_input, type_id), exist_ok=True)
            os.makedirs(os.path.join(result_dir_gt, type_id), exist_ok=True)
            utils.save_img((os.path.join(result_dir, type_id, os.path.splitext(
                os.path.split(inp_path)[-1])[0] + '.png')), img_as_ubyte(restored))
            utils.save_img((os.path.join(result_dir_input, type_id, os.path.splitext(
                os.path.split(inp_path)[-1])[0] + '.png')), img_as_ubyte(input_save))
            utils.save_img((os.path.join(result_dir_gt, type_id, os.path.splitext(
                os.path.split(inp_path)[-1])[0] + '.png')), img_as_ubyte(target))
else:

    input_dir = opt['datasets']['val']['dataroot_lq']
    target_dir = opt['datasets']['val']['dataroot_gt']
    print(input_dir)
    print(target_dir)

    input_paths = natsorted(
        glob(os.path.join(input_dir, '*.png')) + glob(os.path.join(input_dir, '*.jpg')))

    target_paths = natsorted(glob(os.path.join(
        target_dir, '*.png')) + glob(os.path.join(target_dir, '*.jpg')))

    with torch.inference_mode():
        for inp_path, tar_path in tqdm(zip(input_paths, target_paths), total=len(target_paths)):

            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()

            img = np.float32(utils.load_img(inp_path)) / 255.
            target = np.float32(utils.load_img(tar_path)) / 255.

            img = torch.from_numpy(img).permute(2, 0, 1)
            input_ = img.unsqueeze(0).cuda()

            B, C, H, W = input_.shape
            corp_size_arg = 384
            overlap_size_arg = 350

            split_data, starts = splitimage(input_, crop_size=corp_size_arg, overlap_size=overlap_size_arg)
            for i, data in enumerate(split_data):
                split_data[i] = model_restoration(data).cpu()
            restored = mergeimage(split_data, starts, crop_size=corp_size_arg, resolution=(B, C, H, W))


            restored = torch.clamp(restored, 0, 1).cpu(
            ).detach().permute(0, 2, 3, 1).squeeze(0).numpy()

            if args.GT_mean:
                # This test setting is the same as KinD, LLFlow, and recent diffusion models
                # Please refer to Line 73 (https://github.com/zhangyhuaee/KinD/blob/master/evaluate_LOLdataset.py)
                mean_restored = cv2.cvtColor(restored.astype(np.float32), cv2.COLOR_BGR2GRAY).mean()
                mean_target = cv2.cvtColor(target.astype(np.float32), cv2.COLOR_BGR2GRAY).mean()
                restored = np.clip(restored * (mean_target / mean_restored), 0, 1)

            psnr.append(utils.PSNR(target, restored))
            ssim.append(utils.calculate_ssim(
                img_as_ubyte(target), img_as_ubyte(restored)))
            utils.save_img((os.path.join(result_dir, os.path.splitext(
                os.path.split(inp_path)[-1])[0] + '.png')), img_as_ubyte(restored))

psnr = np.mean(np.array(psnr))
ssim = np.mean(np.array(ssim))
print("PSNR: %f " % (psnr))
print("SSIM: %f " % (ssim))
