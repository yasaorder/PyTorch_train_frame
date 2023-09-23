import glob

import numpy as np
import matplotlib.pyplot as plt
import os
import torchvision.transforms as transforms
join = os.path.join
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
# import monai
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
import random
import cv2
from collections import defaultdict
from segment_anything.utils.transforms import ResizeLongestSide

torch.manual_seed(2023)

def randomHueSaturationValue(image, hue_shift_limit=(-180, 180),
                             sat_shift_limit=(-255, 255),
                             val_shift_limit=(-255, 255), u=0.5):
    if np.random.random() < u:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(image)
        hue_shift = np.random.randint(hue_shift_limit[0], hue_shift_limit[1]+1)
        hue_shift = np.uint8(hue_shift)
        h += hue_shift
        sat_shift = np.random.uniform(sat_shift_limit[0], sat_shift_limit[1])
        s = cv2.add(s, sat_shift)
        val_shift = np.random.uniform(val_shift_limit[0], val_shift_limit[1])
        v = cv2.add(v, val_shift)
        image = cv2.merge((h, s, v))
        #image = cv2.merge((s, v))
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    return image

def randomShiftScaleRotate(image, mask,
                           shift_limit=(-0.0, 0.0),
                           scale_limit=(-0.0, 0.0),
                           rotate_limit=(-0.0, 0.0),
                           aspect_limit=(-0.0, 0.0),
                           borderMode=cv2.BORDER_CONSTANT, u=0.5):
    if np.random.random() < u:
        height, width, channel = image.shape

        angle = np.random.uniform(rotate_limit[0], rotate_limit[1])
        scale = np.random.uniform(1 + scale_limit[0], 1 + scale_limit[1])
        aspect = np.random.uniform(1 + aspect_limit[0], 1 + aspect_limit[1])
        sx = scale * aspect / (aspect ** 0.5)
        sy = scale / (aspect ** 0.5)
        dx = round(np.random.uniform(shift_limit[0], shift_limit[1]) * width)
        dy = round(np.random.uniform(shift_limit[0], shift_limit[1]) * height)

        cc = np.math.cos(angle / 180 * np.math.pi) * sx
        ss = np.math.sin(angle / 180 * np.math.pi) * sy
        rotate_matrix = np.array([[cc, -ss], [ss, cc]])

        box0 = np.array([[0, 0], [width, 0], [width, height], [0, height], ])
        box1 = box0 - np.array([width / 2, height / 2])
        box1 = np.dot(box1, rotate_matrix.T) + np.array([width / 2 + dx, height / 2 + dy])

        box0 = box0.astype(np.float32)
        box1 = box1.astype(np.float32)
        mat = cv2.getPerspectiveTransform(box0, box1)
        image = cv2.warpPerspective(image, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                    borderValue=(
                                        0, 0,
                                        0,))
        mask = cv2.warpPerspective(mask, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                   borderValue=(
                                       0, 0,
                                       0,))

    return image, mask

# 随机水平翻转
def randomHorizontalFlip(image, mask, u=0.5):
    if np.random.random() < u:
        image = cv2.flip(image, 1)
        mask = cv2.flip(mask, 1)

    return image, mask

#随机垂直翻转
def randomVerticleFlip(image, mask, u=0.5):
    if np.random.random() < u:
        image = cv2.flip(image, 0)
        mask = cv2.flip(mask, 0)

    return image, mask

def randomRotate90(image, mask, u=0.5):
    if np.random.random() < u:
        image=np.rot90(image)
        mask=np.rot90(mask)

    return image, mask

# 数据导入，做了一定的数据增强
def default_loader(id, root):
    img = cv2.imread(os.path.join(root,'{}.png').format(id))
    mask = cv2.imread(os.path.join(root+'{}.png').format(id), cv2.IMREAD_GRAYSCALE)

    img = randomHueSaturationValue(img,
                                   hue_shift_limit=(-30, 30),
                                   sat_shift_limit=(-5, 5),
                                   val_shift_limit=(-15, 15))

    img, mask = randomShiftScaleRotate(img, mask,
                                       shift_limit=(-0.1, 0.1),
                                       scale_limit=(-0.1, 0.1),
                                       aspect_limit=(-0.1, 0.1),
                                       rotate_limit=(-0, 0))
    img, mask = randomRotate90(img, mask)

    mask = np.expand_dims(mask, axis=2)
    img = np.array(img, np.float32).transpose(2,0,1)/255.0 * 3.2 - 1.6
    mask = np.array(mask, np.float32).transpose(2,0,1)/255.0
    mask[mask>=0.5] = 1
    mask[mask<=0.5] = 0
    #mask = abs(mask-1)
    return img, mask

from pathlib import Path
import simplejson

def own_prepare(data_path ,label_path, device, class_name,
                model_type = 'vit_b',
                checkpoint = 'checkpoint/sam_vit_b_01ec64.pth',
                data_num=-1,
                train_val_p=0.9):
    """
    对数据进行预处理，image encoder提取embedding存入npz
    """
    prompts = {}
    ground_truth_masks = {}
    ids = []

    # 采用sam的内置预处理函数
    from segment_anything import SamPredictor, sam_model_registry
    sam_model = sam_model_registry[model_type](checkpoint=checkpoint)
    sam_model.to(device)

    # 图像尺寸转换器
    transform = ResizeLongestSide(sam_model.image_encoder.img_size)
    transformed_data = defaultdict(dict)

    # 载入提示文件,转为张量并保存,载入gt,转为二值单通道
    for file in sorted(Path(label_path).iterdir()):
        k = file.stem[:]  # stem代表文件名
        ids.append(k)
        # 读取标签的json文件，并保存points至points_lists
        with open(data_path + f'{k}.json') as f:
            label_data = simplejson.load(f)
        points_list = []
        label_list = []
        for shape in label_data['shapes']:
            points = shape['points']
            label = shape['label']
            points_list.append(points[0])
            if label == class_name:
                label_list.append(1.0)
            else:
                label_list.append(0.0)

        points_torch = torch.as_tensor([points_list], dtype=torch.float32, device=device)
        label_torch = torch.as_tensor([label_list], dtype=torch.float32, device=device)
        coord = (points_torch, label_torch)
        prompts[k] = coord

        gt_grayscale_org = cv2.imread(label_path + f'{k}.png', cv2.IMREAD_GRAYSCALE)
        ground_truth_masks[k] = (gt_grayscale_org == 255)

    # ------------------ 预处理原始图像 ----------------------

        # 导入原始图像
        image = cv2.imread(data_path + f'{k}.png')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 将图像修改为适合模型的尺寸---->(1024,1024)
        # 转为[H,W,C]数组
        input_image = transform.apply_image(image)
        # cv2.imshow("image", input_image)
        input_image_torch = torch.as_tensor(input_image, device=device)
        # 重排[H,W,C],加上batchsize维度->[b,C,H,w]
        transformed_image = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]

        # 提取图像embedding
        input_image = sam_model.preprocess(transformed_image, )
        original_image_size = image.shape[:2]
        input_size = tuple(transformed_image.shape[-2:])

        transformed_data[k]['image'] = input_image
        transformed_data[k]['input_size'] = input_size
        transformed_data[k]['original_image_size'] = original_image_size
        torch.cuda.empty_cache()

    if data_num == -1:
        data_num = int(len(ids))
    train_ids = ids[0:int(data_num * train_val_p)]
    val_ids = ids[int(data_num * train_val_p):data_num]

    print(f"载入数据完毕, 划分训练集{len(train_ids)}张, 验证集{len(val_ids)}张")

    return [train_ids, val_ids, transformed_data, prompts, ground_truth_masks]

# 构造自己的Dataset继承Dataset类
class MyDataset(Dataset):
    def __init__(self, data_root, label_root, device, type='train', p=0.9):
        # 预处理后的所有数据,输出[ids, 原始图像(张量), 提示(张量), 真实标签(单通道二值图)]
        self.device = device
        self.sum_data = own_prepare(data_root, label_root, self.device)
        self.ids = self.sum_data[0]
        self.type = type
        if self.type == 'train':
            self.ids = self.sum_data[0][0:int(len(self.ids)*p)]
        elif self.type == 'val':
            self.ids = self.sum_data[0][int(len(self.ids) * p):]

        self.transformed_data = self.sum_data[1]
        self.prompts = self.sum_data[2]
        self.ground_truth_masks = self.sum_data[3]
        print("载入数据\n")

    def __getitem__(self, index):
        # input_image = self.transformed_data[self.ids[index]]['image'].to(self.device)
        # input_size = self.transformed_data[self.ids[index]]['input_size']
        # original_image_size = self.transformed_data[self.ids[index]]['original_image_size']
        # ppoints = self.prompts[self.ids[index]]
        # ground_truth_mask = self.ground_truth_masks[self.ids[index]]

        return self.transformed_data[self.ids[index]]

    def __len__(self):
        return len(self.ids)
