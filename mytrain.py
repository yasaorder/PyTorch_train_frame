import collections
import os
import random
import time
from pathlib import Path
import simplejson
import numpy
import numpy as np
import matplotlib.pyplot as plt
import cv2
from statistics import mean
from collections import defaultdict
from Eval import eval
import Frame
from segment_anything.utils.transforms import ResizeLongestSide
import torchvision.utils
from tqdm import tqdm
from torch.nn.functional import threshold, normalize
import torch
import modules
from Frame import load_show
from torch.utils.data import Dataset, DataLoader
from Dataset import own_prepare, MyDataset
from tensorboardX import SummaryWriter
import datetime

# ------------------ 训练 ---------------------------
def Mytrain(# model, optimizer, scheduler=None,
            datas, seed, num_epochs, lr, device, wd, loss_name,
            model_type='vit_b', checkpoint='checkpoint/sam_vit_b_01ec64.pth',
            save_dir=None,
            visdual=False,
            load_button=False,
            train_batchsize=2, val_batchsize=4):
    # 导入数据
    [train_ids, val_ids, transformed_data, prompts, ground_truth_masks] = datas
    data_num = len(train_ids) + len(val_ids)

    # -------------------------  训练准备  -------------------------
    # 权重保存目录
    nowtime = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    if save_dir == None:
        save_dir = f'log/epochs{num_epochs}_lr{lr}_wd{wd}_{loss_name}_data{data_num}_{model_type}_seed{seed}_time{nowtime}/'

    # ---------------------- 导入模型 -------------------
    from segment_anything import SamPredictor, sam_model_registry
    sam_model = sam_model_registry[model_type](checkpoint=checkpoint)
    sam_model.to(device)
    sam_model.train()

    # 设置优化器和学习率
    import torch.optim.lr_scheduler as scheduler
    optimizer = torch.optim.Adam(sam_model.mask_decoder.parameters(), lr=lr, weight_decay=wd)
    scheduler = scheduler.ExponentialLR(optimizer, gamma=0.98)

    # 设置损失函数
    loss_fn = torch.nn.MSELoss()
    if loss_name == 'MSE':
        loss_fn = torch.nn.MSELoss()
    elif loss_name == 'Focal':
        loss_fn = modules.B_FocalLoss()
    elif loss_name == 'aaadwdFocal':
        loss_fn = modules.B_FocalLoss()
    elif loss_name == 'BCE':
        loss_fn = torch.nn.BCELoss()

    # 准备结束工作
    from Frame import myframe
    solver = myframe()
    solver.loads(epoch_nums=num_epochs, model=sam_model, scheduler=scheduler, optimizer=optimizer, lr=lr, save_dir=save_dir)
    solver.load_stop_pars()

    # 可视化特征图
    from visdual import visdualer
    visdualer = visdualer(is_visdual=visdual)
    visdualer.is_visdual = visdual

    # ------------------------------------ 训练 ------------------------------------
    for epoch in range(num_epochs):
        epoch_train_losses = []
        epoch_val_losses = []
        epoch_train_accuracy = []
        epoch_val_accuracy = []
        epoch_train_recall = []
        epoch_val_recall = []
        torch.cuda.empty_cache()
        train_num = 0

        # 可视化特征图准备
        visdualer.get_img_to_pred_before(transformed_data[train_ids[0]]['image'].shape[1:],
                                         transformed_data[train_ids[0]]['original_image_size'])

        start_time_train = time.perf_counter()

        for id in train_ids:
            # 将其传入gpu,因为dataset类中返回的是long类型的,避免了前面加上batch的维度
            # [1,c,h,w]
            input_image = transformed_data[id]['image'].to(device)
            input_size = transformed_data[id]['input_size']
            original_image_size = transformed_data[id]['original_image_size']
            ppoints = prompts[id]
            ground_truth_mask = ground_truth_masks[id]

            # 提取图像embedding
            with (torch.no_grad()):
                # 我们希望通过将编码器包装在torch.no_grad()上下文管理器中来嵌入图像，
                # 否则我们将遇到内存问题，同时我们并不打算对image encoder微调。
                # 在该模块下，所有计算得出的tensor的requires_grad都自动设置为False
                # 这样所有的节点在反向传播的时候就不会自动求导了，大大节约了显存。
                # [1,256,64,64]
                image_embedding = sam_model.image_encoder(input_image)
                sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
                    points=ppoints,
                    boxes=None,
                    masks=None,
                )
            # 最后，我们可以生成mask。
            # 请注意，此处我们处于单mask生成模式（与通常输出的3个mask相反）。
            # [1,1,256,256]
            low_res_masks, iou_predictions = sam_model.mask_decoder(
                image_embeddings=image_embedding,
                image_pe=sam_model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )
            # 最后一步这里是将掩码放大回原始图像大小，因为它们的分辨率较低。
            # 我们可以使用Sam.postprocess_masks来实现这一点。
            # 我们还希望从预测的mask中生成二值maks，
            # 以便我们可以将它们与我们的ground truths进行比较。
            # 使用torch函数非常重要，以免破坏反向传播。
            # 以下都是[1,1,512,512]
            upscaled_masks = sam_model.postprocess_masks(low_res_masks, input_size, original_image_size).to(device)
            binary_mask = normalize(threshold(upscaled_masks, 0.0, 0))

            gt_mask_resized = torch.from_numpy(np.resize(ground_truth_mask, (
                1, 1, ground_truth_mask.shape[0], ground_truth_mask.shape[1]))).to(device)
            gt_binary_mask = torch.as_tensor(gt_mask_resized > 0, dtype=torch.float32)

            per_train_loss = loss_fn(binary_mask, gt_binary_mask)
            per_train_accuracy = eval.pixel_accuracy(binary_mask, gt_binary_mask)
            per_train_recall = eval.recall(binary_mask, gt_binary_mask)

            optimizer.zero_grad()
            per_train_loss.backward()
            optimizer.step()

            epoch_train_losses.append(per_train_loss.item())
            epoch_train_accuracy.append(per_train_accuracy)
            epoch_train_recall.append(per_train_recall)

            # 进度条
            load_show(train_num, len(train_ids), start_time_train, legend='train', load_button=load_button)
            train_num = train_num + 1

            # 可视化特征图
            visdualer.get_img_to_pred_load(torch.squeeze(input_image, 0), binary_mask, epoch, 'train')

        # --------------------------- 验证集部分 -----------------------------
        torch.cuda.empty_cache()
        with (torch.no_grad()):
            sam_model.eval()

            visdualer.get_img_to_pred_before(transformed_data[val_ids[0]]['image'].shape[1:], transformed_data[val_ids[0]]['original_image_size'])
            val_num = 0

            start_time_val = time.perf_counter()
            for id in val_ids:
                input_image = transformed_data[id]['image'].to(device)
                input_size = transformed_data[id]['input_size']
                original_image_size = transformed_data[id]['original_image_size']
                ppoints = prompts[id]
                ground_truth_mask = ground_truth_masks[id]

                # 提取图像embedding
                image_embedding = sam_model.image_encoder(input_image)
                sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
                    points=ppoints,
                    boxes=None,
                    masks=None,
                )

                # 最后，我们可以生成mask。
                # 请注意，此处我们处于单mask生成模式（与通常输出的3个mask相反）。
                low_res_masks, iou_predictions = sam_model.mask_decoder(
                    image_embeddings=image_embedding,
                    image_pe=sam_model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False,
                )
                # 最后一步这里是将掩码放大回原始图像大小，因为它们的分辨率较低。
                # 我们可以使用Sam.postprocess_masks来实现这一点。
                # 我们还希望从预测的mask中生成二值maks，
                # 以便我们可以将它们与我们的ground truths进行比较。
                # 使用torch函数非常重要，以免破坏反向传播。
                upscaled_masks = sam_model.postprocess_masks(low_res_masks, input_size, original_image_size).to(device)
                binary_mask = normalize(threshold(upscaled_masks, 0.0, 0))

                gt_mask_resized = torch.from_numpy(np.resize(ground_truth_mask, (
                    1, 1, ground_truth_mask.shape[0], ground_truth_mask.shape[1]))).to(device)
                gt_binary_mask = torch.as_tensor(gt_mask_resized > 0, dtype=torch.float32)

                per_loss_val = loss_fn(binary_mask, gt_binary_mask)
                per_val_accuracy = eval.pixel_accuracy(binary_mask, gt_binary_mask)
                per_val_recall = eval.recall(binary_mask, gt_binary_mask)

                epoch_val_losses.append(per_loss_val.item())
                epoch_val_accuracy.append(per_val_accuracy)
                epoch_val_recall.append(per_val_recall)

                # 进度条
                load_show(val_num, len(val_ids), start_time_val, legend='val', load_button=load_button)
                val_num = val_num + 1

                # 可视化特征图
                visdualer.get_img_to_pred_load(torch.squeeze(input_image, 0), binary_mask, epoch, 'train')

        # 更新
        solver.epoch_step(epoch_train_losses, epoch_val_losses, epoch_train_accuracy, epoch_val_accuracy, epoch_train_recall, epoch_val_recall)
        if solver.is_stop:
            break

        # 可视化loss
        visdualer.get_loss(mean(epoch_train_losses), mean(epoch_val_losses), epoch)

    solver.finsh()
    plt = solver.get_lossfig()
    solver.get_acc_recall_fig()
    solver.get_f1_score()
    solver.get_lrfig()

    visdualer.close()
    return save_dir
    # 一旦我们完成了训练并对性能提升感到满意，我们可以使用:
    # torch.save(model.state_dict(), PATH)
    # 来将调整后的模型的状态字典保存。
    # 然后，在我们希望对与我们用于微调模型的数据相似的数据执行推理时，
    # 我们可以加载此状态字典。

def test(test_dir, save_dir, class_name, device,
            model_type='vit_b', checkpoint='checkpoint/sam_vit_b_01ec64.pth'):

    # Load up the model with default weights
    from segment_anything import sam_model_registry, SamPredictor
    sam_model_orig = sam_model_registry[model_type](checkpoint=checkpoint)
    sam_model_orig.to(device)

    # Set up predictors for both tuned and original models
    tuned_checkpoint = save_dir + 'weight.pth'
    sam_model_tuned = sam_model_registry[model_type](checkpoint=checkpoint)
    sam_model_tuned.to(device)
    Frame.load(sam_model_tuned, tuned_checkpoint)

    predictor_original = SamPredictor(sam_model_orig)
    predictor_tuned = SamPredictor(sam_model_tuned)

    # ----------------------- 测试 --------------------------

    for file in sorted(Path(test_dir+'gt/').iterdir()):
        k = file.stem[:]  # stem代表文件名

        # 导入原始图像
        image = cv2.imread(test_dir + f'image_prompts/{k}.png')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 读取标签的json文件，并保存points至points_lists
        with open(test_dir + f'image_prompts/{k}.json') as f:
            label_data = simplejson.load(f)
        points_list = []
        label_list = []
        for shape in label_data['shapes']:
            points = shape['points']
            points_list.append(points[0])
            if shape['label'] == class_name:
                label_list.append(1.0)
            else:
                label_list.append(0.0)

        points_torch = torch.as_tensor([points_list], dtype=torch.float32, device=device)
        label_torch = torch.as_tensor([label_list], dtype=torch.float32, device=device)

        # 读取GT mask
        gt_grayscale_org = cv2.imread(test_dir + f'gt/{k}.png', cv2.IMREAD_GRAYSCALE)
        gt = (gt_grayscale_org == 255)

        torch.cuda.empty_cache()

        predictor_tuned.set_image(image)
        predictor_original.set_image(image)

        input_points = points_torch.cpu().numpy().reshape(-1, 2)
        input_label = label_torch.cpu().numpy()[0,:]

        masks_tuned, _, _ = predictor_tuned.predict(
            point_coords=input_points,
            point_labels=input_label,
            box=None,
            multimask_output=False,
        )

        masks_orig, _, _ = predictor_original.predict(
            point_coords=input_points,
            point_labels=input_label,
            box=None,
            multimask_output=False,
        )

        # %matplotlib inline
        _, axs = plt.subplots(1, 3, figsize=(24, 8))

        mngr = plt.get_current_fig_manager()  # 获取当前figure manager
        mngr.window.wm_geometry("+0+0")  # 调整窗口在屏幕上弹出的位置
        plt.subplots_adjust(left=0.01, right= 0.99, wspace=0.01)

        axs[0].imshow(image)
        show_mask(masks_tuned, axs[0])
        show_points(input_points, input_label, axs[0])
        axs[0].set_title('Mask with Tuned Model', fontsize=12)
        axs[0].axis('off')

        axs[1].imshow(image)
        show_mask(masks_orig, axs[1])
        show_points(input_points, input_label, axs[1])
        axs[1].set_title('Mask with Untuned Model', fontsize=12)
        axs[1].axis('off')

        axs[2].imshow(image)
        show_mask(gt, axs[2])
        show_points(input_points, input_label, axs[2])
        axs[2].set_title('GT Mask', fontsize=12)
        axs[2].axis('off')

        plt.show()


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
