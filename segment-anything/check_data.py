from pathlib import Path
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy
import simplejson
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
import tkinter as tk

# 对数据进行预处理，image encoder提取embedding存入npz
class show_data:
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
        pos_points = coords[labels == 1]
        neg_points = coords[labels == 0]
        ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
                   linewidth=1.25)
        ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
                   linewidth=1.25)

class check_Data:
    def __init__(self, image_dir, gt_dir, prompt_dir):
        self.image_dir = image_dir
        self.gt_dir =gt_dir
        self.prompt_dir = prompt_dir
        self.ids = []
        self.images_list = {}
        self.points_list = {}
        self.labels_list = {}
        self.gts_list = {}
        self.class_name = '__background__'

    def check_data(self, class_name):
        self.class_name = class_name

        for file in sorted(Path(self.gt_dir).iterdir()):
            k = file.stem[:]  # stem代表文件名
            self.ids.append(k)

            self.load_per_img(k)
            self.load_per_json(k)
            self.load_per_gt(k)

        self.check_1()

    def check_1(self):
        if len(self.labels_list) % 2 != 0:
            print("数据集数量应为偶数")
            return 1

        for i in range(0, len(self.ids)):
            figure = plt.figure(figsize=(8, 10))

            mngr = plt.get_current_fig_manager()  # 获取当前figure manager
            mngr.window.wm_geometry("+0+0")  # 调整窗口在屏幕上弹出的位置
            plt.subplots_adjust(left=0.01, right=0.99, wspace=0.01)

            k1 = self.ids[i]

            plt.imshow(self.images_list[k1])
            self.show_mask(self.gts_list[k1], plt)
            self.show_points(self.points_list[k1], self.labels_list[k1], plt)
            plt.title(f'--image--{k1}', fontsize=12)
            plt.axis('off')

            # 将图形显示在左半部分
            # root = tk.Tk()
            # root.geometry("1800x900")  # 指定窗口大小
            # frame_left = tk.Frame(root, width=800, height=900, bg='#FFF')
            # frame_left.pack(side=tk.LEFT)
            # canvas = FigureCanvasTkAgg(figure, master=frame_left)
            # canvas.draw()
            # canvas.get_tk_widget().pack()

            # 显示窗口
            plt.show()
            # root.mainloop()


    def check_2(self):
        if len(self.labels_list) % 2 != 0:
            print("数据集数量应为偶数")
            return 1

        for i in range(0, len(self.ids),2):
            # %matplotlib inline
            _, axs = plt.subplots(1, 2, figsize=(24, 8))

            mngr = plt.get_current_fig_manager()  # 获取当前figure manager
            mngr.window.wm_geometry("+0+0")  # 调整窗口在屏幕上弹出的位置
            plt.subplots_adjust(left=0.01, right=0.99, wspace=0.01)

            k1 = self.ids[i]
            k2 = self.ids[i+1]

            axs[0].imshow(self.images_list[k1])
            self.show_mask(self.gts_list[k1], axs[0])
            self.show_points(self.points_list[k1], self.labels_list[k1], axs[0])
            axs[0].set_title(f'--image--{k1}', fontsize=12)
            axs[0].axis('off')

            axs[1].imshow(self.images_list[k2])
            self.show_mask(self.gts_list[k2], axs[1])
            self.show_points(self.points_list[k2], self.labels_list[k2], axs[1])
            axs[1].set_title(f'--image--{k2}', fontsize=12)
            axs[1].axis('off')

            plt.show()

    def check_per6(self):
        # 循环遍历，每次取6张图片
        x = 3
        for i in range(0, len(self.images_list), x):
            # 取x张图片并过滤掉为空的图片
            images = list(filter(None, self.images_list[i:i + x]))

            # 计算所需要的行列数
            num_rows = (len(images) // x) + (len(images) % x != 0)
            num_cols = x if len(images) < x else len(images)

            # 创建子图
            fig, axes = plt.subplots(num_rows, num_cols)

            # 绘制图片
            for j, image in enumerate(images):
                if len(images) > x:
                    axes[j // x, j % x].imshow(image)
                    axes[j // x, j % x].set_title(f"Image {j + 1}")
                else:
                    axes[j].imshow(image)
                    axes[j].set_title(f"Image {j + 1}")

            # 隐藏多余的子图
            for j in range(len(images), num_rows * num_cols):
                if len(images) > x:
                    axes[j // x, j % x].set_visible(False)
                else:
                    axes[j].set_visible(False)

            # 自适应调整子图间距
            fig.tight_layout(pad=0)
            # 显示图像
            plt.show()

    def load_per_img(self, k):
        # 导入原始图像
        image = cv2.imread(self.image_dir + f'{k}.png')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.images_list[k] = image

    def load_per_json(self, k):
        # 读取标签的json文件，并保存points至points_lists
        with open(self.prompt_dir + f'{k}.json') as f:
            label_data = simplejson.load(f)
        points_list = []
        label_list = []
        for shape in label_data['shapes']:
            points = shape['points']
            label = shape['label']
            points_list.append(points[0])
            if label == self.class_name:
                label_list.append(1.0)
            else:
                label_list.append(0.0)

        self.points_list[k] = np.asarray([points_list]).reshape(-1, 2)
        self.labels_list[k] = np.asarray([label_list])[0, :]

    def load_per_gt(self, k):
        # 读取GT mask
        gt_grayscale_org = cv2.imread(self.gt_dir + f'{k}.png', cv2.IMREAD_GRAYSCALE)
        gt = (gt_grayscale_org == 255)
        self.gts_list[k] = gt

    def show_mask(self, mask, ax, random_color=False):
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)

    def show_box(self, box, ax):
        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]
        ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

    def show_points(self, coords, labels, ax, marker_size=375):
        pos_points = coords[labels == 1]
        neg_points = coords[labels == 0]
        ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='.', s=marker_size, edgecolor='white',
                   linewidth=1.25)
        ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='.', s=marker_size, edgecolor='white',
                   linewidth=1.25)


