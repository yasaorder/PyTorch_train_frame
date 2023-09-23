import torch
import random
import os
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from statistics import mean

def set_seed(seed):
    random.seed(seed)# python设置随机种子
    np.random.seed(seed) # numpy
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed) # cpu
    torch.cuda.manual_seed(seed) # 当前gpu
    torch.cuda.manual_seed_all(seed) # 所有gpu
    torch.backends.cudnn.deterministic = True # 保证卷积算法一致
    torch.backends.cudnn.benchmark = False

    # 如果报错：CUDA版本为10.2或更高，则少数CUDA操作不确定，
    # 改正：设置了环境变量 CUBLAS_WORKSPACE_CONFIG=:4096:8 或 CUBLAS_WORKSPACE_CONFIG=:16:8
    # 以下只需一句
    # os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
    # os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    # torch.use_deterministic_algorithms(True) # 用于排查不可复现的问题
    print(f"随机种子：{seed}")

import time
def load_show(i, num, start_time, legend='', load_button=True):
    if load_button==False:
        pass
    else:
        i += 1
        scale = 20
        i = int(i / num * scale)

        loads = "*" * i
        noloads = "." * (scale - i)
        percent = (i / scale) * 100
        dur = time.perf_counter() - start_time
        print("\r" + legend + "    {:^3.0f}%[{}->{}]{:.2f}s".format(percent, loads, noloads, dur), end="")

        if i == scale:
            print("")

# --------------------- 保存和加载的方法，可复用 ------------------
def save(model, path):
    torch.save(model.state_dict(), path)

def load(model, path):
    with open(path, "rb") as f:
        state_dict = torch.load(f)
    model.load_state_dict(state_dict)

class myframe:
    def __init__(self, ):
        self.cur_epoch = 0
        self.best_loss = 1000
        self.no_optim = 0
        self.best_epoch = 0
        self.is_stop = False

        self.train_losses = []
        self.val_losses = []

    def loads(self ,epoch_nums ,lr ,model ,optimizer ,save_dir, scheduler=None ,log_name='mylog.log', weight_name='weight.pth'):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.epoch_nums = epoch_nums
        self.cur_lr = lr

        self.save_dir = save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.log_path = save_dir + log_name
        self.log_file =open(self.log_path, 'w')
        self.weight_name = weight_name

    def load_stop_pars(self ,no_optim_patience=5 ,stop_lr=1e-20 ,re_new_lr=5, is_times=True):
        """
        早停相关参数
        Arguments：
        stop_lr=1e-8： loss连续上升多少次停止
        re_new_lr=5：早停后每次更新学习率的倍数
        is_times=True: 为真则每次更新学习率乘以re_new_lr， 为假则每次更新学习率为re_new_lr
        """
        self.no_optim_patience = no_optim_patience
        self.stop_lr = stop_lr
        self.re_new_lr = re_new_lr
        self.is_times = is_times

    def epoch_step(self, epoch_train_losses, epoch_val_losses):
        """
        每次epoch 更新相关参数，保存训练集的损失、验证集的损失
        Arugments:
        epoch_train_losses: [list], 每个epoch每张训练集图片的训练损失
        epoch_val_losses: [list], 每个epoch每张验证集图片的训练损失

        !!!!
        判断早停用的是验证集损失， 最优损失和权重也是对应的验证集
        如果验证集损失上升的同时，经过了int(no_optim_patience * 0.6)次训练，则会降低学习率，载入目前最优权重，以尝试挽救训练
        如果学习率低于stop_lr，则视为已经收敛，不再训练
        """
        self.cur_loss = mean(epoch_val_losses)

        self.train_losses.append(epoch_train_losses)
        self.val_losses.append(epoch_val_losses)
        self.cur_lr = self.optimizer.param_groups[0]['lr']

        print("--------- epoch %d ---------" % self.cur_epoch)
        print(f'train Mean loss: {mean(epoch_train_losses)}, val Mean loss: {mean(epoch_val_losses)}')
        print(f'lr: {self.cur_lr}')
        self.log_file.write("--------- epoch %d ---------\n" % self.cur_epoch)
        self.log_file.write(f'train Mean loss: {mean(epoch_train_losses)}, val Mean loss: {mean(epoch_val_losses)}\n')
        self.log_file.write(f'lr: {self.cur_lr}\n')

        # 设置学习策略则更新学习率
        if self.scheduler != None:
            self.scheduler.step()

        # 更新最优loss， 保存最优权重
        if self.cur_loss < self.best_loss:
            self.best_loss = self.cur_loss
            self.best_epoch = self.cur_epoch
            self.no_optim = 0
            self.save_path = self.save_dir + self.weight_name
            save(self.model, self.save_path)
        else:
            self.no_optim += 1

        # 判断训练是否早停
        if self.no_optim > self.no_optim_patience:
            print('early stop at %d epoch' % self.cur_epoch)
            self.log_file.write('early stop at %d epoch\n' % self.cur_epoch)
            self.is_stop = True

        # 如果验证集损失上升的同时，经过了int(no_optim_patience * 0.6)次训练，则会降低学习率，载入目前最优权重，以尝试挽救训练
        # 如果学习率低于stop_lr，则视为已经收敛，不再训练
        if self.no_optim > int(self.no_optim_patience * 0.6):
            if self.cur_lr < self.stop_lr:
                print("cur_lr < {:.4e}\n".format(self.stop_lr))
                print('early stop at %d epoch' % self.cur_epoch)
                self.log_file.write("cur_lr < {:.4e}\n".format(self.stop_lr))
                self.log_file.write('early stop at %d epoch\n' % self.cur_epoch)
                self.is_stop = True

            load(self.model, self.save_path)
            self.update_lr()
            self.no_optim = 0

        self.cur_epoch += 1
        self.log_file.flush()

    def finsh(self):
        self.mean_train_losses = [mean(x) for x in self.train_losses]
        self.mean_val_losses = [mean(x) for x in self.val_losses]
        self.log_file.write(' --epoch_best_loss:' + str(float(self.best_loss)) +
                ' --epoch_best:' + str(self.best_epoch) + '\n' +
                ' --train_losses:' + str(self.train_losses) + '\n' +
                ' --val_losses:' + str(self.val_losses))
        print(self.log_file, 'Finish!')
        print('Finish!')
        self.log_file.close()

    def get_lossfig(self ,figsize=(8, 6)):
        # Set up plot
        plt.figure(figsize=figsize)
        plt.title('Training and Validation Loss', fontsize=18)
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel('Loss', fontsize=14)
        # Plot train loss
        plt.plot(list(range(len(self.mean_train_losses))), self.mean_train_losses, 'b-', label='Train')
        # Plot validation loss
        plt.plot(list(range(len(self.mean_val_losses))), self.mean_val_losses, 'r--', label='Val')
        # Add legend
        plt.legend(fontsize=14)
        # Show plot
        plt.savefig(self.save_dir + '/loss.png')

        return plt

    def update_lr(self,):
        old_lr = self.optimizer.param_groups[0]['lr']
        if self.is_times: # 以倍数形式更新lr
            new_lr = old_lr / self.re_new_lr
        else:
            new_lr = self.re_new_lr

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr

        print(f'update learning rate: {old_lr} -> {new_lr}')
        self.log_file.write(f'update learning rate: {old_lr} -> {new_lr} \n')
        self.cur_lr = new_lr