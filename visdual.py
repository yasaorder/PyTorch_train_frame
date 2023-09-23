from torchvision.utils import make_grid
from tensorboardX import SummaryWriter
# from Frame import myframe
import torch

class visdualer:
    def __init__(self ,is_visdual=True ,logdir='log/visdual' ,device='cuda:0' ,muti=1):
        self.is_visdual = is_visdual
        if self.is_visdual == False:
            pass
        else:
            self.writer = SummaryWriter(logdir=logdir)
            self.device = device

    def get_loss(self ,train_loss ,val_loss ,cur_epoch):
        """
        可视化输入训练loss
        Arguments:
        train_loss: float ,每个epoch的loss
        val_loss: float
        cur_epoch: int
        """
        if not self.is_visdual:
            pass
        else:
            self.writer.add_scalars('loss',
                                    {"train": train_loss, 'val_loss': val_loss},
                                    global_step=cur_epoch)

    def get_lr_curve(self ,optimizer ,cur_epoch):
        """
        可视化输入训练loss
        Arguments:
        train_loss: float ,每个epoch的loss
        val_loss: float
        cur_epoch: int
        """
        if not self.is_visdual:
            pass
        else:
            self.writer.add_scalar('lr', optimizer.param_groups[0]['lr'], global_step=cur_epoch)

    def get_img_to_pred_before(self ,image_size ,pred_size ,multi=1):
        """
        可视化输入图片和标签， 在每个epoch内，载入数据集前使用
        Arguments:
        image_size: [C,H,W]
        pred_size: [H,W], 01单通道
        multi: 一次可视化图片的数量，默认1

        return vis_image:tensor, [1,C,H,W]
                vis_label:tensor, [1,1,H,W]
        """
        self.multi = multi
        if not self.is_visdual:
            pass
        else:
            if self.multi == 1:
                pass
            else:
                self.count = 0
                self.vis_image = torch.ones(image_size, device=self.device)[None,:,:,:]
                self.vis_label = torch.ones(pred_size, device=self.device)[None,None,:,:]

    def get_img_to_pred_load(self ,input_image ,input_pred ,cur_epoch ,tag):
        """
        可视化输入图片和标签， 在训练每张输入图片时使用
        Arguments:
        input_image:tensor, [C,H,W]
        input_pred:tensor, [H,W]
        """
        if not self.is_visdual:
            pass
        else:
            if self.multi==1:
                self.add_ims = input_image
                self.add_las = input_pred
            else:
                self.count += 1
                self.vis_image = torch.cat((self.vis_image, input_image[None, :, :, :]), 0)
                self.vis_label = torch.cat((self.vis_label, input_pred * 255), 0)

                if (self.count == self.multi):
                    self.add_ims = make_grid(self.vis_image[1:], normalize=True, scale_each=True)
                    self.add_las = make_grid(self.vis_label[1:], normalize=True, scale_each=True)

            self.writer.add_image(f"{tag}_input_image", self.add_ims, cur_epoch)
            self.writer.add_image(f"{tag}_label_pred", self.add_las, cur_epoch)

    def close(self):
        if not self.is_visdual:
            pass
        else:
            self.writer.close()

# # loss曲线
# tb_logger.add_scalar('loss_train', loss, curr_step)
# # 输入图片和标签的可视化
# tb_logger.add_image('image', input[0], curr_step)
# # 单通道特征图可视化
#
#
# tb_logger.add_image('feature_map', make_grid([feature_map1, feature_map2, fetare_map3], padding=20, normalize=True,
#                                              scale_each=True, pad_value=1), curr_step)
# # 多通道特征图的可视化
# tb_logger.add_image('channels', make_grid(feature_map[0].detach().cpu().unsqueeze(dim=1), nrow=5, padding=20, normalize=False, pad_value=1), curr_step)
