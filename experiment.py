from mytrain import Mytrain, test
from Dataset import own_prepare

# 设置随机种子
seed = 40
from Frame import set_seed
set_seed(seed)

data_path='data/point_label_try/'
label_path='data/label_try/'
device = 'cuda:0'
class_name='road'

[is_train, is_test, is_check] = [1,0,0]

# optimizer = torch.optim.Adam(, lr=learning_rate)
# scheduler = torch.ReduceLROnPlateau(self.optimizer, mode='min', patience=patience, verbose=True)
# Mytrain(data_path='data/point_label_try/', label_path='data/label_try/', lr=1e-5, data_num=2, num_epochs=50,
#         visdual=False, load_button=False)

if is_train:
        # 导入训练集和验证集
        # 归一化, 划分为训练集和验证集
        datas = own_prepare(data_path, label_path ,device=device, class_name=class_name, data_num=2)
        [train_ids, val_ids, transformed_data, prompts, ground_truth_masks] = datas

        save_dir = Mytrain(datas, lr=1e-5, wd=0, num_epochs=50, device=device, loss_name='Focal', seed=seed,
                visdual=False, load_button=False)

if is_test:
        save_dir = "D:\pycharmprogram\segment-anything\log\epochs50_lr1e-05_wd0_Focal_data2_vit_b_seed40_time20230922_2043/"
        test(save_dir=save_dir,test_dir='data/test/', class_name='road',device=device)

if is_check:
        from check_data import check_Data
        check=check_Data(image_dir='data/image_prompts/', prompt_dir='data/image_prompts/', gt_dir='data/gt/')
        check.check_data(class_name)