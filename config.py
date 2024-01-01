import os
from torchvision import transforms

use_gpu = True
gpu_name = 0

#pre_model = os.path.join('pth', '8_stage1_resnet50_SE.pth')
pre_model = os.path.join('pth', 'RSR_model_stage1_epoch200.pth')
save_path = "pth"

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(32),
    transforms.RandomChoice([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2)]),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
])

stage1_dataset_path = r'T10/'
stage1_max_epoch = 1000
stage1_batch_size = 1536
stage1_lr = 1e-5

stage2_max_epoch = 300
stage2_batch_size = 100 # 100
stage2_lr = 1e-3

eval_batch_size = 100 # 512

num_workers = 80
