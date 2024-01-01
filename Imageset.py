import os
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset


catalog_path = r'Dataset_stage2/'
train_path = os.path.join(catalog_path, 'train/')
test_path = os.path.join(catalog_path, 'test/')
# type_names = os.listdir(train_path)
type_names =[1,2,3,4,5,6,7,8]


# 计算平均大小
def get_avg_size():
    count = 0
    total_width = 0
    total_height = 0

    for data_purpose in os.listdir(catalog_path):
        # print(data_purpose)
        purpose_path = os.path.join(catalog_path, data_purpose)
        for folder in os.listdir(purpose_path):
            folder_path = os.path.join(purpose_path, folder)
            for picture in os.listdir(folder_path):
                if picture.endswith('.jpg') or picture.endswith('.png'):
                    img = Image.open(os.path.join(folder_path, picture))
                    # 彩色图片，shape形式为(height, width, channels)
                    width, height = np.array(img).shape[1::-1]
                    total_height += height
                    total_width += width
                    count += 1

    # 计算平均分辨率
    avg_width_count = round(total_width / count)
    avg_height_count = round(total_height / count)
    return avg_width_count, avg_height_count


avg_width, avg_height = get_avg_size()


class ImageDataset(Dataset):

    def __init__(self, root_dir, label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(root_dir, label_dir)
        self.img_path = os.listdir(self.path)
        self.avg_width = avg_width
        self.avg_height = avg_height

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, item):
        # print('path',self.img_path)
        img_name = self.img_path[item]
        item_path = os.path.join(self.path, img_name)
        img = Image.open(item_path)
        channel = np.array(img).shape[2]
        if channel == 4:
            # 转换为3通道
            img = img.convert('RGB')
        img = img.resize((self.avg_width, self.avg_height),resample=Image.Resampling.BICUBIC)
        label = int(self.label_dir)

        tensor_trans = transforms.ToTensor()
        tensor_img = tensor_trans(img)

        return tensor_img, label


def get_train_data():
    dataset_list = []
    for img_type in type_names:
        img_dataset = ImageDataset(train_path, str(img_type))
        # print(train_path)
        dataset_list.append(img_dataset)
    train_data = dataset_list[0]
    for i in range(1, len(dataset_list)):
        train_data += dataset_list[i]
        # print('dataset_list',dataset_list[i])
    return train_data


def get_test_data():
    dataset_list = []
    for img_type in type_names:
        img_dataset = ImageDataset(test_path, str(img_type))
        dataset_list.append(img_dataset)
    test_data = dataset_list[0]
    for i in range(1, len(dataset_list)):
        test_data += dataset_list[i]

    return test_data
