import torch
import cv2
import os
import glob
# import sys
# sys.path.append(os.path.dirname(os.getcwd()))

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import v2

MADISON_DATA = '/home/syurtseven/gsoc-2024/data/madison'
PLOT_DIRECTORY = '/home/syurtseven/gsoc-2024/reports'


class MadisonDataset(Dataset):

    def __init__(self, image_paths):
        
        self.image_paths = image_paths
        self.filtered_paths = []

        img_dict = {}
        for img in self.image_paths:

            re = cv2.imread(img, cv2.IMREAD_UNCHANGED)
            if len(re.shape) == 2 :
                self.filtered_paths.append(img)

    def __len__(self):
        return len(self.filtered_paths)

    def __getitem__(self, index):

        img = read_image(self.filtered_paths[index])
        img = resize_torch_tensor(img)

        return img
        

def read_image(filepath):
    return cv2.imread(filepath, cv2.IMREAD_UNCHANGED)

def min_max_normalize(tensor, min_val=0.0, max_val=1.0):
    tensor_min = tensor.min()
    tensor_max = tensor.max()
    normalized_tensor = (tensor - tensor_min) / (tensor_max - tensor_min) * (max_val - min_val) + min_val
    return normalized_tensor

def resize_torch_tensor(tensor, w=256, h=256):

    transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((w, h)),
    ])
    tensor = transform(tensor)
    tensor = tensor.float()

    tensor = min_max_normalize(tensor)
    _ = ''

    return tensor, _



if __name__ == '__main__':

    image_paths = glob.glob(os.path.join(MADISON_DATA, '*'))
    print(f"#IMAGES:{len(image_paths)}")

    dataset    = MadisonDataset(image_paths=image_paths)
    dataloader = DataLoader(dataset=dataset, batch_size=30)
    print(f"#BATCHES:{len(dataloader)}")
    data = next(iter(dataloader))
    print(data.shape)