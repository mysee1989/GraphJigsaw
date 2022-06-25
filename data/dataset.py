import os
from PIL import Image
import torch
import torchvision
from torch.utils import data
import numpy as np
from torchvision import transforms as T
import torchvision
import cv2
import sys
import pdb


class Dataset(data.Dataset):

    def __init__(self, root, data_list_file, phase='train', input_shape=(3, 224, 224)):
        self.phase = phase
        self.input_shape = input_shape

        with open(os.path.join(data_list_file), 'r') as fd:
            imgs = fd.readlines()

        # imgs =imgs[:400]
        imgs = [os.path.join(root, img[:-1]) for img in imgs]
        # self.imgs = np.random.permutation(imgs)
        self.imgs = imgs

        
        normalize = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        # normalize = T.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
        
        if self.phase == 'train':
            self.transforms = T.Compose([
                T.Resize((256,256)),
                T.RandomCrop(self.input_shape[1:]),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                normalize
            ])
        else:
            self.transforms = T.Compose([
                T.Resize((256,256)),
                T.CenterCrop(self.input_shape[1:]),
                T.ToTensor(),
                normalize
            ])

    def __getitem__(self, index):
        sample = self.imgs[index]
        splits = sample.split()
        img_path = splits[0]
        data = Image.open(img_path).convert('RGB')
        # data = data.convert('L')
        data = self.transforms(data)
        label = np.int32(splits[1])
        if self.phase == 'train':
            return data.float(), label
        else:
            return data.float(),label,img_path

    def __len__(self):
        return len(self.imgs)


if __name__ == '__main__':
    dataset = Dataset(root='/data/Datasets/fv/dataset_v1.1/dataset_mix_aligned_v1.1',
                      data_list_file='/data/Datasets/fv/dataset_v1.1/mix_20w.txt',
                      phase='test',
                      input_shape=(1, 128, 128))

    trainloader = data.DataLoader(dataset, batch_size=10)
    for i, (data, label) in enumerate(trainloader):
        # imgs, labels = data
        # print imgs.numpy().shape
        # print data.cpu().numpy()
        # if i == 0:
        img = torchvision.utils.make_grid(data).numpy()
        # print img.shape
        # print label.shape
        # chw -> hwc
        img = np.transpose(img, (1, 2, 0))
        # img *= np.array([0.229, 0.224, 0.225])
        # img += np.array([0.485, 0.456, 0.406])
        img += np.array([1, 1, 1])
        img *= 127.5
        img = img.astype(np.uint8)
        img = img[:, :, [2, 1, 0]]

        cv2.imshow('img', img)
        cv2.waitKey()
        # break
        # dst.decode_segmap(labels.numpy()[0], plot=True)
