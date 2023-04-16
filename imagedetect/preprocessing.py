import numpy as np
from PIL import Image, ImageFilter, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import pandas as pd
import os
import torch as t
from imageio import imread
from torch.utils.data import TensorDataset
from os import path

img_size = 32
# Image augmentation class
# Input: hflip, rotate, blurring
# Output: augmented image
class Augmenter:
    def __init__(self, hflip=True, rotate=True, blurring=False):
        self.hflip = hflip
        self.rotate = rotate
        self.blurring = blurring

    def extract_mutiview(self, voxel):
        x, y, z = voxel.shape
        return [voxel[x // 2, :, :],
                voxel[:, y // 2, :],
                voxel[:, :, z // 2]]

    def augment(self, voxel):
        for view in self.extract_mutiview(voxel):
            im = Image.fromarray(view, mode='L')
            yield im
            if self.hflip:
                yield im.transpose(Image.FLIP_LEFT_RIGHT)
            if self.rotate:
                yield im.transpose(Image.ROTATE_90)
                yield im.transpose(Image.ROTATE_180)
                yield im.transpose(Image.ROTATE_270)
            if self.blurring:
                yield im.filter(ImageFilter.GaussianBlur(1))


def resize(im):
    im = im.crop((8, 8, 40, 40))
    return im

# Generate dataset function, generates images from the processed npy files
# Input: dir
# Output: images, csv mapping images to labels
def generate_dataset(dir):
    # Load in csv created in LIDC.py
    df = pd.read_csv(dir+'/labels.csv')

    # Get voxel data
    voxels = np.zeros((len(df),48,48,48), dtype=np.uint8)
    # Augmenter for training data
    augmenter = Augmenter(hflip=True, rotate=True, blurring=True)
    augmenter2 = Augmenter(hflip=False, rotate=False, blurring=False)

    for i, row in df.iterrows():
        voxels[int(row.id)] = np.load('{0}/{1:.0f}.npy'.format(dir,row.id))

    for i in range(1):
        folder = '{0}/'.format(dir)
        if not os.path.exists(folder):
            os.makedirs(folder)
        tests = df[df.testing == 1].copy()
        trains = df[df.testing == 0].copy()
        # trains.testing = 0
        new_df = pd.concat([tests, trains])
        new_df.to_csv(folder+'/labels.csv', index=False)

        for j, row in tests.iterrows():
            voxel = voxels[int(row.id)]
            for e,im in enumerate(augmenter2.augment(voxel)):
                im2 = resize(im)
                im2.save('{0}{1:.0f}.{2}.png'.format(folder, row.id, e))

        for j, row in trains.iterrows():
            voxel = voxels[int(row.id)]
            for e,im in enumerate(augmenter.augment(voxel)):
                im2 = resize(im)
                im2.save('{0}{1:.0f}.{2}.png'.format(folder, row.id, e))
        
# Map malignancy function
# Input: malignancy
# Output: 0, 1, 2
def map_malignancy_th(malignancy):
    if malignancy >= 3.5:
        return  2
    elif malignancy <= 2:
        return  0
    else:
        return  1
# Get datset function to load in data
# Input: directory of data
# Output: trainset, testsset    
def get_dataset(dir):
    df = pd.read_csv(path.join(dir, 'labels.csv'))
    # Create label that is 0 if maligancy < 3, 1 if maligancy = 3, 2 if maligancy > 3
    df_test = df[df.testing==1]
    df_train = df[df.testing == 0]

    # Augment data
    num_data = len(df_train)
    aug_size = 3
    x = t.zeros((num_data * aug_size, 1, img_size, img_size))
    y = t.zeros((num_data * aug_size, 1))
    c = 0
    for i, row in df_train.iterrows():
        id = int(row.id)
        for j in range(aug_size):
            print(path.join(dir,f'{id:.0f}.{j}.png'))
            im = imread(path.join(dir,f'{id:.0f}.{j}.png'))
            x[c * aug_size + j, 0, :, :] = t.from_numpy(im)
            y[c * aug_size + j][0] = row.malignancy_th
        c += 1

    mu = x.mean()
    sd = x.std()
    x = (x - mu) / sd

    trainset = TensorDataset(x, y)
    aug_size = 3
    num_data = len(df_test)
    x = t.zeros((num_data*aug_size, 1, img_size, img_size))
    y = t.zeros((num_data*aug_size, 1))
    c = 0
    for i, row in df_test.iterrows():
        id = int(row.id)
        for j in range(aug_size):
            im = imread(path.join(dir, f'{id:.0f}.{j}.png'))
            x[c * aug_size + j, 0, :, :] = t.from_numpy(im)
            y[c * aug_size + j][0] = row.malignancy_th
        c += 1

    x = (x - mu) / sd
    testset = TensorDataset(x, y)

    return trainset, testset
# Get datset function to load in data for ResNet
# Input: directory of data
# Output: trainset, testsset
def get_dataset3d(dir):
    df = pd.read_csv(path.join(dir, 'labels.csv'))
    df_test = df[df.testing==1]
    df_train = df[df.testing == 0]
    df_train["malignancy_response"] = df_train["malignancy"].apply(map_malignancy_th)
    df_test["malignancy_response"] = df_test["malignancy"].apply(map_malignancy_th)
    print(df_train["malignancy_response"].value_counts())
    num_data = len(df_train)
    aug_size = 18
    x = t.zeros((num_data * aug_size, 3, img_size, img_size))
    y = t.zeros((num_data * aug_size, 1))
    c = 0
    # Loop through each image
    for i, row in df_train.iterrows():
        id = int(row.id)
        for j in range(aug_size):
            im = imread(path.join(dir, f'{id:.0f}.{j}.png'))
            x[c * aug_size + j, 0, :, :] = t.from_numpy(im)
            y[c * aug_size + j][0] = row.malignancy_response
            x[c * aug_size + j, 1, :, :] = x[c * aug_size + j, 0, :, :]
            x[c * aug_size + j, 2, :, :] = x[c * aug_size + j, 0, :, :]
        c += 1

    mu = x.mean()
    sd = x.std()
    x = (x - mu) / sd
    trainset = TensorDataset(x, y)
    aug_size = 3
    num_data = len(df_test)
    x = t.zeros((num_data*aug_size, 3, img_size, img_size))
    y = t.zeros((num_data*aug_size, 1))
    c = 0
    for i, row in df_test.iterrows():
        id = int(row.id)
        for j in range(aug_size):
            im = imread(path.join(dir, f'{id:.0f}.{j}.png'))
            print(path.join(dir,f'{id:.0f}.{j}.png'))
            x[c * aug_size + j, 0, :, :] = t.from_numpy(im)
            y[c * aug_size + j][0] = row.malignancy_response
            x[c * aug_size + j, 1, :, :] = x[c * aug_size + j, 0, :, :]
            x[c * aug_size + j, 2, :, :] = x[c * aug_size + j, 0, :, :]
        c += 1

    x = (x - mu) / sd
    testset = TensorDataset(x, y)

    return trainset, testset


if __name__ == '__main__':
    import sys
 
    generate_dataset("/Users/alex/dev/STAT 447B/Project/Data/Meta/vision_preprocess_output")
 