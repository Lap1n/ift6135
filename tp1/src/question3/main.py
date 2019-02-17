# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as transforms
import PIL
import torch

import dataset as ds

if __name__ == '__main__':
    img_directory = "trainset/"
    dataset = ds.CatDogDataset(img_directory)
    
    # Show transformations    
    img,label = dataset[11000]
    img = transforms.ToPILImage()(img)
    
    plt.figure()
    plt.subplot(231)
    plt.imshow(img)
    plt.title("Original Image")
    plt.subplot(232)
    plt.imshow(transforms.RandomHorizontalFlip()(img))
    plt.title("Random Horizon Flip")
    plt.subplot(233)
    plt.imshow(transforms.RandomRotation(30, resample=PIL.Image.BILINEAR)(img))
    plt.title("Random rotation (+/- 30 degrees)")
    plt.subplot(234)
    plt.imshow(transforms.ColorJitter(hue=0.1, saturation=1)(img))
    plt.title("Random Color Jitter")
    plt.subplot(235)
    plt.imshow(transforms.RandomGrayscale(3)(img))
    plt.title("Grayscale")
    plt.subplot(236)
    plt.imshow(transforms.RandomAffine(0, translate=(0.2, 0.2), scale=(0.7, 1.4))(img))
    plt.title("Random Affine")

#%%
    # transform applied to Dataset when data augmentation is set to True
    rand_tsfm = transforms.RandomApply([transforms.RandomHorizontalFlip(),
                                        transforms.RandomRotation(20, resample=PIL.Image.BILINEAR),
                                        transforms.ColorJitter(hue=0.1, saturation=0.1),
                                        transforms.RandomGrayscale(3),
                                        transforms.RandomAffine(30, (0.1, 0.1), (0.7,1.3))])
    
    # Show augmented dataset
    transform = transforms.Compose([transforms.ToPILImage(), rand_tsfm])
    
    plt.figure()
    for i in range(16):
        img, label = dataset[i]
        img_t = transform(img)
        # original dataset sample
        ax = plt.subplot(4, 4, i+1)

    # Show first 4 cat images
    fig = plt.figure()
    for i in range(4):
        img, label = train_dataset[i]
        print(i, img.shape, label)
        print("mean : {}, std: {}".format(np.mean(img), np.std(img)))

        ax = plt.subplot(1, 4, i + 1)
        plt.tight_layout()
        plt.imshow(img)
        ax.set_title('Cat Sample #{}'.format(i))
        ax.axis('off')
    plt.show()

    # Show first 4 dog images
    fig = plt.figure()
    for i in range(4):
        img, label = train_dataset[i + train_dataset.getDogIdx()]
        print(i, img.shape, label)

        ax = plt.subplot(1, 4, i + 1)
        plt.tight_layout()
        plt.imshow(img_t)
        ax.axis('off')
    plt.show()

