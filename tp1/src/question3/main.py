# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np

import dataset as ds

if __name__ == '__main__':
    img_directory = "trainset/"
    train_dataset = ds.CatDogDataset(img_directory)
    
    # Show first 4 cat images
    fig = plt.figure()
    for i in range(4):
        img, label = train_dataset[i]
        print(i, img.shape, label)
        print("mean : {}, std: {}".format(np.mean(img), np.std(img)))
        
        ax = plt.subplot(1, 4, i+1)
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
        
        ax = plt.subplot(1, 4, i+1)
        plt.tight_layout()
        plt.imshow(img)
        ax.set_title('Dog Sample #{}'.format(i))
        ax.axis('off')
    plt.show()
    
    # Show 
    
    
    