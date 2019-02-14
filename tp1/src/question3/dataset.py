# -*- coding: utf-8 -*-p
import os
import numpy as np
from skimage import io
from torch.utils.data import Dataset
import PIL

class CatDataset(Dataset):
    """ Cat dataset"""
    
    def __init__(self, img_dir, transform=None, augment=None):
        """
        Args:
            img_dir (string) : Directory with the cat images.
        """
        self.img_dir = img_dir
        self.transform = transform
        self.augment = augment
        
    def __len__(self):
        return len([name for name in os.listdir(self.img_dir)])
         
        
    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, str(idx+1) + ".Cat.jpg")
        image = io.imread(img_name)
        
        # Find a random image that is not empty
        while (image.shape != (64,64,3)):
            img_name = os.path.join(self.img_dir, str(np.random.randint(0, len(self))) + ".Cat.jpg")
            image = io.imread(img_name)
        
        if (self.augment):
            image = self.augment(image)

        if (self.transform):
            image = self.transform(image)

        sample = (image, 0)
        return sample
        
        
class DogDataset(Dataset):
    """ Dog dataset"""
    
    def __init__(self, img_dir, transform=None, augment=None):
        """
        Args:
            img_dir (string) : Directory with the dog images 
        """
        self.img_dir = img_dir
        self.transform = transform
        self.augment = augment
        
    def __len__(self):
        return len([name for name in os.listdir(self.img_dir)])
        
    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, str(idx+1) + ".Dog.jpg")
        image = io.imread(img_name)
        
        # Find a random image that is not empty
        while (image.shape != (64,64,3)):
            img_name = os.path.join(self.img_dir, str(np.random.randint(0, len(self))) + ".Dog.jpg")
            image = io.imread(img_name)
        
        if (self.augment):
            image = self.augment(image)

        if (self.transform):
            image = self.transform(image)

        sample = (image, 1)
    
        return sample

    
class CatDogDataset(Dataset):
    """ Cat and dog combined dataset. Cat images followed by dog images."""
    def __init__(self, img_dir, transform=None, augment=None):
        """
        Args:
            img_dir (string) : Directory with the dog and cat folders 
                containing their images.
        """
        self.img_dir = img_dir
        self.transform = transform
        # Data augmentation transforms
        self.augment = augment
        
        # Create individual cat and dog datasets
        cat_dir = os.path.join(self.img_dir, "Cat/")
        dog_dir = os.path.join(self.img_dir, "Dog/")
        self.catDataset = CatDataset(cat_dir, self.transform, self.augment)
        self.dogDataset = DogDataset(dog_dir, self.transform, self.augment)
        
    def __len__(self):
        return len(self.catDataset + self.dogDataset)
        
    def __getitem__(self, idx):
        # if index is greater than 9998 than get dog, else get cat
        idx = idx.item()
        dataset_len = self.__len__()
        if (idx < dataset_len):
            if (idx >= len(self.catDataset)):
                sample = self.dogDataset[idx - len(self.catDataset)]
            else:
                sample = self.catDataset[idx]
        else:
            sample = None
            
        return sample
    
    def getCatIdx(self):
        """ Returns the first index of the cat images """
        return 0
    
    def getDogIdx(self):
        """ Returns the first index of the dog images """
        return len(self.catDataset)
    
class TestDataset(Dataset):
    """ Cat and dog test set."""

    def __init__(self, img_dir, transform=None):
        """
        Args:
            img_dir (string) : Directory with the dog images 
        """
        self.img_dir = img_dir
        self.transform = transform
        
    def __len__(self):
        return len([name for name in os.listdir(self.img_dir)])
        
    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, str(idx+1) + ".jpg")
        image = io.imread(img_name)
        
        # Find a random image that is not empty
        while (image.shape != (64,64,3)):
            img_name = os.path.join(self.img_dir, str(np.random.randint(0, len(self))) + ".jpg")
            image = io.imread(img_name)
        
        if (self.transform is not None):
            image = self.transform(image)
    
        return image
    
  
