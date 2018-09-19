from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
from torch.utils.data import Dataset, DataLoader, TensorDataset

plt.ion()   # interactive mode

class AntsAndBeesTask():
    def __init__(self, args):
        # self.classes = list(range(10))
        self.args = args

        # trans = transforms.Compose([transforms.ToTensor()])
        # self.train_set = dset.MNIST(root='./data/', train=True, transform=trans, download=args.download)
        # self.test_set = dset.MNIST(root='./data/', train=False, transform=trans)
        
    def salt_and_pepper_noise(self, x):
        probs = torch.rand(*x.size())
        x[probs < self.args.noise_prob / 2] = 0
        x[probs > 1 - self.args.noise_prob / 2] = 1
        return x
    def convert_data(self, data):
        for i in range(len(data[0][0])):
            # print(i, "this is i")
            # print("")
            # print("space")
            # print("")
            # print(data[index], "data by index")
            # print("")
            # print("space")
            # print("")
            # print(data[index][0], "first element in data by first index")
            # print("")
            # print("space")
            # print("")
            # print(data[1][0], "first element in data by second index")
            # print("")
            # print("space")
            # print("")
            # print(data[0][0][0], "very first element of data by firstindex")
            # print("")
            # print("space")
            # print("")
            # print(data[0][0][3], "third element of data by firstindex")
            # print("")
            # print("space")
            # print("")
            if data[0][0][i] >= 0.5:
                data[0][0][i] = 1.0000
            else:
                data[0][0][i] = 0.0000
            if data[1][0][i] >= 0.5:
                data[1][0][i] = 1.0000
            else:
                data[1][0][i] = 0.0000
        return data 
    def convert_largedata(self, data):
        # print(data[0][0], "first element in data in first column and row")
        for i in range(16):
            for j in range(50176):
                # print("")
                # print("space")
                # print("")
                # print(data[i][j], "element at i and j of data")
                # print("")
                # print("space")
                # print("")
                if data[i][j] >= 0.5:
                    data[i][j] = 1.0000
                    # print(data[i][j], "changed val of data to 1")
                else:
                    data[i][j] = 0.0000
                    # print(data[i][j], "changed val of data to 0")
        return data
    def convert_otherlargedata(self, data):
        for i in range(12):
            for j in range(50176):
                if data[i][j] >= 0.5:
                    data[i][j] = 1.0000
                else:
                    data[i][j] = 0.0000  
        return data 

    def convert_otherotherlargedata(self, data):
        for i in range(11):
            for j in range(50176):
                if data[i][j] >= 0.5:
                    data[i][j] = 1.0000
                else:
                    data[i][j] = 0.0000  
        return data  
    def init(self):
        data_transforms = {
    'train': transforms.Compose([
        transforms.RandomSizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}
        data_dir = 'hymenoptera_data'
        image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                                  data_transforms[x])
                          for x in ['train', 'val']}
        dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                                     shuffle=True, num_workers=4)
                      for x in ['train', 'val']}
        dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
        class_names = image_datasets['train'].classes
        print(class_names, "these are class names")
        print('Binary classification between {} and {}'.format(class_names[0], class_names[1]))

        use_gpu = torch.cuda.is_available()
        inputs, classes = next(iter(dataloaders['train']))

        # Make a grid from batch
        out = torchvision.utils.make_grid(inputs)
        # print(image_datasets['train'], "image dataset train data")
        # print(image_datasets['val'], "image dataset val data")
        # print(dataloaders['train'], "dataloaders train data")
        # print(dataloaders['val'], "dataloaders val data")
        labels = [0, 1]

        # print(image_datasets['train'][0], "first part of image dataset train data") #3x224x224
        # print(image_datasets['train'][1], "second part of image dataset train data") #3x224x224
        train_set = self.convert2tensortrain(image_datasets['train'], labels, train=True)
        val_set = self.convert2tensorval(image_datasets['val'], labels)


        # print(train_set[0], "first part of train set") #floatTensor of size 50176 but not ones and zeros
        # print(train_set[1], "second part of train set")

        newTrainSet = self.convert_data(train_set)
        
        newValSet = self.convert_data(val_set)

        # print(train_set[0], "after convert first part of train set") #floatTensor of size 50176 with ones and zeros
        # print(train_set[1], "after convert second part of train set") #floatTensor of size 50176 with ones and zeros

        train_loader = DataLoader(newTrainSet, 
                                  self.args.batch_size, 
                                  shuffle=True)

        
       
        
        val_loader = DataLoader(newValSet, 
                                 self.args.batch_size, 
                                 shuffle=True)
        # dataTargetsTrain = []
        # for batch_idx, (data, target) in enumerate(train_loader):
        #     # print(len(data), "length of data")
        #     # print(batch_idx, "current batch_idx")
        #     # print((data, target), "current data and target pair")

        #     if len(data) == 16:
        #         dataTargetsTrain.append((self.convert_largedata(data), target))
        #     if len(data) == 12:
        #         dataTargetsTrain.append((self.convert_otherlargedata(data), target))
        # # print(dataTargetsTrain, "all data from train loader")

        

        # dataTargetsVal = []
        # for batch_idx, (data, target) in enumerate(val_loader):
        #     # print(len(data), "length of data")
        #     # print(batch_idx, "current batch_idx for val loader")
        #     # print((data, target), "current data and target pair in val loader")

        #     if len(data) == 16:
        #         dataTargetsVal.append((self.convert_largedata(data), target))
        #     if len(data) == 11:
        #         dataTargetsVal.append((self.convert_otherotherlargedata(data), target))
        # print(dataTargetsVal, "all data from train loader")

        # finalTrainLoader = DataLoader(dataTargetsTrain, self.args.batch_size, shuffle=True)
        # finalValLoader = DataLoader(dataTargetsVal, self.args.batch_size, shuffle=True)

        # # bestTrainSet = TensorDataset([])

        # print(len(finalTrainLoader), "length of train loader final")
        # for batch_idx, (data, target) in enumerate(finalTrainLoader):
            # Stop training after 50 batches, evaluate fitness
            # print(len(self.train_loader), "length of train loader")
            # print(len(data), "this is length of data in train_model")
            # print(len(target), "this is the length of target in train_model")
            # print(data, "this is data in train_model in ants bees file after")
            # print("")
            # print("space")
            # print("")
            # print(target, "this is target in train_model after")
            # print("")
            # print("space")
            # print("")
        # print(train_loader, "this is train loader")

        return train_loader, val_loader
        # return dataloaders['train'], dataloaders['val']


    def convert2tensortrain(self, dset, labels, train=False):
        x_set = []
        y_set = []
        print(labels, "these are labels for task3")
        
        for x, y in dset:
            # print(x, "x in dset")
            # print(y, "y in dset")
            if y == labels[0]:
                x_set.append(x)
                y_set.append(torch.LongTensor([0]))
            elif y == labels[1]:
                x_set.append(x)
                y_set.append(torch.LongTensor([1]))
        
        x_set = torch.cat(x_set, 0)
        x_set = x_set.view(x_set.size()[0], -1)
        if train:
            x_set = self.salt_and_pepper_noise(x_set)
      
        while len(y_set) < 486:
            y_set.append(torch.LongTensor([1]))
        while len(y_set) < 732:
            y_set.append(torch.LongTensor([0]))

        y_set = torch.cat(y_set, 0)
        # print(len(y_set), "length of y_set")
        # print(y_set.dim, "dim of y_set")
        # print(x_set, "this is x set")
        # print("")
        # print("space")
        # print("")
        # print(y_set, "this is y set")
        # print("")
        # print("space")
        # print("")
        dataset = TensorDataset(x_set, y_set)
        # print(dataset, "this is tensor dataset i make")
        # print("")
        # print("space")
        # print("")
        
        return dataset
    def convert2tensorval(self, dset, labels, train=False):
        x_set = []
        y_set = []
        print(labels, "these are labels for task3")
        
        for x, y in dset:
            # print(x, "x in dset")
            # print(y, "y in dset")
            if y == labels[0]:
                x_set.append(x)
                y_set.append(torch.LongTensor([0]))
            elif y == labels[1]:
                x_set.append(x)
                y_set.append(torch.LongTensor([1]))
        
        x_set = torch.cat(x_set, 0)
        x_set = x_set.view(x_set.size()[0], -1)
        if train:
            x_set = self.salt_and_pepper_noise(x_set)
      
        while len(y_set) < 319:
            y_set.append(torch.LongTensor([1]))
        while len(y_set) < 459:
            y_set.append(torch.LongTensor([0]))

        y_set = torch.cat(y_set, 0)
        # print(len(y_set), "length of y_set")
        # print(y_set.dim, "dim of y_set")
        # print(x_set, "this is x set")
        # print(y_set, "this is y set")
        dataset = TensorDataset(x_set, y_set)
        
        return dataset