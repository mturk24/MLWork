from __future__ import print_function, division
import matplotlib.pyplot as plt
import time
import os
import torch
import numpy as np
from torch.autograd import Variable
import itertools
from copy import deepcopy
import torch.optim as optim
import torch.nn as nn
import torchvision
from torchvision import datasets, models, transforms
from torch.optim import lr_scheduler


class EvolutionTrainer(object):
    def __init__(self, model, optimizer, loss_func, 
                 train_loader, test_loader, args, 
                 convergence_threshold, batch_epochs=50):
        
        self.model = model
        self.args = args
        self.loss_func = loss_func
        self.batch_epochs = batch_epochs
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.convergence_threshold = convergence_threshold
        self.optimizer = optimizer
      
    def initialize_pathways(self):
        layer_configs = list(itertools.combinations_with_replacement(
                                    list(range(self.args.M)), self.args.N))
        layer_configs = np.array(layer_configs)
        indices = np.random.choice(len(layer_configs), (self.args.P, self.args.L))
        pathways = layer_configs[indices]

        return pathways # Shape: P x L x N
    
    def mutate(self, pathway):
        prob_mutate = 1./ (self.args.L * self.args.N) # Increase probability of mutation

        # Probability of mutation for every element
        prob = np.random.rand(self.args.L, self.args.N)

        # Mutations for chosen elements
        permutations = np.random.randint(-2, 2, size=(self.args.L, self.args.N))
        permutations[prob > prob_mutate] = 0

        # Mutate
        pathway = (pathway + permutations) % self.args.M
        
        return pathway
    
    def evaluate(self, pathway):
        correct = 0
        
        for x, y in self.test_loader:
            if self.args.type_task == 3:
                if len(x) == 16:
                    x = self.convert_largedata(x)
                elif len(x) == 12:
                    x = self.convert_otherlargedata(x)
                elif len(x) == 11:
                    x = self.convert_otherotherlargedata(x)
            print(x, "this is x in self.test_loader in evaluate")
            print("")
            print("space")
            print("")
            print(y, "this is y in self.test_loader in evaluate")
            x, y = Variable(x, volatile=True), Variable(y, volatile=True)
            
            output = self.model(x, pathway)
            _, pred = torch.max(output.data, 1)
            
            correct += (pred == y.data).sum()
        # print(self.test_loader, "this is test loader in evaluate function of fitness")
        # print("space")
        # print(correct, "this is calculated sum of prediction")
        accuracy = correct * 1.0 / len(self.test_loader) / self.args.batch_size

        # print(accuracy, "this is accuracy in evaluate function")


        return accuracy

    # def train_model_antsbees(self, pathway):
    #     for batch_idx, (data, target) in enumerate(self.train_loader):
    #         # Stop training after 50 batches, evaluate fitness
    #         # print(len(self.train_loader), "length of train loader")
    #         # print(len(data), "this is length of data in train_model")
    #         # print(len(target), "this is the length of target in train_model")
    #         if batch_idx >= self.batch_epochs:
    #             fitness = self.evaluate(pathway)
    #             return fitness
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

    def train_modelOther(self, model, criterion, optimizer, scheduler, num_epochs):
        use_gpu = torch.cuda.is_available()
        since = time.time()

        best_model_wts = model.state_dict()
        best_acc = 0.0
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
        # dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
        #                                              shuffle=True, num_workers=4)
        #               for x in ['train', 'val']}
        # print(dataloaders, "this is dataloaders")
        # dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
        # class_names = image_datasets['train'].classes

        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    scheduler.step()
                    model.train(True)  # Set model to training mode
                else:
                    model.train(False)  # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                # for data in dataloaders[phase]:
                #     # get the inputs
                #     print(data, "this is data")
                #     inputs, labels = data
                #     print(inputs, "these are inputs")
                #     print(labels, "these are labels")

                #     # wrap them in Variable
                #     if use_gpu:
                #         inputs = Variable(inputs.cuda())
                #         labels = Variable(labels.cuda())
                #     else:
                #         inputs, labels = Variable(inputs), Variable(labels)

                #     # zero the parameter gradients
                #     optimizer.zero_grad()

                #     # forward
                #     outputs = model(inputs)
                #     print(outputs, "these are outputs")
                #     _, preds = torch.max(outputs.data, 1)
                #     print(preds, "predictions")
                #     loss = criterion(outputs, labels)

                #     # backward + optimize only if in training phase
                #     if phase == 'train':
                #         loss.backward()
                #         optimizer.step()

                #     # statistics
                #     running_loss += loss.data[0]
                #     running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects / dataset_sizes[phase]

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = model.state_dict()

            print()

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

        # load best model weights
        model.load_state_dict(best_model_wts)
        return model 
    
    def train_model(self, pathway):
        for batch_idx, (data, target) in enumerate(self.train_loader):
            # Stop training after 50 batches, evaluate fitness
            # print(len(self.train_loader), "length of train loader")
            # print(len(data), "this is length of data in train_model")
            # print(len(target), "this is the length of target in train_model")
            # print(batch_idx, "this is batch idx currently")
            # print(self.batch_epochs, "this is batch epochs")
            if batch_idx >= self.batch_epochs:
                fitness = self.evaluate(pathway)
                print(fitness, "this is fitness after evaluate")
                return fitness

            if batch_idx == 45 and self.args.type_task == 3:
                fitness = self.evaluate(pathway)
                print(fitness, "this is fitness after evaluate")
                return fitness

            if self.args.type_task == 3:
                if len(data) == 16:
                    data = self.convert_largedata(data)
                elif len(data) == 12:
                    data = self.convert_otherlargedata(data)

            # print(data, "this is data in train_model in evolution")
            # print("")
            # print("space")
            # print("")
            # print(target, "this is target")
            # print("")
            # print("space")
            # print("")
            # print((data, target), "this is tuple of both")
            # print("")
            # print("space")
            # print("")
            self.optimizer.zero_grad()
            # print(self.optimizer.zero_grad(), "zero")


            data, target = Variable(data), Variable(target)
            
            # print(data, "this is data after making it a variable")
            
            output = self.model(data, pathway)
            # print(output, "this is output after passing in data and pathway")
            

            loss = self.loss_func(output, target)
            # print(loss, "loss after passing in output and target")

            loss.backward()
            # print(loss.backward(), "backward loss computation")
            self.optimizer.step()
            # print(self.optimizer.step(), "step of optimizer")
    
    def train(self):
        
        self.model.train()
        
        fitnesses = []
        best_pathway = None
        best_fitness = -float('inf')
        pathways = self.initialize_pathways()
        gen = 0
        
        while best_fitness < self.convergence_threshold and gen <= 20:

            chosen_pathways = pathways[np.random.choice(self.args.P, 2)]
            # print(chosen_pathways, "chosen pathways here")
            
            current_fitnesses = []

            
            for pathway in chosen_pathways:
                print(pathway, "current pathway in train")

                if self.args.type_task == 3:

                    use_gpu = torch.cuda.is_available()
                    model_conv = torchvision.models.resnet18(pretrained=True)
                    for param in model_conv.parameters():
                        param.requires_grad = False

                    # Parameters of newly constructed modules have requires_grad=True by default
                    num_ftrs = model_conv.fc.in_features
                    model_conv.fc = nn.Linear(num_ftrs, 2)

                    if use_gpu:
                        model_conv = model_conv.cuda()

                    criterion = nn.CrossEntropyLoss()

                    # Observe that only parameters of final layer are being optimized as
                    # opposed to before.
                    # optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)
                    optimizer_conv = optim.Adam(model_conv.fc.parameters(), lr=0.001)

                    # Decay LR by a factor of 0.1 every 7 epochs
                    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)


                    ######################################################################
                    # Train and evaluate
                    # ^^^^^^^^^^^^^^^^^^
                    #
                    # On CPU this will take about half the time compared to previous scenario.
                    # This is expected as gradients don't need to be computed for most of the
                    # network. However, forward does need to be computed.
                    #

                    model_conv = self.train_modelOther(model_conv, criterion, optimizer_conv,
                                             exp_lr_scheduler, 1)
                    fitness = self.train_model(pathway)
                    print(fitness, "this is fitness of model_conv ants bees")

                fitness = self.train_model(pathway)
                print(fitness, "this is fitness in train of mnist task")
                
                current_fitnesses.append(fitness)
                
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_pathway = pathway
                
            # All pathways finished evaluating, copy the one with highest fitness
            # to all other ones and mutate
            pathways = np.array([best_pathway] + [self.mutate(deepcopy(best_pathway)) 
                                              for _ in range(self.args.P - 1)])
            
            fitnesses.append(max(current_fitnesses))
            
            if gen % 20 == 0:
                print('Generation {} best fitness is {}'.format(gen, best_fitness))
            gen += 1
        
        # Task training is done
        # print()
        self.model.done_task(best_pathway)
        
        return best_pathway, gen, fitnesses