import argparse
from mnist_task import BinaryClassificationTask
from ants_bees_task import AntsAndBeesTask
from pathnet import PathNet
from evolutionary_trainer import EvolutionTrainer
import torch.optim as optim
import torch.nn as nn
import numpy as np

parser = argparse.ArgumentParser(description='PathNet MNIST Binary Classification Task')

parser.add_argument('--L', type=int, default=3, help='Number of layers')
parser.add_argument('--M', type=int, default=10, help='Number of modules in each layer')
parser.add_argument('--N', type=int, default=3, help='Number of maximum allowable active modules \
                                                      in one layer')

parser.add_argument('--P', type=int, default=64, help='Number of population pathways')
parser.add_argument('--download', type=bool, default=False, help='Download MNIST dataset')
parser.add_argument('--noise-prob', type=int, default=0.5, help='Probability for salt and pepper noise.')
parser.add_argument('--batch-size', type=int, default=16, help='Batch size for data')
parser.add_argument('--n-tasks', type=int, default=3, help='Number of tasks to transfer learn \
                                                             (not presented in the paper, \
                                                              for personal experiment only.)')
parser.add_argument('--type_task', type=int, default=1, help='type of task by number')

args = parser.parse_args()

def main():
    # Data loader
    task = BinaryClassificationTask(args)

    
    antsBeesTask = AntsAndBeesTask(args)
    # PathNet
    pathnet = PathNet(args)
    loss_func = nn.CrossEntropyLoss()
    task_layers = [nn.Sequential(nn.Linear(20, 2), 
                                 nn.Softmax()) for _ in range(args.n_tasks)]


    #this loop gets run twice since 2 tasks - so if i increase tasks 
    for i in range(args.n_tasks):
        if i == 2:
          args.type_task = 3
          pathnet = PathNet(args)
          train_loader, test_loader = antsBeesTask.init() #RuntimeError: size mismatch, m1: [2688 x 224], m2: [784 x 20] 
        else:
          train_loader, test_loader = task.init()

        print("Starting task {}".format(i))
        # Optimizer / loss
        pathnet.initialize_new_task(task_layers[i])
        optimizer_params = pathnet.get_optimizer_params()
        optimizer = optim.Adam(optimizer_params, lr=0.001)
          
        # PathNet Trainer
        
        evol_trainer = EvolutionTrainer(pathnet, optimizer, loss_func, 
                                                train_loader, test_loader, args,
                                                convergence_threshold = 0.99)

        best_task_pathway, converge_generation, fitnesses = evol_trainer.train()
          
        np.save('task_{}_fitnesses.npy'.format(i), fitnesses)
          
        print("Task {} converges at generation {}".format(i, converge_generation))
     

if __name__ == '__main__':
    main()
     
