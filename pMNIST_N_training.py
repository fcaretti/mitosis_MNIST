import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.optim as optim
import random
from collections import OrderedDict
import math
import argparse
import pickle
from utils_MNIST import fully_connected_new,PCA,create_pMNIST_PCA_dataset
#python pMNIST_N_training.py 512 5 10 500 5 1e-3 10000 128
#python pMNIST_N_training.py 2048 2 784 500 5 1e-4 10000 128
parser = argparse.ArgumentParser()
parser.add_argument("W", help="width of the network to train",
                    type=int)
parser.add_argument("depth", help= "depth of the network to train",
                    type=int)
parser.add_argument("input_dim", help="input dimension, maximum 784", type=int, default=784)
parser.add_argument("n_epochs", help="number of epochs of training", type=int, default= 500)
parser.add_argument("ensemble_size", help="how many networks to train to evaluate the limit error", type=int, default=5)
parser.add_argument("weight_decay",help="weight decay", type=float, default=1e-3)
parser.add_argument("sample_size", help="size of the training set, 60000 max", type=int, default=10000)
parser.add_argument("batch_size", help="bathc size during training", type=int, default=128)
args = parser.parse_args()
text_file = open(f'./logs/MNIST_{args.depth}_layer_{args.W}_wd_{args.weight_decay}_inputdim_{args.input_dim}_training_parsed.txt', 'w')



transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize(0.5, 0.5)])

trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                      download=True, transform=transform)
testset = torchvision.datasets.MNIST(root='./data', train=False,
                                     download=True, transform=transform)
# Here we just convert MNIST into parity MNIST

input_dim=args.input_dim
if input_dim != 784:
    try:
        trainset = torch.load(f'./data/trainset_dim_{input_dim}.pt')
        testset = torch.load(f'./data/testset_dim_{input_dim}.pt')
        print('Loaded dataset')
    except IOError:
        trainset,testset=PCA(trainset,testset,784,input_dim)
        torch.save(trainset, f'./data/trainset_dim_{input_dim}.pt')
        torch.save(testset, f'./data/testset_dim_{input_dim}.pt')
        print('Saved Dataset')

    #trainset,testset=PCA(trainset,testset,784,input_dim)
    #torch.save(trainset, f'./data/trainset_dim_{input_dim}.pt')
    #torch.save(testset, f'./data/testset_dim_{input_dim}.pt')
#with open(f'testset_dim_{input_dim}.pickle', 'wb') as f:
    #pickle.dump(testset, f)
#create_pMNIST_PCA_dataset(trainset, testset, 784, args.input_dim)
#trainset=torch.open(f'./data/MNIST_PCA_{final_dim}_train')
trainset = list(trainset)
for i in range(len(trainset)):
    trainset[i] = list(trainset[i])

for i in range(0, 60000):
    if trainset[i][1] % 2 == 0:
        trainset[i][1] = 0
    else:
        trainset[i][1] = 1

for i in range(len(trainset)):
    trainset[i] = tuple(trainset[i])
trainset = tuple(trainset)
trainset = trainset[:args.sample_size]
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                          shuffle=True, num_workers=2)
trainloader2 = torch.utils.data.DataLoader(trainset, batch_size=args.sample_size,
                                           shuffle=False, num_workers=2)

testset = list(testset)
for i in range(len(testset)):
    testset[i] = list(testset[i])

for i in range(0, len(testset)):
    if testset[i][1] % 2 == 0:
        testset[i][1] = 0

    else:
        testset[i][1] = 1
testset = tuple(testset)

testloader = torch.utils.data.DataLoader(testset, batch_size=len(testset),
                                         shuffle=False, num_workers=2)

criterion = nn.BCEWithLogitsLoss()
criterion2= nn.BCELoss()


total_outputs=torch.zeros(len(testset))
for k in range(args.ensemble_size):
    big_net = fully_connected_new(args.W, depth=args.depth, input_size=input_dim, output_size=1,
                                dropout=False, batch_norm=False, orthog_init=True)
    #print(big_net)
    #optimizer = optim.SGD(big_net.parameters(), lr=0.001,momentum=0.9, weight_decay=args.weight_decay)
    optimizer = optim.Adam(big_net.parameters(), lr=0.001, weight_decay=args.weight_decay)
    train_losses = []
    test_losses=[]
    train_counter=[]
    counter=0
    for epoch in range(args.n_epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = big_net(inputs)
            outputs = torch.squeeze(outputs)
            #print(outputs)
            loss = criterion(outputs.to(torch.float32), labels.to(torch.float32))
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i%35 == 34:
                predicted = torch.round(torch.sigmoid(outputs))
                correct = (predicted == labels).sum().item()
                print(f'[{epoch + 1}, {i +1:5d}] loss: {running_loss / (i):.3f}', file = text_file)
                print(f'[{epoch + 1}, {i +1:5d}] accuracy: {100*correct / (args.batch_size):.3f}%', file = text_file)
                train_losses.append(running_loss)
                train_counter.append(counter)
                counter=counter+1
                running_loss = 0.0
    print('Finished Training')
    correct = 0
    total = 0
    train_loss=0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in trainloader2:
            images, labels = data
            outputs = big_net(images)
            #for predictions, one must apply a sigmoid, that the BCElogitsloss does implicitly
            predicted = torch.transpose(torch.round(torch.sigmoid(outputs)),0,1)
            outputs= torch.squeeze(outputs)
            #then we add the sigmoid of outputs to the average
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            #criterion is BCEwithlogits
            train_loss += criterion(outputs.to(torch.float32), labels.to(torch.float32))
    print(f'Accuracy of the network on the {args.sample_size} training images: {100 * correct / total} %', file = text_file)
    print(f'Loss of the network on the {args.sample_size} training images: { train_loss} ', file = text_file)
    torch.save(big_net.state_dict(), f'./nets/mnist_trained_{args.depth}_layer_{args.W}_net_{k+1}_parsed_wd_{args.weight_decay}_inputdim_{input_dim}.pth')

fig = plt.figure()
plt.plot(train_counter, train_losses, color='blue')
plt.legend(['Train Loss'], loc='upper right')
plt.xlabel('number of training examples seen')
plt.ylabel('training loss')
plt.savefig(f'./training_losses_plots/pMNIST_{args.depth}_layer_{args.W}_inputdim_{args.input_dim}_training_loss')

text_file.close()