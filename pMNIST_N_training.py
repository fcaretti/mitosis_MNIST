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
from utils_MNIST import fully_connected_new,PCA,create_pMNIST_PCA_dataset, make_binary
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
parser.add_argument("batch_size", help="batch size during training", type=int, default=128)

args = parser.parse_args()
text_file = open(f'./logs/MNIST_{args.depth}_layer_{args.W}_wd_{args.weight_decay}_inputdim_{args.input_dim}_training_parsed.txt', 'w')



transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize(0.5, 0.5)])

trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                      download=True, transform=transform,target_transform=make_binary())
testset = torchvision.datasets.MNIST(root='./data', train=False,
                                     download=True, transform=transform,target_transform=make_binary())
# Here we just convert MNIST into parity MNIST

input_dim=args.input_dim
if input_dim !=784:
    try:
        trainset=torch.load(f'./data/MNIST_PCA_{input_dim}_train.pt')
        testset=torch.load(f'./data/MNIST_PCA_{input_dim}_test.pt')
        print("Loaded dataset")
    except IOError:
        create_pMNIST_PCA_dataset(trainset,testset,784,input_dim)
        trainset=torch.load(f'./data/MNIST_PCA_{input_dim}_train.pt')
        testset=torch.load(f'./data/MNIST_PCA_{input_dim}_test.pt')
        print("Saved dataset")


if args.sample_size!=60000:
    random_list=random.sample(range(60000), args.sample_size)
    trainset=torch.utils.data.Subset(trainset,random_list)


trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                          shuffle=True, num_workers=2)
trainloader2 = torch.utils.data.DataLoader(trainset, batch_size=args.sample_size,
                                           shuffle=False, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=len(testset),
                                         shuffle=False, num_workers=2)

criterion = nn.BCEWithLogitsLoss()
criterion2= nn.BCELoss()

total_outputs=torch.zeros(len(testset))

for k in range(args.ensemble_size):
    big_net = fully_connected_new(args.W, depth=args.depth, input_size=input_dim, output_size=1,
                                dropout=False, batch_norm=False, orthog_init=True)
    optimizer = optim.SGD(big_net.parameters(), lr=0.001,momentum=0.9, weight_decay=args.weight_decay)
    #optimizer = optim.Adam(big_net.parameters(), lr=0.001, weight_decay=args.weight_decay)
    train_losses = []
    test_losses=[]
    train_counter=[]
    counter=0
    for epoch in range(args.n_epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        correct=0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = big_net(inputs)
            outputs = torch.squeeze(outputs)
            loss = criterion(outputs.to(torch.float32), labels.to(torch.float32))
            predicted = torch.round(torch.sigmoid(outputs))
            correct += (predicted == labels).sum().item()
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            '''if i%35 == 34:
                predicted = torch.round(torch.sigmoid(outputs))
                correct = (predicted == labels).sum().item()
                print(f'[{epoch + 1}, {i +1:5d}] loss: {running_loss / (i):.3f}', file = text_file)
                print(f'[{epoch + 1}, {i +1:5d}] accuracy: {100*correct / (args.batch_size):.3f}%', file = text_file)
                train_losses.append(running_loss)
                train_counter.append(counter)
                counter=counter+1
                running_loss = 0.0'''
        running_loss=running_loss/args.sample_size
        correct=correct/args.sample_size
        print(f'[{epoch + 1}] loss: {running_loss :.3f}', file=text_file)
        print(f'[{epoch + 1}] accuracy: {100 * correct :.3f}%', file=text_file)
        train_losses.append(running_loss)
        train_counter.append(counter)
        counter = counter + 1
        running_loss = 0.0
        correct=0
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