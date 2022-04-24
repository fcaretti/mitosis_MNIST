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
from utils_MNIST import fully_connected_new,PCA,create_pMNIST_PCA_dataset, make_binary, create_XOR_dataset

parser =argparse.ArgumentParser()

parser.add_argument("dataset", help="which dataset to use, pMNIST or XOR? (or maybe other in the future")
parser.add_argument("W", help="width of the network to train",
                    type=int)
parser.add_argument("depth", help= "depth of the network to train",
                    type=int)
parser.add_argument("input_dim", help="input dimension, maximum 784", type=int, default=784)
parser.add_argument("n_epochs", help="number of epochs of training", type=int, default= 500)
parser.add_argument("ensemble_size", help="how many networks to train to evaluate the limit error", type=int, default=5)
parser.add_argument("learning_rate", help="learning rate", type=float, default=1e-3)
parser.add_argument("weight_decay",help="weight decay", type=float, default=1e-3)
parser.add_argument("sample_size", help="size of the training set, 60000 max", type=int, default=10000)
parser.add_argument("batch_size", help="batch size during training", type=int, default=128)
parser.add_argument("signal_noise_ratio", help="only useful with artificial data", type=float, default=1.)


args = parser.parse_args()
dataset=args.dataset
signal_noise_ratio=args.signal_noise_ratio
if dataset=='pMNIST':
    text_file = open(f'./logs/MNIST_{args.depth}_layer_{args.W}_lr_{args.learning_rate}_wd_{args.weight_decay}_inputdim_{args.input_dim}_training_parsed.txt', 'w')
if dataset=='XOR':
    text_file = open(f'./logs/{dataset}_{signal_noise_ratio}_ratio_{args.depth}_layer_{args.W}_lr_{args.learning_rate}_wd_{args.weight_decay}_inputdim_{args.input_dim}_training_parsed.txt', 'w')

input_dim=args.input_dim
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize(0.5, 0.5)])
if dataset=='pMNIST':
    trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                          download=True, transform=transform,target_transform=make_binary())
    testset = torchvision.datasets.MNIST(root='./data', train=False,
                                         download=True, transform=transform,target_transform=make_binary())
    # Here we just convert MNIST into parity MNIST

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
if dataset=='XOR':
    try:
        trainset=torch.load(f'./data/XOR_{input_dim}_dimension_{signal_noise_ratio}_ratio_train_{args.sample_size}_samples.pt')
        testset=torch.load(f'./data/XOR_{input_dim}_dimension_{signal_noise_ratio}_ratio_test_{10000}_samples.pt')
        print('Loaded Dataset')

    except IOError:
        create_XOR_dataset(args.sample_size,10000,input_dim,signal_noise_ratio)
        print('Saved Dataset')
        trainset = torch.load(f'./data/XOR_{input_dim}_dimension_{signal_noise_ratio}_ratio_train_{args.sample_size}_samples.pt')
        testset = torch.load(f'./data/XOR_{input_dim}_dimension_{signal_noise_ratio}_ratio_test_{10000}_samples.pt')

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
    optimizer = optim.SGD(big_net.parameters(), lr=args.learning_rate,momentum=0.9, weight_decay=args.weight_decay)
    #optimizer = optim.Adam(big_net.parameters(), lr=0.001, weight_decay=args.weight_decay)
    train_losses = []
    test_losses=[]
    train_counter=[]
    counter=0
    still_training=0
    epoch=0
    #for epoch in range(args.n_epochs):  # loop over the dataset multiple times
    while still_training<5 and epoch<args.n_epochs:
        epoch+=1
        running_loss = 0.0
        correct=0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            if dataset=='XOR':
                outputs = big_net(inputs.float())
            if dataset=='pMNIST':
                outputs=big_net(inputs)
            outputs = torch.squeeze(outputs)
            loss = criterion(outputs.to(torch.float32), labels.to(torch.float32))
            predicted = torch.round(torch.sigmoid(outputs))
            correct += (predicted == labels).sum().item()
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.item()
        running_loss=running_loss/args.sample_size
        correct=correct/args.sample_size
        print(f'[{epoch + 1}] loss: {running_loss :.3f}', file=text_file)
        print(f'[{epoch + 1}] accuracy: {100 * correct :.3f}%', file=text_file)
        if correct==1:
            still_training+=1
        train_losses.append(running_loss)
        train_counter.append(counter)
        counter = counter + 1
        running_loss = 0.0
        correct=0
    print(f'Finished Training in {epoch} epochs')
    correct = 0
    total = 0
    train_loss=0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in trainloader2:
            images, labels = data
            if dataset == 'XOR':
                outputs = big_net(images.float())
            if dataset == 'pMNIST':
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
    if dataset=='pMNIST':
        torch.save(big_net.state_dict(), f'./nets/{dataset}_trained_{args.depth}_layer_{args.W}_net_{k+1}_lr_{args.learning_rate}_wd_{args.weight_decay}_inputdim_{input_dim}.pth')
    if dataset=='XOR':
        torch.save(big_net.state_dict(),f'./nets/{dataset}_trained_{args.depth}_layer_{args.W}_net_{k + 1}_lr_{args.learning_rate}_wd_{args.weight_decay}_inputdim_{input_dim}_ratio_{signal_noise_ratio}.pth')

fig = plt.figure()
plt.plot(train_counter, train_losses, color='blue')
plt.legend(['Train Loss'], loc='upper right')
plt.xlabel('number of training examples seen')
plt.ylabel('training loss')
if dataset=='pMNIST':
    plt.savefig(f'./training_losses_plots/{dataset}_{args.depth}_layer_{args.W}_inputdim_{args.input_dim}_training_loss.png')
if dataset=='XOR':
    plt.savefig(
        f'./training_losses_plots/{dataset}_{args.depth}_layer_{args.W}_inputdim_{args.input_dim}_ratio_{signal_noise_ratio}_training_loss.png')
text_file.close()