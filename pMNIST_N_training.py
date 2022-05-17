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
from utils_MNIST import fully_connected_new,create_pMNIST_PCA_dataset, make_binary, create_XOR_dataset, create_teacher_dataset

#example XOR:    python pMNIST_N_training.py --dataset XOR --W 64 --depth 3 --input_dim 20 --n_epochs 500 --ensemble_size 5 --learning_rate 1e-3 --weight_decay 1e-3 --sample_size 1000 --batch_size 64 --signal_noise_ratio 1

parser =argparse.ArgumentParser()

parser.add_argument("--dataset", help="which dataset to use, pMNIST or XOR? (or maybe others in the future")
parser.add_argument("--W", help="width of the network to train",
                    type=int)
parser.add_argument("--depth", help= "depth of the network to train",
                    type=int)
parser.add_argument("--input_dim", help="input dimension, maximum 784", type=int, default=784)
parser.add_argument("--n_epochs", help="number of max epochs of training", type=int, default= 500)
parser.add_argument("--ensemble_size", help="how many networks to train to evaluate the limit error", type=int, default=5)
parser.add_argument("--learning_rate", help="learning rate", type=float, default=1e-3)
parser.add_argument("--weight_decay",help="weight decay", type=float, default=1e-3)
parser.add_argument("--sample_size", help="size of the training set, 60000 max", type=int, default=10000)
parser.add_argument("--batch_size", help="batch size during training", type=int, default=128)
parser.add_argument("--signal_noise_ratio", help="only useful with artificial data", type=float, default=1.)
parser.add_argument("--teacher_width", help="only useful in teacher-student setup", type=int, default=4)
parser.add_argument("--square_edge", help="only for generalized XOR", type=int, default=3)


args = parser.parse_args()
dataset=args.dataset
W=args.W
depth=args.depth
input_dim=args.input_dim
n_epochs=args.n_epochs
ensemble_size=args.ensemble_size
learning_rate=args.learning_rate
weight_decay=args.weight_decay
sample_size=args.sample_size
batch_size=args.batch_size
signal_noise_ratio=args.signal_noise_ratio
teacher_width=args.teacher_width
square_edge=args.square_edge

if dataset=='pMNIST':
    text_file = open(f'./logs/MNIST_{depth}_layer_{W}_lr_{learning_rate}_wd_{weight_decay}_inputdim_{input_dim}_training_parsed.txt', 'w')
if dataset=='XOR':
    text_file = open(f'./logs/{dataset}_{signal_noise_ratio}_ratio_{depth}_layer_{W}_lr_{learning_rate}_wd_{weight_decay}_'
                     f'inputdim_{input_dim}_training_parsed.txt', 'w')
if dataset=='teacher':
    text_file = open(f'./logs/{dataset}_teacherwidth_{teacher_width}_ratio_{signal_noise_ratio}_'
                     f'{depth}_layer_{W}_lr_{learning_rate}_wd_{weight_decay}_inputdim_{input_dim}_training_parsed.txt','w')
if dataset=='generalized_XOR':
    text_file = open(
        f'./logs/{dataset}_{square_edge}_edge_{signal_noise_ratio}_ratio_{depth}_layer_{W}_lr_{learning_rate}_wd_{weight_decay}_inputdim_'
        f'{input_dim}_training_parsed.txt','w')
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
            print("Saved dataset")
            trainset=torch.load(f'./data/MNIST_PCA_{input_dim}_train.pt')
            testset=torch.load(f'./data/MNIST_PCA_{input_dim}_test.pt')
    if args.sample_size!=60000:
        random_list=random.sample(range(60000), args.sample_size)
        trainset=torch.utils.data.Subset(trainset,random_list)



if dataset=='XOR':
    try:
        trainset=torch.load(f'./data/XOR_{input_dim}_dimension_{signal_noise_ratio}_ratio_train_{sample_size}_samples.pt')
        testset=torch.load(f'./data/XOR_{input_dim}_dimension_{signal_noise_ratio}_ratio_test_{10000}_samples.pt')
        print('Loaded Dataset')
    except IOError:
        create_XOR_dataset(args.sample_size,10000,input_dim,signal_noise_ratio)
        print('Saved Dataset')
        trainset = torch.load(f'./data/XOR_{input_dim}_dimension_{signal_noise_ratio}_ratio_train_{sample_size}_samples.pt')
        testset = torch.load(f'./data/XOR_{input_dim}_dimension_{signal_noise_ratio}_ratio_test_{10000}_samples.pt')



if dataset=='teacher':
    try:
        trainset = torch.load(f'./data/{dataset}_{input_dim}_dimension_teacherwidth_{teacher_width}_{signal_noise_ratio}_ratio_train_{sample_size}_samples.pt')
        testset = torch.load(f'./data/{dataset}_{input_dim}_dimension_teacherwidth_{teacher_width}_{signal_noise_ratio}_ratio_test_{10000}_samples.pt')
        print('Loaded Dataset')
    except IOError:
        create_teacher_dataset(sample_size, 10000, input_dim, teacher_width,signal_noise_ratio)
        print('Saved Dataset')
        trainset = torch.load(
            f'./data/{dataset}_{input_dim}_dimension_teacherwidth_{teacher_width}_{signal_noise_ratio}_ratio_train_{sample_size}_samples.pt')
        testset = torch.load(
            f'./data/{dataset}_{input_dim}_dimension_teacherwidth_{teacher_width}_{signal_noise_ratio}_ratio_test_{10000}_samples.pt')

if dataset=='generalized_XOR':
    trainset = torch.load(f'./data/{dataset}_{square_edge}_edge_{input_dim}_dimension_{signal_noise_ratio}_ratio_'
                          f'train_{sample_size}_samples.pt')
    testset = torch.load(f'./data/generalized_XOR_{square_edge}_edge_{input_dim}_dimension_'
                                              f'{signal_noise_ratio}_ratio_test_{10000}_samples.pt')


trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)
trainloader2 = torch.utils.data.DataLoader(trainset, batch_size=sample_size,
                                           shuffle=False, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=len(testset),
                                         shuffle=False, num_workers=2)



criterion = nn.BCEWithLogitsLoss()
criterion2= nn.BCELoss()
total_outputs=torch.zeros(len(testset))
total_maj_outputs=torch.zeros(len(testset))
test_accuracies_sum=0.

for k in range(ensemble_size):
    big_net = fully_connected_new(W, depth=depth, input_size=input_dim, output_size=1,
                                dropout=False, batch_norm=False, orthog_init=True)
    optimizer = optim.SGD(big_net.parameters(), lr=learning_rate,momentum=0.9, weight_decay=weight_decay)
    #optimizer = optim.Adam(big_net.parameters(), lr=0.001, weight_decay=args.weight_decay)
    train_losses = []
    test_losses=[]
    train_counter=[]
    counter=0
    still_training=0
    epoch=0
    #for epoch in range(args.n_epochs):  # loop over the dataset multiple times
    while still_training<5 and epoch<n_epochs:
        epoch+=1
        running_loss = 0.0
        correct=0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            if dataset=='XOR' or dataset=='generalized_XOR':
                outputs = big_net(inputs.float())
            if dataset=='pMNIST' or dataset=='teacher':
                outputs=big_net(inputs)
            outputs = torch.squeeze(outputs)
            loss = criterion(outputs.to(torch.float32), labels.to(torch.float32))
            predicted = torch.round(torch.sigmoid(outputs))
            correct += (predicted == labels).sum().item()
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.item()
        running_loss=running_loss/sample_size
        correct=correct/sample_size
        print(f'[{epoch + 1}] loss: {running_loss :.3f}', file=text_file)
        print(f'[{epoch + 1}] accuracy: {100 * correct :.3f}%', file=text_file)
        if (epoch+1)%100==0:
            print(f'[{epoch + 1}] accuracy: {100 * correct :.3f}%')
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
            if dataset == 'XOR' or dataset =='generalized_XOR':
                outputs = big_net(images.float())
            if dataset == 'pMNIST' or dataset=='teacher':
                outputs = big_net(images)
            #for predictions, one must apply a sigmoid, that the BCElogitsloss does implicitly
            predicted = torch.transpose(torch.round(torch.sigmoid(outputs)),0,1)
            outputs= torch.squeeze(outputs)
            #then we add the sigmoid of outputs to the average
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            #criterion is BCEwithlogits
            train_loss += criterion(outputs.to(torch.float32), labels.to(torch.float32))
    print(f'Accuracy of the network on the {sample_size} training images: {100 * correct / total} %')
    print(f'Accuracy of the network on the {sample_size} training images: {100 * correct / total} %', file = text_file)
    print(f'Loss of the network on the {sample_size} training images: { train_loss} ', file = text_file)
    if dataset=='pMNIST':
        torch.save(big_net.state_dict(), f'./nets/{dataset}_trained_{depth}_layer_{W}_net_{k+1}_lr_{learning_rate}_wd_{weight_decay}_'
                                         f'inputdim_{input_dim}.pth')
    if dataset=='XOR':
        torch.save(big_net.state_dict(),f'./nets/{dataset}_trained_{depth}_layer_{W}_net_{k + 1}_lr_{learning_rate}_wd_{weight_decay}'
                                        f'_inputdim_{input_dim}_ratio_{signal_noise_ratio}.pth')
    if dataset=='teacher':
        torch.save(big_net.state_dict(),f'./nets/{dataset}_trained_{depth}_layer_{W}_net_{k + 1}_lr_{learning_rate}_wd_{weight_decay}_'
                                        f'inputdim_{input_dim}_teacherwidth_{teacher_width}_ratio_{signal_noise_ratio}.pth')
    if dataset=='generalized_XOR':
        torch.save(big_net.state_dict(),
                   f'./nets/{dataset}_trained_{depth}_layer_{W}_net_{k + 1}_lr_{learning_rate}_wd_{weight_decay}'
                   f'_inputdim_{input_dim}_edge_{square_edge}_ratio_{signal_noise_ratio}.pth')

    print(f'Starting test of network # {k + 1}')
    correct = 0
    total = 0
    test_loss = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = big_net(images.float())
            # for predictions, one must apply a sigmoid, that the BCElogitsloss does implicitly
            predicted = torch.transpose(torch.round(torch.sigmoid(outputs)), 0, 1)
            outputs = torch.squeeze(outputs)
            # then we add the sigmoid of outputs to the average
            total_outputs.add_(torch.sigmoid(outputs))
            total_maj_outputs.add_(torch.squeeze(predicted))
            # print(total_outputs)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            # criterion is BCEwithlogits
            test_loss += criterion(outputs.to(torch.float32), labels.to(torch.float32))
    test_accuracies_sum+=100*correct/total
    print(f'Accuracy of the network on the 10000 test images: {100 * correct / total} %', file=text_file)
    print(f'Accuracy of the network on the 10000 test images: {100 * correct / total} %')
    print(f'Loss of the network on the 10000 test images: {test_loss} ', file=text_file)
    print(f'Finished test of network # {k + 1}')

total_outputs_1=total_outputs/ensemble_size
total_maj_outputs_1=total_maj_outputs/float(ensemble_size)
correct1=0
correct2=0
test_loss=0
total=0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        # calculate outputs by running images through the network
        # the class with the highest energy is what we choose as prediction
        predicted1 = torch.round(total_outputs_1)
        predicted2=torch.round(total_maj_outputs_1)
        total += labels.size(0)
        correct1 += (predicted1 == labels).sum().item()
        correct2 += (predicted2 == labels).sum().item()
        #print(total_outputs_1.to(torch.float32))
        test_loss += criterion2(total_outputs_1.to(torch.float32), labels.to(torch.float32))
accuracy_inf2=100*correct2/total
accuracy_inf1=100*correct1/total
print(f'Accuracy of the network on the 10000 test images via ensemble average: {accuracy_inf2} %', file = text_file)
print(f'Loss of the network on the 10000 test images via ensemble average: { test_loss} ', file = text_file)



if dataset == 'pMNIST':
    with open(f'./testerrors/ensemble_testerror_{dataset}_{depth}_layer_{W}_inputdim_{input_dim}_lr_{learning_rate}_wd_{weight_decay}.npy', 'wb') as f:
        np.save(f, accuracy_inf2)
if dataset == 'XOR':
    with open(f'./testerrors/ensemble_testerror_{dataset}_{depth}_layer_{W}_inputdim_{input_dim}_lr_{learning_rate}_wd_{weight_decay}_ratio_{signal_noise_ratio}.npy',
            'wb') as f:
        np.save(f, accuracy_inf2)
if dataset == 'teacher':
    with open(f'./testerrors/ensemble_testerror_{dataset}_{depth}_layer_{W}_inputdim_{input_dim}_lr_{learning_rate}_wd_{weight_decay}_'
              f'teacherwidth_{teacher_width}_ratio_{signal_noise_ratio}.npy', 'wb') as f:
        np.save(f, accuracy_inf2)
if dataset == 'generalized_XOR':
    with open(f'./testerrors/ensemble_testerror_{dataset}_{depth}_layer_{W}_inputdim_{input_dim}_lr_{learning_rate}_wd_{weight_decay}_'
              f'edge_{square_edge}_ratio_{signal_noise_ratio}.npy','wb') as f:
        np.save(f, accuracy_inf2)

#delta_error=(test_accuracies_sum-ensemble_size*accuracy_inf2)/ensemble_size
delta_error=(ensemble_size*accuracy_inf2-test_accuracies_sum)/ensemble_size

if dataset == 'pMNIST':
    with open(f'./deltaerrors/ensemble_deltaerror_{dataset}_{depth}_layer_{W}_inputdim_{input_dim}_lr_{learning_rate}_wd_{weight_decay}.npy', 'wb') as f:
        np.save(f, delta_error)
if dataset == 'XOR':
    with open(f'./deltaerrors/ensemble_deltaerror_{dataset}_{depth}_layer_{W}_inputdim_{input_dim}_lr_{learning_rate}_wd_{weight_decay}_ratio_{signal_noise_ratio}.npy',
            'wb') as f:
        np.save(f, delta_error)
        print("Saved XOR accuracy")
if dataset == 'teacher':
    with open(f'./deltaerrors/ensemble_deltaerror_{dataset}_{depth}_layer_{W}_inputdim_{input_dim}_lr_{learning_rate}_wd_{weight_decay}_'
              f'teacherwidth_{teacher_width}_ratio_{signal_noise_ratio}.npy', 'wb') as f:
        np.save(f, delta_error)
if dataset == 'generalized_XOR':
    with open(f'./deltaerrors/ensemble_deltaerror_{dataset}_{depth}_layer_{W}_inputdim_{input_dim}_lr_{learning_rate}_wd_{weight_decay}_'
              f'edge_{square_edge}_ratio_{signal_noise_ratio}.npy',
            'wb') as f:
        np.save(f, delta_error)


fig = plt.figure()
plt.plot(train_counter, train_losses, color='blue')
plt.legend(['Train Loss'], loc='upper right')
plt.xlabel('number of training examples seen')
plt.ylabel('training loss')
if dataset=='pMNIST':
    plt.savefig(f'./training_losses_plots/{dataset}_{depth}_layer_{W}_inputdim_{input_dim}_training_loss.png')
if dataset=='XOR':
    plt.savefig(f'./training_losses_plots/{dataset}_{depth}_layer_{W}_inputdim_{input_dim}_ratio_{signal_noise_ratio}_training_loss.png')
if dataset=='teacher':
    plt.savefig(f'./training_losses_plots/{dataset}_{depth}_layer_{W}_inputdim_{input_dim}_teacherwidth_{teacher_width}_ratio_{signal_noise_ratio}_training_loss.png')
if dataset=='generalized_XOR':
    plt.savefig(f'./training_losses_plots/{dataset}_{depth}_layer_{W}_inputdim_{input_dim}_'
                f'edge_{square_edge}_ratio_{signal_noise_ratio}_training_loss.png')

text_file.close()

print(f'End of training of a net with the following args: dataset {dataset} width {W} depth {depth} input_dim {input_dim} '
      f'n_epochs {n_epochs} number_nets {ensemble_size} learning_rate {learning_rate} weight_decay {weight_decay} '
      f'size_of_training_set {sample_size} batch_size {batch_size} signal_noise_ratio {signal_noise_ratio} square_edge {square_edge} teacher_width {teacher_width}')

