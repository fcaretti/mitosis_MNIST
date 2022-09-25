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
import saving_utils
import utils_MNIST

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
parser.add_argument("--batch_size", help="batch size during training", type=int, default=64)
parser.add_argument("--signal_noise_ratio", help="only useful with artificial data", type=float, default=1.)
parser.add_argument("--teacher_width", help="only useful in teacher-student setup", type=int, default=4)
parser.add_argument("--square_edge", help="only for generalized XOR", type=int, default=3)
parser.add_argument("--optimizer", help="Choice between Adam and SGD", default='SGD')
parser.add_argument("--task", help="Only for teacher ATM, as in Chizat-Bach", default='classification')

args = parser.parse_args()
dataset=args.dataset
W=args.W
depth=args.depth
input_dim=args.input_dim
n_epochs=args.n_epochs
ensemble_size=args.ensemble_size
learning_rate=args.learning_rate
weight_decay=args.weight_decay
if args.weight_decay==0.:
    weight_decay=None
sample_size=args.sample_size
batch_size=args.batch_size
signal_noise_ratio=args.signal_noise_ratio
teacher_width=args.teacher_width
square_edge=args.square_edge
optimizer_choice=args.optimizer
task=args.task

text_file=saving_utils.open_text_file(dataset,depth,W,learning_rate,weight_decay, input_dim, optimizer_choice, teacher_width, signal_noise_ratio, square_edge)

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.1307,),(0.3081))])



trainset,testset=saving_utils.dataset_loader('train',dataset, transform, input_dim, sample_size,signal_noise_ratio,teacher_width, square_edge)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)
trainloader2 = torch.utils.data.DataLoader(trainset, batch_size=sample_size,
                                           shuffle=False, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=len(testset),
                                         shuffle=False, num_workers=2)



criterion = nn.BCEWithLogitsLoss()
criterion2= nn.BCELoss()
criterion3=nn.MSELoss()
total_outputs=torch.zeros(len(testset))
total_maj_outputs=torch.zeros(len(testset))
test_accuracies_sum=0.

for k in range(ensemble_size):
    big_net = utils_MNIST.fully_connected_new(W, depth=depth, input_size=input_dim, output_size=1,
                                dropout=False, batch_norm=False, orthog_init=True)
    if optimizer_choice=='Adam':
        optimizer = optim.Adam(big_net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    if optimizer_choice=='SGD':
        optimizer = optim.SGD(big_net.parameters(), lr=learning_rate,momentum=0.9, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = n_epochs, eta_min = 10**-5)
    train_losses = []
    test_losses=[]
    train_counter=[]
    counter=0
    still_training=0
    epoch=0
    #for epoch in range(args.n_epochs):  # loop over the dataset multiple times
    while still_training<200 and epoch<n_epochs:
        epoch+=1
        running_loss = 0.0
        correct=0
        actual_sample_size=0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            if dataset=='XOR' or dataset=='generalized_XOR':
                outputs = big_net(inputs.float())
            if dataset=='pMNIST' or dataset=='teacher':
                outputs = big_net(inputs)
            outputs = torch.squeeze(outputs)
            loss = criterion(outputs.to(torch.float32), labels.to(torch.float32))
            predicted = torch.round(torch.sigmoid(outputs))
            correct += (predicted == labels).sum().item()
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.item()
            actual_sample_size+=labels.size(0)
        if optimizer_choice=='SGD':
            scheduler.step()
        running_loss=running_loss/sample_size
        #correct=correct/float(sample_size)
        correct=correct/float(actual_sample_size)
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

    #Evaluate accuracy on the training set
    train_acc, train_loss,_=utils_MNIST.evaluate_on_dataset(dataset,trainloader2,big_net,criterion)
    print(f'Accuracy of the network on the {sample_size} training images: {train_acc} %')
    print(f'Accuracy of the network on the {sample_size} training images: {train_acc} %', file=text_file)
    print(f'Loss of the network on the {sample_size} training images: {train_loss} ', file=text_file)

    saving_utils.save_weights(big_net, dataset, depth, W, k, learning_rate, weight_decay, input_dim, optimizer_choice,n_epochs, square_edge,
                 signal_noise_ratio,teacher_width)

    print(f'Starting test of network # {k + 1}')
    test_acc,test_loss,outputs=utils_MNIST.evaluate_on_dataset(dataset,testloader,big_net,criterion)
    total_outputs+=torch.sigmoid(outputs)
    test_accuracies_sum+=test_acc
    print(f'Accuracy of the network on the 10000 test images: {test_acc} %', file=text_file)
    print(f'Accuracy of the network on the 10000 test images: {test_acc} %')
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
        #predicted2=torch.round(total_maj_outputs_1)
        total += labels.size(0)
        correct1 += (predicted1 == labels).sum().item()
        #correct2 += (predicted2 == labels).sum().item()
        #print(total_outputs_1.to(torch.float32))
        test_loss += criterion2(total_outputs_1.to(torch.float32), labels.to(torch.float32))
#accuracy_inf2=100*correct2/total
accuracy_inf1=100*correct1/total
accuracy_inf=accuracy_inf1
print(f'Accuracy of the network on the 10000 test images via ensemble average: {accuracy_inf} %', file = text_file)
print(f'Accuracy of the network on the 10000 test images via ensemble average: {accuracy_inf} %')
print(f'Loss of the network on the 10000 test images via ensemble average: { test_loss} ', file = text_file)

average_test_error=test_accuracies_sum/ensemble_size
saving_utils.save_average_acc(average_test_error,dataset,depth,W,input_dim,learning_rate,weight_decay,square_edge,signal_noise_ratio,teacher_width,optimizer_choice,n_epochs)
saving_utils.save_accuracy_inf(accuracy_inf,dataset,depth,W,input_dim,learning_rate,weight_decay,square_edge,signal_noise_ratio,teacher_width,optimizer_choice,n_epochs)


#delta_error=(test_accuracies_sum-ensemble_size*accuracy_inf2)/ensemble_size
delta_error=(ensemble_size*accuracy_inf-test_accuracies_sum)/ensemble_size

saving_utils.save_delta_error(delta_error,dataset,depth,W,input_dim,learning_rate,weight_decay,optimizer_choice,n_epochs,signal_noise_ratio,teacher_width,square_edge)


saving_utils.save_train_loss(train_counter, train_losses,dataset,depth,W,input_dim,optimizer_choice,n_epochs,signal_noise_ratio,teacher_width,square_edge)


text_file.close()

print(f'End of training of a net with the following args: dataset {dataset} width {W} depth {depth} input_dim {input_dim} '
      f'n_epochs {n_epochs} number_nets {ensemble_size} learning_rate {learning_rate} weight_decay {weight_decay} '
      f'size_of_training_set {sample_size} batch_size {batch_size} signal_noise_ratio {signal_noise_ratio} square_edge {square_edge} teacher_width {teacher_width}'
      f'optimizer_choice {optimizer_choice}')

