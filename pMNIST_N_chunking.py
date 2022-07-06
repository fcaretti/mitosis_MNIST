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
import utils_MNIST
import saving_utils

#example XOR: python pMNIST_N_chunking.py --dataset XOR --W 64 --depth 3 --input_dim 20 --ensemble_size 5 --learning_rate 1e-3 --weight_decay 1e-3 --N_samples 10 --signal_noise_ratio 1
#example generalized_XOR: python pMNIST_N_chunking.py --dataset generalized_XOR --W 256 --depth 3 --input_dim 20 --ensemble_size 10 --learning_rate 1e-2 --weight_decay 5e-4 --N_samples 10 --signal_noise_ratio 1.
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", help="which dataset to use, pMNIST or XOR? (or maybe other in the future")
parser.add_argument("--W", help="width of the network to train",
                    type=int)
parser.add_argument("--depth", help= "depth of the network to train",
                    type=int)
parser.add_argument("--input_dim", help="size of the input, modified by PCA", type=int, default=784)
parser.add_argument("--ensemble_size", help="how many networks to average the chunks over", type=int, default=5)
parser.add_argument("--learning_rate", help="learning rate used for the network used during training", type=float, default=1e-3)
parser.add_argument("--weight_decay", help="weight decay used for the network used during training", type=float, default= 1e-3)
parser.add_argument("--N_samples", help="number of samples for each chunk size and for each network", type=int, default=10)
parser.add_argument("--n_epochs", help="number of max epochs of training", type=int, default= 500)
parser.add_argument("--signal_noise_ratio", help="only useful with artificial data", type=float, default=1.)
parser.add_argument("--teacher_width", help="only useful in teacher-student setup", type=int, default=4)
parser.add_argument("--square_edge", help="only for generalized XOR", type=int, default=3)
parser.add_argument("--optimizer_choice", help="choice between Adam and SGD", default='SGD')
parser.add_argument("--sample_size", help="size of the training set, 60000 max", type=int, default=10000)

args = parser.parse_args()

dataset=args.dataset
W=args.W
depth=args.depth
input_dim=args.input_dim
number_nets=args.ensemble_size
learning_rate=args.learning_rate
weight_decay=args.weight_decay
N_samples=args.N_samples
signal_noise_ratio=args.signal_noise_ratio
teacher_width=args.teacher_width
square_edge=args.square_edge
optimizer_choice=args.optimizer_choice
n_epochs=args.n_epochs
sample_size=args.sample_size


if n_epochs==500:
    if dataset=='pMNIST':
        text_file = open(f'./logs/MNIST_{depth}_layer_{W}_chunks_lr_{learning_rate}_wd_{weight_decay}_{input_dim}_{optimizer_choice}_inputdim.txt', 'w')
    if dataset=='XOR':
        text_file = open(f'./logs/{dataset}_{signal_noise_ratio}_ratio_{depth}_layer_{W}_lr_'
                         f'{learning_rate}_wd_{weight_decay}_{input_dim}_{optimizer_choice}_inputdim.txt', 'w')
    if dataset=='teacher':
        text_file = open(f'./logs/{dataset}_{teacher_width}_teacherwidth_{signal_noise_ratio}_ratio_'
                         f'{depth}_layer_{W}_lr_{learning_rate}_wd_{weight_decay}_{input_dim}_{optimizer_choice}_inputdim.txt', 'w')
    if dataset=='generalized_XOR':
        text_file = open(f'./logs/{dataset}_edge_{square_edge}_{signal_noise_ratio}_ratio_{depth}_layer_{W}_lr_'
                         f'{learning_rate}_wd_{weight_decay}_{input_dim}_{optimizer_choice}_inputdim.txt', 'w')
else:
    if dataset=='pMNIST':
        text_file = open(f'./logs/MNIST_{depth}_layer_{W}_chunks_lr_{learning_rate}_wd_{weight_decay}_{input_dim}_{optimizer_choice}_{n_epochs}.txt', 'w')
    if dataset=='XOR':
        text_file = open(f'./logs/{dataset}_{signal_noise_ratio}_ratio_{depth}_layer_{W}_lr_'
                         f'{learning_rate}_wd_{weight_decay}_{input_dim}_{optimizer_choice}_{n_epochs}.txt', 'w')
    if dataset=='teacher':
        text_file = open(f'./logs/{dataset}_{teacher_width}_teacherwidth_{signal_noise_ratio}_ratio_'
                         f'{depth}_layer_{W}_lr_{learning_rate}_wd_{weight_decay}_{input_dim}_{optimizer_choice}_{n_epochs}.txt', 'w')
    if dataset=='generalized_XOR':
        text_file = open(f'./logs/{dataset}_edge_{square_edge}_{signal_noise_ratio}_ratio_{depth}_layer_{W}_lr_'
                         f'{learning_rate}_wd_{weight_decay}_{input_dim}_{optimizer_choice}_{n_epochs}.txt', 'w')


criterion = nn.BCEWithLogitsLoss()
criterion2= nn.BCELoss()

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize(0.5, 0.5)])

trainset,testset=saving_utils.dataset_loader('test',dataset, transform, input_dim, sample_size,signal_noise_ratio,teacher_width, square_edge)

trainloader2 = torch.utils.data.DataLoader(trainset, batch_size=sample_size,
                                           shuffle=False, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=len(testset),
                                         shuffle=False, num_workers=2)

big_net = utils_MNIST.fully_connected_new(W, depth=depth, input_size=input_dim, output_size=1,
                                dropout=False, batch_norm=False, orthog_init=False)


accuracy_inf=saving_utils.load_accuracy_inf(dataset,depth,W,input_dim,learning_rate,weight_decay,optimizer_choice,n_epochs,signal_noise_ratio,teacher_width,square_edge)

#******I take N samples for each chunk size************************

#*******************Array of chunk sizes*************************

sizes=list(utils_MNIST.gen(int(math.log(W,2))-1))

sizes.append(sizes[-1]-sizes[-3])
sizes.append(sizes[-3]-sizes[-5])
if W<=512 and W>=64:
    sizes.append(sizes[-5] - sizes[-7])
    sizes.append(sizes[-7] - sizes[-9])

sizes.sort()
sizes.append(round(sizes[-2]*0.9))
sizes.sort()
losses=[]
accuracies=[]
acc_errors=[]
train_accs=[]
train_acc_errors=[]

#***Iterate on chunk sizes, and take average of accuracies and losses for each chunk size***
for chunk_size in sizes:
    mean_loss=0
    accuracy_per_size=[]
    train_acc_per_size=[]
    for i in range(0,number_nets):
        PATH=saving_utils.load_weights(dataset,depth,W,i,learning_rate,weight_decay,input_dim,optimizer_choice,n_epochs,signal_noise_ratio,teacher_width,square_edge)

        weights_dict = torch.load(PATH)
        original_tensor=weights_dict['linear_out.weight']
        for k in range(0, N_samples):
            mask=torch.cat((torch.ones(W-chunk_size),torch.zeros(chunk_size)),0)
            mask = mask.view(-1)[torch.randperm(W)].view(mask.size())
            mask=torch.t(mask.ge(0.5))
            big_net.load_state_dict(weights_dict)
            weights_dict['linear_out.weight']=original_tensor.masked_fill(mask,0)
            train_correct=0
            train_total = 0
            correct = 0
            total = 0
            chunk_loss=0
            # since we're not training, we don't need to calculate the gradients for our outputs
            accuracy,chunk_loss,_=utils_MNIST.evaluate_on_dataset(dataset, testloader, big_net, criterion)
            print(f'Chunk Size: {chunk_size} Accuracy of the network on the 10000 test images: {accuracy} %', file = text_file)
            print(f'Chunk Size: {chunk_size} Loss of the network on the 10000 test images: {chunk_loss} ', file = text_file)
            train_accuracy,_,_=utils_MNIST.evaluate_on_dataset(dataset, trainloader2, big_net, criterion)
            print(f'Chunk Size: {chunk_size} Accuracy of the network on the 10000 train images: {train_accuracy} %', file = text_file)
            mean_loss+=chunk_loss
            accuracy_per_size.append(accuracy)
            train_acc_per_size.append(train_accuracy)
    train_mean=sum(train_acc_per_size)/len(train_acc_per_size)
    mean=sum(accuracy_per_size)/len(accuracy_per_size)
    train_acc_errors.append(math.sqrt(sum([(number-train_mean) ** 2 for number in train_acc_per_size]))/(N_samples*number_nets-1))
    acc_errors.append(math.sqrt(sum([(number-mean) ** 2 for number in accuracy_per_size]))/(N_samples*number_nets-1))
    accuracies.append(mean)
    train_accs.append(train_mean)
    losses.append(mean_loss/N_samples/number_nets)
    print(f'Accuracy at chunk size= {chunk_size}: {mean}')
    print(f'Accuracy at chunk size on train set= {chunk_size}: {train_mean}')



fig = plt.figure()
ax = plt.gca()
arr1 = np.array(sizes)
arr5 = np.array(accuracies)
acc_errors=np.array(acc_errors)
train_accs=np.array(train_accs)
train_acc_errors=np.array(train_acc_errors)
arr5=accuracy_inf-arr5

while arr5[-1]<0 or arr5[-1]==0 or arr5[-1]<(1/len(testset)):
    arr5=arr5[:-1]
    arr1=arr1[:-1]
    acc_errors=acc_errors[:-1]

'''if n_epochs==500:
    if dataset=='pMNIST':
        with open(f'./arrays/sizes_{depth}_layer_{W}_inputdim_{input_dim}_lr_{learning_rate}_wd_{weight_decay}_{optimizer_choice}.npy', 'wb') as f:
            np.save(f, arr1)
        with open(f'./arrays/error_{depth}_layer_{W}_inputdim_{input_dim}_lr_{learning_rate}_wd_{weight_decay}_{optimizer_choice}.npy', 'wb') as f:
            np.save(f, arr5)
        with open(f'./arrays/error_on_error_{depth}_layer_{W}_inputdim_{input_dim}_lr_{learning_rate}_wd_{weight_decay}_{optimizer_choice}.npy', 'wb') as f:
            np.save(f, acc_errors)
        with open(f'./arrays/train_error_{depth}_layer_{W}_inputdim_{input_dim}_lr_{learning_rate}_wd_{weight_decay}_{optimizer_choice}.npy', 'wb') as f:
            np.save(f, train_accs)
        with open(f'./arrays/train_acc_error_{depth}_layer_{W}_inputdim_{input_dim}_lr_{learning_rate}_wd_{weight_decay}_{optimizer_choice}.npy', 'wb') as f:
            np.save(f, train_acc_errors)
    if dataset=='XOR':
        with open(f'./arrays/sizes_{dataset}_{depth}_layer_{W}_inputdim_{input_dim}_lr_{learning_rate}_wd_{weight_decay}_'
                  f'ratio_{signal_noise_ratio}_{optimizer_choice}.npy', 'wb') as f:
            np.save(f, arr1)
        with open(f'./arrays/error_{dataset}_{depth}_layer_{W}_inputdim_{input_dim}_lr_{learning_rate}_wd_{weight_decay}_'
                  f'ratio_{signal_noise_ratio}_{optimizer_choice}.npy', 'wb') as f:
            np.save(f, arr5)
        with open(f'./arrays/error_on_error_{dataset}_{depth}_layer_{W}_inputdim_{input_dim}_lr_{learning_rate}_wd_{weight_decay}_'
                  f'ratio_{signal_noise_ratio}_{optimizer_choice}.npy', 'wb') as f:
            np.save(f, acc_errors)
        with open(f'./arrays/train_error_{dataset}_{depth}_layer_{W}_inputdim_{input_dim}_lr_{learning_rate}_wd_{weight_decay}_'
                  f'ratio_{signal_noise_ratio}_{optimizer_choice}.npy', 'wb') as f:
            np.save(f, train_accs)
        with open(f'./arrays/train_acc_error_{dataset}_{depth}_layer_{W}_inputdim_{input_dim}_lr_{learning_rate}_wd_{weight_decay}_'
                  f'ratio_{signal_noise_ratio}_{optimizer_choice}.npy', 'wb') as f:
            np.save(f, train_acc_errors)
    if dataset=='teacher':
        with open(f'./arrays/sizes_{dataset}_{depth}_layer_{W}_inputdim_{input_dim}_lr_{learning_rate}_wd_{weight_decay}_'
                  f'teacherwidth_{teacher_width}_ratio_{signal_noise_ratio}_{optimizer_choice}.npy', 'wb') as f:
            np.save(f, arr1)
        with open(f'./arrays/error_{dataset}_{depth}_layer_{W}_inputdim_{input_dim}_lr_{learning_rate}_wd_{weight_decay}_'
                  f'teacherwidth_{teacher_width}_ratio_{signal_noise_ratio}_{optimizer_choice}.npy', 'wb') as f:
            np.save(f, arr5)
        with open(f'./arrays/error_on_error_{dataset}_{depth}_layer_{W}_inputdim_{input_dim}_lr_{learning_rate}_wd_'
                  f'{weight_decay}_{optimizer_choice}.npy', 'wb') as f:
            np.save(f, acc_errors)
        with open(f'./arrays/train_error_{dataset}_{depth}_layer_{W}_inputdim_{input_dim}_lr_{learning_rate}_wd_'
                  f'{weight_decay}_{optimizer_choice}.npy', 'wb') as f:
            np.save(f, train_accs)
        with open(f'./arrays/train_acc_error_{dataset}_{depth}_layer_{W}_inputdim_{input_dim}_lr_{learning_rate}_wd_'
                  f'{weight_decay}_{optimizer_choice}.npy', 'wb') as f:
            np.save(f, train_acc_errors)
    if dataset=='generalized_XOR':
        with open(f'./arrays/sizes_{dataset}_{depth}_layer_{W}_inputdim_{input_dim}_lr_{learning_rate}_wd_{weight_decay}_'
                  f'edge_{square_edge}_ratio_{signal_noise_ratio}_{optimizer_choice}.npy', 'wb') as f:
            np.save(f, arr1)
        with open(f'./arrays/error_{dataset}_{depth}_layer_{W}_inputdim_{input_dim}_lr_{learning_rate}_wd_{weight_decay}_'
                  f'edge_{square_edge}_ratio_{signal_noise_ratio}_{optimizer_choice}.npy', 'wb') as f:
            np.save(f, arr5)
        with open(f'./arrays/error_on_error_{dataset}_{depth}_layer_{W}_inputdim_{input_dim}_lr_{learning_rate}_wd_{weight_decay}_'
                  f'edge_{square_edge}_ratio_{signal_noise_ratio}_{optimizer_choice}.npy', 'wb') as f:
            np.save(f, acc_errors)
        with open(f'./arrays/train_error_{dataset}_{depth}_layer_{W}_inputdim_{input_dim}_lr_{learning_rate}_wd_{weight_decay}_'
                  f'edge_{square_edge}_ratio_{signal_noise_ratio}_{optimizer_choice}.npy', 'wb') as f:
            np.save(f, train_accs)
        with open(f'./arrays/tran_acc_error_{dataset}_{depth}_layer_{W}_inputdim_{input_dim}_lr_{learning_rate}_wd_{weight_decay}_'
                  f'edge_{square_edge}_ratio_{signal_noise_ratio}_{optimizer_choice}.npy', 'wb') as f:
            np.save(f, train_acc_errors)
else:
    if dataset=='pMNIST':
        with open(f'./arrays/sizes_{depth}_layer_{W}_inputdim_{input_dim}_lr_{learning_rate}_wd_{weight_decay}_'
                  f'{optimizer_choice}_{n_epochs}.npy', 'wb') as f:
            np.save(f, arr1)
        with open(f'./arrays/error_{depth}_layer_{W}_inputdim_{input_dim}_lr_{learning_rate}_wd_{weight_decay}_'
                  f'{optimizer_choice}_{n_epochs}.npy', 'wb') as f:
            np.save(f, arr5)
        with open(f'./arrays/error_on_error_{depth}_layer_{W}_inputdim_{input_dim}_lr_{learning_rate}_wd_{weight_decay}_'
                  f'{optimizer_choice}_{n_epochs}.npy', 'wb') as f:
            np.save(f, acc_errors)
        with open(f'./arrays/train_error_{depth}_layer_{W}_inputdim_{input_dim}_lr_{learning_rate}_wd_{weight_decay}_'
                  f'{optimizer_choice}_{n_epochs}.npy', 'wb') as f:
            np.save(f, train_accs)
        with open(f'./arrays/train_acc_error_{depth}_layer_{W}_inputdim_{input_dim}_lr_{learning_rate}_wd_{weight_decay}_'
                  f'{optimizer_choice}_{n_epochs}.npy', 'wb') as f:
            np.save(f, train_acc_errors)
    if dataset=='XOR':
        with open(f'./arrays/sizes_{dataset}_{depth}_layer_{W}_inputdim_{input_dim}_lr_{learning_rate}_wd_{weight_decay}_'
                  f'ratio_{signal_noise_ratio}_{optimizer_choice}_{n_epochs}.npy', 'wb') as f:
            np.save(f, arr1)
        with open(f'./arrays/error_{dataset}_{depth}_layer_{W}_inputdim_{input_dim}_lr_{learning_rate}_wd_{weight_decay}_'
                  f'ratio_{signal_noise_ratio}_{optimizer_choice}_{n_epochs}.npy', 'wb') as f:
            np.save(f, arr5)
        with open(f'./arrays/error_on_error_{dataset}_{depth}_layer_{W}_inputdim_{input_dim}_lr_{learning_rate}_wd_{weight_decay}_'
                  f'ratio_{signal_noise_ratio}_{optimizer_choice}_{n_epochs}.npy', 'wb') as f:
            np.save(f, acc_errors)
        with open(f'./arrays/train_error_{dataset}_{depth}_layer_{W}_inputdim_{input_dim}_lr_{learning_rate}_wd_{weight_decay}_'
                  f'ratio_{signal_noise_ratio}_{optimizer_choice}_{n_epochs}.npy', 'wb') as f:
            np.save(f, train_accs)
        with open(f'./arrays/train_acc_error_{dataset}_{depth}_layer_{W}_inputdim_{input_dim}_lr_{learning_rate}_wd_{weight_decay}_'
                  f'ratio_{signal_noise_ratio}_{optimizer_choice}_{n_epochs}.npy', 'wb') as f:
            np.save(f, train_acc_errors)
    if dataset=='teacher':
        with open(f'./arrays/sizes_{dataset}_{depth}_layer_{W}_inputdim_{input_dim}_lr_{learning_rate}_wd_{weight_decay}_'
                  f'teacherwidth_{teacher_width}_ratio_{signal_noise_ratio}_{optimizer_choice}_{n_epochs}.npy', 'wb') as f:
            np.save(f, arr1)
        with open(f'./arrays/error_{dataset}_{depth}_layer_{W}_inputdim_{input_dim}_lr_{learning_rate}_wd_{weight_decay}_'
                  f'teacherwidth_{teacher_width}_ratio_{signal_noise_ratio}_{optimizer_choice}_{n_epochs}.npy', 'wb') as f:
            np.save(f, arr5)
        with open(f'./arrays/error_on_error_{dataset}_{depth}_layer_{W}_inputdim_{input_dim}_lr_{learning_rate}_wd_{weight_decay}_'
                  f'teacherwidth_{teacher_width}_ratio_{signal_noise_ratio}_{optimizer_choice}_{n_epochs}.npy', 'wb') as f:
            np.save(f, acc_errors)
        with open(f'./arrays/train_error_{dataset}_{depth}_layer_{W}_inputdim_{input_dim}_lr_{learning_rate}_wd_{weight_decay}_'
                  f'teacherwidth_{teacher_width}_ratio_{signal_noise_ratio}_{optimizer_choice}_{n_epochs}.npy', 'wb') as f:
            np.save(f, train_accs)
        with open(f'./arrays/train_acc_error_{dataset}_{depth}_layer_{W}_inputdim_{input_dim}_lr_{learning_rate}_wd_{weight_decay}_'
                  f'teacherwidth_{teacher_width}_ratio_{signal_noise_ratio}_{optimizer_choice}_{n_epochs}.npy', 'wb') as f:
            np.save(f, train_acc_errors)
    if dataset=='generalized_XOR':
        with open(f'./arrays/sizes_{dataset}_{depth}_layer_{W}_inputdim_{input_dim}_lr_{learning_rate}_wd_{weight_decay}_'
                  f'edge_{square_edge}_ratio_{signal_noise_ratio}_{optimizer_choice}_{n_epochs}.npy', 'wb') as f:
            np.save(f, arr1)
        with open(f'./arrays/error_{dataset}_{depth}_layer_{W}_inputdim_{input_dim}_lr_{learning_rate}_wd_{weight_decay}_'
                  f'edge_{square_edge}_ratio_{signal_noise_ratio}_{optimizer_choice}_{n_epochs}.npy', 'wb') as f:
            np.save(f, arr5)
        with open(f'./arrays/error_on_error_{dataset}_{depth}_layer_{W}_inputdim_{input_dim}_lr_{learning_rate}_wd_{weight_decay}_'
                  f'edge_{square_edge}_ratio_{signal_noise_ratio}_{optimizer_choice}_{n_epochs}.npy', 'wb') as f:
            np.save(f, acc_errors)
        with open(f'./arrays/train_error_{dataset}_{depth}_layer_{W}_inputdim_{input_dim}_lr_{learning_rate}_wd_{weight_decay}_'
                  f'edge_{square_edge}_ratio_{signal_noise_ratio}_{optimizer_choice}_{n_epochs}.npy', 'wb') as f:
            np.save(f, train_accs)
        with open(f'./arrays/train_acc_error_{dataset}_{depth}_layer_{W}_inputdim_{input_dim}_lr_{learning_rate}_wd_{weight_decay}_'
                  f'edge_{square_edge}_ratio_{signal_noise_ratio}_{optimizer_choice}_{n_epochs}.npy', 'wb') as f:
            np.save(f, train_acc_errors)'''
saving_utils.save_all_arrays(dataset,n_epochs,depth,W,input_dim,learning_rate,weight_decay,optimizer_choice,arr1,arr5,acc_errors,
                    train_accs,train_acc_errors,signal_noise_ratio,teacher_width,square_edge)


ax.errorbar(arr1, arr5,yerr=acc_errors,fmt="bo")
ax.set_xlim([2,1.2*W])
x = np.linspace(2,1.2*arr1[-1],1000)
ax.set_ylim([0.8*arr5[-1],100])
a=arr5[-1]/(arr1[-1]**(-0.5))
y=a*x**(-0.5)
ax.plot(x,y,label='w^(-0.5)')
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_title(f'FCN of depth {depth}, input dimension={input_dim} and wd={weight_decay}')
plt.xlabel('chunks width')
plt.ylabel('$\Delta error$')
if dataset=='pMNIST':
    plt.savefig(f'./plots/{dataset}_{depth}_layer_{W}_lr_{learning_rate}_wd_{weight_decay}_inputdim_{input_dim}_{optimizer_choice}_{n_epochs}_error.png')
if dataset=='XOR':
    plt.savefig(f'./plots/{dataset}_{depth}_layer_{W}_lr_{learning_rate}_wd_{weight_decay}_inputdim_{input_dim}_'
                f'ratio_{signal_noise_ratio}_{optimizer_choice}_{n_epochs}_error.png')
if dataset=='teacher':
    plt.savefig(f'./plots/{dataset}_{depth}_layer_{W}_lr_{learning_rate}_wd_{weight_decay}_inputdim_{input_dim}_'
                f'teacherwidth_{teacher_width}_ratio_{signal_noise_ratio}_{optimizer_choice}_{n_epochs}_error.png')
if dataset=='generalized_XOR':
    plt.savefig(f'./plots/{dataset}_{depth}_layer_{W}_lr_{learning_rate}_wd_{weight_decay}_inputdim_{input_dim}_'
                f'edge_{square_edge}_ratio_{signal_noise_ratio}_{optimizer_choice}_{n_epochs}_error.png')


#ax.errorbar(arr1, train_accs,yerr=train_acc_errors,fmt="bo")
saving_utils.save_train_acc(arr1, train_accs, train_acc_errors, dataset, depth, W, input_dim, weight_decay, learning_rate,
                            optimizer_choice, n_epochs, signal_noise_ratio, teacher_width, square_edge)









