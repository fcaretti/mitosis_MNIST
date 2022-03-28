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
from utils_MNIST import chunked_net, fully_connected_new, gen, PCA
#python pMNIST_N_chunking.py 512 5 10 5 1e-3 10

parser = argparse.ArgumentParser()
parser.add_argument("W", help="width of the network to train",
                    type=int)
parser.add_argument("depth", help= "depth of the network to train",
                    type=int)
parser.add_argument("input_dim", help="size of the input, modified by PCA", type=int, default=784)
parser.add_argument("ensemble_size", help="how many networks to average the chunks over", type=int, default=5)
parser.add_argument("weight_decay", help="weight decay used for the network used during training", type=float, default= 1e-3)
parser.add_argument("N_samples", help="number of samples for each chunk size and for each network", type=int, default=10)
args = parser.parse_args()


W=args.W
depth=args.depth
input_dim=args.input_dim
number_nets=args.ensemble_size
wd=args.weight_decay
N_samples=args.N_samples
text_file = open(f'./logs/MNIST_{depth}_layer_{W}_chunks_{wd}_wd_{input_dim}_inputdim_parsed.txt', 'w')
criterion = nn.BCEWithLogitsLoss()
criterion2= nn.BCELoss()



'''transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize(0.5, 0.5)])
trainset=torchvision.datasets.MNIST(root='./data', train=True,
                                     download=True, transform=transform)
testset = torchvision.datasets.MNIST(root='./data', train=False,
                                     download=True, transform=transform)
print(trainset[0])
if input_dim != 784:
    trainset,testset=PCA(trainset,testset,784,input_dim)
print(trainset[0])'''

if input_dim==784:
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(0.5, 0.5)])
    testset = torchvision.datasets.MNIST(root='./data', train=False,
                                         download=True, transform=transform)
if input_dim!=784:
    testset = torch.load(f'./data/testset_dim_{input_dim}.pt')
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
correct = 0
total = 0
test_loss=0
total_outputs=torch.zeros(len(testset))
total_maj_outputs=torch.zeros(len(testset))

for i in range(0,number_nets):
    print(f'Starting test of network # {i+1}')
    PATH = f'./nets/mnist_trained_{depth}_layer_{W}_net_{i + 1}_parsed_wd_{wd}_inputdim_{input_dim}.pth'
    weights_dict = torch.load(PATH)
    big_net = fully_connected_new(W, depth=depth, input_size=input_dim, output_size=1,
                                dropout=False, batch_norm=False, orthog_init=False)
    big_net.load_state_dict(weights_dict)
    correct = 0
    total = 0
    test_loss = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = big_net(images)
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
    print(f'Accuracy of the network on the 10000 test images: {100 * correct / total} %', file=text_file)
    print(f'Loss of the network on the 10000 test images: {test_loss} ', file=text_file)
    print(f'Finished test of network # {i+1}')
total_outputs_1=total_outputs/number_nets
total_maj_outputs_1=total_maj_outputs/float(number_nets)
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
accuracy_inf=accuracy_inf2

#******I take N samples for each chunk size************************

#*******************Array of chunk sizes*************************

sizes=list(gen(int(math.log(W,2))-1))

sizes.append(sizes[-1]-sizes[-3])
sizes.append(sizes[-3]-sizes[-5])

#for i in range(round(W/16+1),round(W/4-1)):
#    if (i%8==0 and i!=(W/8)):
#        sizes.append(i)
#print(sizes)
sizes.sort()
losses=[]
accuracies=[]
acc_errors=[]

#***Iterate on chunk sizes, and take average of accuracies and losses for each chunk size***
for chunk_size in sizes:
    small_net = chunked_net(W,chunk_size, depth,input_dim)
    #print(small_net)

    mean_loss=0
    accuracy_per_size=[]
    for i in range(0,number_nets):
        PATH = f'./nets/mnist_trained_{depth}_layer_{W}_net_{i+1}_parsed_wd_{wd}_inputdim_{input_dim}.pth'
        weights_dict = torch.load(PATH)
        old_weights_A = list(weights_dict.items())[-4][1]
        old_weights_B = list(weights_dict.items())[-3][1]
        old_weights_C = torch.transpose(list(weights_dict.items())[-2][1],0,1)
        for k in range(0, N_samples):
            #generator of batch_size numbers out of W
            random_positions=sorted((random.sample(range(W), chunk_size)))
            new_weights_A=torch.empty(chunk_size,W)
            if depth==2:
                new_weights_A = torch.empty(chunk_size, input_dim)
            new_weights_B=torch.empty(chunk_size)
            new_weights_C=torch.empty(chunk_size,1)
            j=0
            for i in random_positions:
                new_weights_A[j]=old_weights_A[i]
                new_weights_B[j]=old_weights_B[i]
                new_weights_C[j]=old_weights_C[i]
                j=j+1
            new_weights_C=torch.transpose(new_weights_C,0,1)
            #nw2=torch.Tensor(new_weights_2)
            weights_dict_1=OrderedDict()

            for i in range(len(list(weights_dict.items()))-4):
                weights_dict_1[list(weights_dict.items())[i][0]] = list(weights_dict.items())[i][1]
            weights_dict_1[list(weights_dict.items())[-4][0]]=new_weights_A
            weights_dict_1[list(weights_dict.items())[-3][0]]=new_weights_B
            weights_dict_1[list(weights_dict.items())[-2][0]]=new_weights_C
            weights_dict_1['linear_out.bias']=weights_dict['linear_out.bias']

            small_net.load_state_dict(weights_dict_1)
            correct = 0
            total = 0
            chunk_loss=0
            # since we're not training, we don't need to calculate the gradients for our outputs
            with torch.no_grad():
                for data in testloader:
                    images, labels = data
                    # calculate outputs by running images through the network
                    outputs = small_net(images)
                    # round to zero or one for the prediction
                    predicted=torch.transpose(torch.round(torch.sigmoid(outputs)),0,1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    outputs=torch.squeeze(outputs)
                    chunk_loss += criterion(outputs.to(torch.float32), labels.to(torch.float32))

            print(f'Chunk Size: {chunk_size} Accuracy of the network on the 10000 test images: {100 * correct / total} %', file = text_file)
            print(f'Chunk Size: {chunk_size} Loss of the network on the 10000 test images: {chunk_loss} ', file = text_file)
            mean_loss+=chunk_loss
            accuracy_per_size.append(100*correct/total)
    mean=sum(accuracy_per_size)/len(accuracy_per_size)
    acc_errors.append(math.sqrt(sum([(number-mean) ** 2 for number in accuracy_per_size]))/(N_samples*number_nets-1))
    accuracies.append(mean)
    losses.append(mean_loss/N_samples/number_nets)
    print(f'Finished chunk size = {chunk_size}')


fig = plt.figure()
ax = plt.gca()
arr1 = np.array(sizes)
arr5 = np.array(accuracies)
arr5=accuracy_inf-arr5
with open(f'./arrays/sizes_{depth}_layer_{W}_inputdim_{input_dim}_wd_{wd}.npy', 'wb') as f:
    np.save(f, arr1)
with open(f'./arrays/error_{depth}_layer_{W}_inputdim_{input_dim}_wd_{wd}.npy', 'wb') as f:
    np.save(f, arr5)
ax.errorbar(arr1, arr5,yerr=acc_errors,fmt="bo")
ax.set_xlim([2,1.2*W])
x = np.linspace(2,1.2*W,1000)
ax.set_ylim([0.8*arr5[-1],100])
a=arr5[-1]/(W**(-0.5))
y=a*x**(-0.5)
ax.plot(x,y,label='w^(-0.5)')
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_title(f'FCN{depth} of depth {depth}, input dimension={input_dim} and wd={wd}')
plt.xlabel('chunks width')
plt.ylabel('$\Delta error$')
plt.savefig(f'./plots/pMNIST_{depth}_layer_{W}_wd_{wd}_inputdim_{input_dim}_error.png')








'''text_file.close()
fig = plt.figure()
ax = plt.gca()
arr2 = np.array(losses)
test_loss=float(test_loss)
arr2+=(-test_loss)
ax.scatter(arr1, arr2, label='chunks loss')
x = np.linspace(1,1.5*W,1000)
a=arr2[-1]/(W**(-0.5))
y=a*x**(-0.5)
ax.plot(x,y,label='w^(-0.5)')
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_xlim([2,1.5*W])
ax.set_ylim([0.8*arr2[-1],1.2*arr2[0]])
plt.xlabel('chunks width')
plt.ylabel('cross-entropy loss - loss at inf')
ax.legend()
plt.savefig(f'./plots/pMNIST_{depth}_layer_{W}_loss.png')'''

'''fig = plt.figure()
ax = plt.gca()
arr1 = np.array(sizes)
arr3 = np.array(losses)
#test_loss=float(test_loss)
#arr2+=(-test_loss/total)
ax.scatter(arr1, arr3, label='chunks loss')
x = np.linspace(1,1.5*W,1000)
a=arr3[-1]/(W**(-0.5))
y=a*x**(-0.5)
ax.plot(x,y,label='w^(-0.5)')
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_xlim([2,1.5*W])
ax.set_ylim([0.8*arr3[-1],1.2*arr3[0]])
plt.xlabel('chunks width')
plt.ylabel('binary cross-entropy loss ')
ax.legend()
plt.savefig(f'./plots/pMNIST_{depth}_layer_{W}_not_norm_loss.png')
'''
'''fig = plt.figure()
ax = plt.gca()
arr4 = np.array(accuracies)
ax.scatter(arr1, arr4)
ax.set_xlim([2,1.2*W])
ax.set_ylim([0,100])
plt.xlabel('chunks width')
plt.ylabel('accuracy')
plt.savefig(f'./plots/pMNIST_{depth}_layer_{W}_acc.png')'''


