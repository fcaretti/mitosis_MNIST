import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import Tensor


class chunked_net(nn.Module):
    def __init__(self, widths, chunk_size, depth, input_size, output_size=1, batch_norm=False, dropout=False, p=0):
        super(chunked_net, self).__init__()
        # an affine operation: y = Wx + b
        if isinstance(widths, int):
            widths = np.array([input_size]+[widths for i in range(depth -2)]+[chunk_size], dtype = 'int')
        elif isinstance(widths, list):
            widths = np.array([input_size]+widths[:-1]+[chunk_size], dtype = 'int')
        else: raise TypeError('expected type int or list of variable widths')

        self.linear_body = make_layers_new(widths, batch_norm, dropout, p=p)
        self.linear_out = nn.Linear(widths[-1], output_size)

    def forward(self, x):
        x = torch.flatten(x, 1)  # flatten all dimensions except the batch dimension
        x = self.linear_body(x)
        x = self.linear_out(x)
        return x

class fully_connected_new(nn.Module):

    def __init__(self, widths, depth = 6, input_size = 784, output_size = 1, orthog_init = False,
            dropout = False, batch_norm = False, p =0.5):
        super(fully_connected_new, self).__init__()

        if isinstance(widths, int):
            widths = np.array([input_size]+[widths for i in range(depth -1)], dtype = 'int')
        elif isinstance(widths, list):
            widths = np.array([input_size]+widths, dtype = 'int')
        else: raise TypeError('expected type int or list of variable widths')
        self.linear_body = make_layers_new(widths, batch_norm, dropout, p=p)
        self.linear_out = nn.Linear(widths[-1], output_size)
        if orthog_init: self.orthog_init()

        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                #nn.init.constant_(m.weight, 1) in resnet: (and by default!!!!)
                #incredible increase in performance if m.weight is set to random uniform 0,1 from constant 1
                nn.init.uniform_(m.weight)
                nn.init.zeros_(m.bias)
                #nn.init.zeros_(m.running_mean)
                #nn.init.ones_(m.running_var)
            # elif isinstance(m, nn.Linear):
            #     nn.init.kaiming_normal_(m.weight) #by default is kaiming_uniform
                #nn.init.zeros_(m.bias) #by default is uniform(fan_in**0.5)

    def orthog_init(self):
        self.apply(self.init_weights)

    def init_weights(self, m):
        if type(m) == nn.Linear:
            nn.init.orthogonal_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        #print(len(x[0][0][0]))
        x = torch.flatten(x, 1)
        #print(len(x))
        x = self.linear_body(x)
        x = self.linear_out(x)
        return x



def make_layers_new(widths, batch_norm, dropout, p):

    layers = []

    if dropout and batch_norm:
        for i in range(len(widths)-1):
            layers.extend([nn.Linear(widths[i], widths[i+1]), nn.BatchNorm1d(widths[i+1]), nn.ReLU(inplace = True), nn.Dropout(p=p)])

    elif dropout and not batch_norm:
        for i in range(len(widths)-1):
            layers.extend([nn.Linear(widths[i], widths[i+1]), nn.ReLU(inplace = True), nn.Dropout(p = p)])

    elif batch_norm and not dropout:
        for i in range(len(widths)-1):
            layers.extend([nn.Linear(widths[i], widths[i+1]), nn.BatchNorm1d(widths[i+1]), nn.ReLU(inplace = True) ])
    else:
        for i in range(len(widths)-1):
            layers.extend([nn.Linear(widths[i], widths[i+1]), nn.ReLU(inplace = True)])

    return nn.Sequential(*layers)

def gen(x):
   i = 2
   for n in range(x + 1):
       yield i
       i <<= 1

def create_pMNIST_PCA_dataset(trainset, testset, init_dim, final_dim):
    x_train, x_test,y_train,y_test= PCA_for_dataset(trainset, testset, init_dim, final_dim)
    train_dataset=pMNISTDataSet(x_train,y_train,final_dim,'binary')
    test_dataset = pMNISTDataSet(x_test, y_test, final_dim,'binary')
    torch.save(train_dataset, f'./data/MNIST_PCA_{final_dim}_train.pt')
    torch.save(test_dataset, f'./data/MNIST_PCA_{final_dim}_test.pt')


def PCA(trainset, testset, init_dim, final_dim):
    if final_dim>=init_dim:
        return trainset, testset
    trainset = list(trainset)
    testset=list(testset)
    x=[]
    y_train=[]
    y_test=[]
    for i in trainset:
        x.append(torch.transpose(torch.unsqueeze(torch.flatten(torch.squeeze(i[0])), 1), 0, 1))
        y_train.append(i[1])
    for i in testset:
        x.append(torch.transpose(torch.unsqueeze(torch.flatten(torch.squeeze(i[0])), 1), 0, 1))
        y_test.append(i[1])
    b = torch.Tensor(len(trainset)+len(testset), init_dim)
    torch.cat(x, dim=0, out=b)
    mean=torch.mean(b,0)
    b = b - mean
    max_dim=np.round(1.5*final_dim)
    if max_dim>final_dim:
        max_dim=final_dim
    U, S, V = torch.pca_lowrank(b, q=max_dim, center=False, niter=4)
    post_PCA_tensor=torch.matmul(b,V[:, :(final_dim)])
    post_PCA_train,post_PCA_test=torch.split(post_PCA_tensor,[len(trainset),len(testset)])
    trainset_list=[]
    testset_list=[]
    for i in range(len(trainset)):
        trainset_list.append([post_PCA_train[i], y_train[i]])
    for i in range(len(testset)):
        testset_list.append([post_PCA_test[i],y_test[i]])
    trainset_tuple=tuple(trainset_list)
    testset_tuple = tuple(testset_list)
    return trainset_tuple,testset_tuple

def PCA_for_dataset(trainset, testset, init_dim, final_dim):
    if final_dim>=init_dim:
        return trainset, testset
    trainset = list(trainset)
    testset=list(testset)
    x=[]
    y_train=[]
    y_test=[]
    for i in trainset:
        x.append(torch.transpose(torch.unsqueeze(torch.flatten(torch.squeeze(i[0])), 1), 0, 1))
        y_train.append(i[1])
    for i in testset:
        x.append(torch.transpose(torch.unsqueeze(torch.flatten(torch.squeeze(i[0])), 1), 0, 1))
        y_test.append(i[1])
    b = torch.Tensor(len(trainset)+len(testset), init_dim)
    torch.cat(x, dim=0, out=b)
    mean=torch.mean(b,0)
    b = b - mean
    max_dim=np.round(1.5*final_dim)
    if max_dim>final_dim:
        max_dim=final_dim
    U, S, V = torch.pca_lowrank(b, q=max_dim, center=False, niter=4)
    post_PCA_tensor=torch.matmul(b,V[:, :(final_dim)])
    post_PCA_train,post_PCA_test=torch.split(post_PCA_tensor,[len(trainset),len(testset)])
    return post_PCA_train,post_PCA_test,np.asarray(y_train),np.asarray(y_test)

class pMNISTDataSet(torch.utils.data.Dataset):
    # images df, labels df, transforms
    # uses labels to determine if it needs to return X & y or just X in __getitem__
    def __init__(self, images, labels, dimension,classification='multi',transforms=None):
        self.X = images
        self.y = labels
        self.dimension=dimension
        self.transforms = transforms

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        #data = self.X.iloc[i, :]  # gets the row
        # reshape the row into the image size
        # (numpy arrays have the color channels dim last)
        #data = np.array(data).astype(np.uint8).reshape(dimension, 1)
        image=self.X[idx]
        # perform transforms if there are any
        if self.transforms:
            image = self.transforms(image)
        # if !test_set return the label as well, otherwise don't
        if self.y is not None:  # train/val
            return (image, self.y[idx])
        else:  # test
            return data

class make_binary(torch.nn.Module):
    def __init__(self, inplace=False):
        super().__init__()
        self.inplace = inplace
    def forward(self, tensor: Tensor) -> Tensor:
        return tensor%2

