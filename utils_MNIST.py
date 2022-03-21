import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


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
    train_dataset=pMNISTDataSet(x_train,y_train,final_dim)
    test_dataset = pMNISTDataSet(x_test, y_test, final_dim)
    torch.save(train_dataset, f'./data/MNIST_PCA_{final_dim}_train')
    torch.save(test_dataset, f'./data/MNIST_PCA_{final_dim}_test')


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
    return post_PCA_train,post_PCA_test,y_train,y_test

class pMNISTDataSet(torch.utils.data.Dataset):
    # images df, labels df, transforms
    # uses labels to determine if it needs to return X & y or just X in __getitem__
    def __init__(self, images, labels, dimensions,transforms=None):
        self.X = images
        self.y = labels
        self.dimensions=dimensions
        self.transforms = transforms

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        data = self.X.iloc[i, :]  # gets the row
        # reshape the row into the image size
        # (numpy arrays have the color channels dim last)
        data = np.array(data).astype(np.uint8).reshape(28, 28, 1)

        # perform transforms if there are any
        if self.transforms:
            data = self.transforms(data)

        # if !test_set return the label as well, otherwise don't
        if self.y is not None:  # train/val
            return (data, self.y[i])
        else:  # test
            return data


class Net5(nn.Module):
    W = 512
    #so init defines the "architecture" of the net, while forward defines the algorithm
    #the backward algorithm is automatically computed using the forward algorithm
    def __init__(self):
        W=512
        super(Net5, self).__init__()
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(28*28, W)  # 5*5 from image dimension
        self.fc2 = nn.Linear(W,W)
        self.fc3 = nn.Linear(W,W)
        self.fc4 = nn.Linear(W,W)
        #linear sarebbe la classica operazione, in questo caso con 120 neuroni nel layer?
        self.fc5 = nn.Linear(W, 1)

    def forward(self, x):
        x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x

class Net4(nn.Module):
    #so init defines the "architecture" of the net, while forward defines the algorithm
    #the backward algorithm is automatically computed using the forward algorithm
    def __init__(self):
        W = 512
        super(Net4, self).__init__()
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(28*28, W)  # 5*5 from image dimension
        self.fc2 = nn.Linear(W,W)
        self.fc3 = nn.Linear(W,W)
        #linear sarebbe la classica operazione, in questo caso con 120 neuroni nel layer?
        self.fc4 = nn.Linear(W, 1)

    def forward(self, x):
        x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class Net3(nn.Module):
    #so init defines the "architecture" of the net, while forward defines the algorithm
    #the backward algorithm is automatically computed using the forward algorithm
    def __init__(self):
        W = 1024
        super(Net3, self).__init__()
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(28*28, W)  # 5*5 from image dimension
        self.fc2 = nn.Linear(W,W)
        #linear sarebbe la classica operazione, in questo caso con 120 neuroni nel layer?
        self.fc3 = nn.Linear(W, 1)

    def forward(self, x):
        x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Net2(nn.Module):
    #so init defines the "architecture" of the net, while forward defines the algorithm
    #the backward algorithm is automatically computed using the forward algorithm
    def __init__(self):
        W = 2048
        super(Net2, self).__init__()
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(28*28, W)  # 5*5 from image dimension
        #linear sarebbe la classica operazione, in questo caso con 120 neuroni nel layer?
        self.fc2 = nn.Linear(W, 1)

    def forward(self, x):
        x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class Net10(nn.Module):
    #so init defines the "architecture" of the net, while forward defines the algorithm
    #the backward algorithm is automatically computed using the forward algorithm
    def __init__(self):
        W=256
        super(Net10, self).__init__()
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(28*28, W)  # 5*5 from image dimension
        self.fc2 = nn.Linear(W,W)
        self.fc3 = nn.Linear(W,W)
        self.fc4 = nn.Linear(W,W)
        self.fc5 = nn.Linear(W,W)
        self.fc6 = nn.Linear(W,W)
        self.fc7 = nn.Linear(W,W)
        self.fc8 = nn.Linear(W,W)
        self.fc9 = nn.Linear(W, W)
        #linear sarebbe la classica operazione, in questo caso con 120 neuroni nel layer?
        self.fc10 = nn.Linear(W, 1)

    def forward(self, x):
        x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        x = F.relu(self.fc8(x))
        x = F.relu(self.fc9(x))
        x = self.fc10(x)
        return x

class Net7(nn.Module):
    W = 512
    #so init defines the "architecture" of the net, while forward defines the algorithm
    #the backward algorithm is automatically computed using the forward algorithm
    def __init__(self):
        W=512
        super(Net7, self).__init__()
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(28*28, W)  # 5*5 from image dimension
        self.fc2 = nn.Linear(W,W)
        self.fc3 = nn.Linear(W,W)
        self.fc4 = nn.Linear(W,W)
        self.fc5 = nn.Linear(W,W)
        self.fc6 = nn.Linear(W,W)
        #linear sarebbe la classica operazione, in questo caso con 120 neuroni nel layer?
        self.fc7 = nn.Linear(W, 1)

    def forward(self, x):
        x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = self.fc7(x)
        return x