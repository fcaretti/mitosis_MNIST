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
            dropout = False, batch_norm = False,gaussian_init=False, p =0.5,gaussian_std=0.5):
        super(fully_connected_new, self).__init__()

        if isinstance(widths, int):
            widths = np.array([input_size]+[widths for i in range(depth -1)], dtype = 'int')
        elif isinstance(widths, list):
            widths = np.array([input_size]+widths, dtype = 'int')
        else: raise TypeError('expected type int or list of variable widths')
        self.linear_body = make_layers_new(widths, batch_norm, dropout, p=p)
        self.linear_out = nn.Linear(widths[-1], output_size)
        if orthog_init: self.orthog_init()
        if gaussian_init:
            self.gaussian_std=gaussian_std
            self.gaussian_init()
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                #nn.init.constant_(m.weight, 1) in resnet: (and by default!!!!)
                #incredible increase in performance if m.weight is set to random uniform 0,1 from constant 1
                if gaussian_init==True:
                    nn.init.normal_(m.weight)
                    nn.init.normal_(m.bias)
                else:
                    nn.init.uniform_(m.weight)
                    nn.init.zeros_(m.bias)
                #nn.init.zeros_(m.running_mean)
                #nn.init.ones_(m.running_var)
            # elif isinstance(m, nn.Linear):
            #     nn.init.kaiming_normal_(m.weight) #by default is kaiming_uniform
                #nn.init.zeros_(m.bias) #by default is uniform(fan_in**0.5)

    def orthog_init(self):
        self.apply(self.init_weights_orthog)

    def gaussian_init(self):
        self.apply(self.init_weights_normal)

    def init_weights_normal(self,m):
        if type(m)==nn.Linear:
            nn.init.normal_(m.weight,0,self.gaussian_std)
            nn.init.normal_(m.bias,0,self.gaussian_std)

    def init_weights_orthog(self, m):
        if type(m) == nn.Linear:
            nn.init.orthogonal_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        x = torch.flatten(x, 1)
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
    train_dataset=CustomDataSet(x_train,y_train,final_dim)
    test_dataset = CustomDataSet(x_test, y_test, final_dim)
    #train_dataset=pMNISTDataSet(x_train,y_train,final_dim,'binary')
    #test_dataset = pMNISTDataSet(x_test, y_test, final_dim,'binary')
    torch.save(train_dataset, f'./data/MNIST_PCA_{final_dim}_train.pt')
    torch.save(test_dataset, f'./data/MNIST_PCA_{final_dim}_test.pt')



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

class CustomDataSet(torch.utils.data.Dataset):
    # images df, labels df, transforms
    # uses labels to determine if it needs to return X & y or just X in __getitem__
    def __init__(self, images, labels, dimension,transforms=None):
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

def create_XOR_dataset(trainset_size, testset_size, input_dim,signal_noise_ratio):
    dataset_size = trainset_size+testset_size
    matrix_random = torch.normal(0, 1, (input_dim - 2, dataset_size))
    matrix_model_1 = torch.normal(0.5, 0.5/signal_noise_ratio, (2, round(dataset_size / 4)))
    matrix_model_2 = torch.normal(-0.5, 0.5/signal_noise_ratio, (2, round(dataset_size / 4)))
    means_3 = torch.tensor(np.array([[0.5], [-0.5]])).repeat(1, round(dataset_size / 4))
    means_4 = torch.tensor(np.array([[-0.5], [0.5]])).repeat(1, round(dataset_size / 4))
    matrix_model_3 = torch.normal(means_3, 0.5/signal_noise_ratio)
    matrix_model_4 = torch.normal(means_4, 0.5/signal_noise_ratio)
    big_matrix = torch.cat((matrix_model_1, matrix_model_2, matrix_model_3, matrix_model_4), 1)
    big_matrix = torch.cat((matrix_random, big_matrix), 0)
    big_matrix = big_matrix[torch.randperm(input_dim)]
    big_matrix = torch.transpose(big_matrix, 0, 1)
    labels = torch.cat((torch.zeros(round(dataset_size / 2)), torch.ones(round(dataset_size / 2))), 0)
    indexes=torch.randperm(dataset_size)
    labels_dataset=labels[indexes]
    X_dataset=big_matrix[indexes]
    train_set = CustomDataSet(X_dataset[:trainset_size], labels_dataset[:trainset_size], input_dim)
    test_set = CustomDataSet(X_dataset[trainset_size:], labels_dataset[trainset_size:], input_dim)
    torch.save(train_set, f'./data/XOR_{input_dim}_dimension_{signal_noise_ratio}_ratio_train_{trainset_size}_samples.pt')
    torch.save(test_set, f'./data/XOR_{input_dim}_dimension_{signal_noise_ratio}_ratio_test_{testset_size}_samples.pt')


def create_teacher_dataset(trainset_size, testset_size, input_dim, teacher_width, noise_ratio):
    depth=2
    sample_size=trainset_size+testset_size
    with torch.no_grad():
        teacher=fully_connected_new(teacher_width, depth, input_dim, output_size=1,
                                    dropout=False, batch_norm=False, orthog_init=False, gaussian_init=True)
        inputs=torch.normal(0,1,(sample_size,round(input_dim)))
        outputs=teacher(inputs)
        mean=torch.mean(outputs).item()
        std=torch.std(outputs).item()
        labels=torch.normal(0,(std/noise_ratio),(sample_size,1))+outputs
        labels=labels-mean
        labels[labels<0]=0
        labels[labels>0]=1
        labels=torch.squeeze(labels)
    train_set = CustomDataSet(inputs[:trainset_size], labels[:trainset_size], input_dim)
    test_set = CustomDataSet(inputs[trainset_size:], labels[trainset_size:], input_dim)
    torch.save(train_set, f'./data/teacher_{input_dim}_dimension_teacherwidth_{teacher_width}_{noise_ratio}_ratio_train_{trainset_size}_samples.pt')
    torch.save(test_set, f'./data/teacher_{input_dim}_dimension_teacherwidth_{teacher_width}_{noise_ratio}_ratio_test_{10000}_samples.pt')



'''def PCA(trainset, testset, init_dim, final_dim):
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
    return trainset_tuple,testset_tuple'''