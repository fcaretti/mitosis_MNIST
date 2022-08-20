import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import Tensor
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
import utils_MNIST

def open_text_file(dataset,depth,W,learning_rate,weight_decay, input_dim, optimizer_choice, teacher_width, signal_noise_ratio, square_edge):
    if dataset == 'pMNIST':
        text_file = open(
            f'./logs/MNIST_{depth}_layer_{W}_lr_{learning_rate}_wd_{weight_decay}_inputdim_{input_dim}_{optimizer_choice}_training_parsed.txt',
            'w')
    if dataset == 'XOR':
        text_file = open(
            f'./logs/{dataset}_{signal_noise_ratio}_ratio_{depth}_layer_{W}_lr_{learning_rate}_wd_{weight_decay}_'
            f'inputdim_{input_dim}_{optimizer_choice}_training_parsed.txt', 'w')
    if dataset == 'teacher':
        text_file = open(f'./logs/{dataset}_teacherwidth_{teacher_width}_ratio_{signal_noise_ratio}_'
                         f'{depth}_layer_{W}_lr_{learning_rate}_wd_{weight_decay}_inputdim_{input_dim}_{optimizer_choice}_training_parsed.txt',
                         'w')
    if dataset == 'generalized_XOR':
        text_file = open(
            f'./logs/{dataset}_{square_edge}_edge_{signal_noise_ratio}_ratio_{depth}_layer_{W}_lr_{learning_rate}_wd_{weight_decay}_inputdim_'
            f'{input_dim}_{optimizer_choice}_training_parsed.txt', 'w')
    return text_file





def dataset_loader(trainortest,dataset, transform, input_dim, sample_size,signal_noise_ratio,teacher_width, square_edge):
    if trainortest=='train':
        if dataset == 'pMNIST':
            trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                                  download=True, transform=transform, target_transform=utils_MNIST.make_binary())
            testset = torchvision.datasets.MNIST(root='./data', train=False,
                                                 download=True, transform=transform, target_transform=utils_MNIST.make_binary())
            # Here we just convert MNIST into parity MNIST
            if input_dim != 784:
                try:
                    trainset = torch.load(f'./data/MNIST_PCA_{input_dim}_train.pt')
                    testset = torch.load(f'./data/MNIST_PCA_{input_dim}_test.pt')
                    print("Loaded dataset")
                except IOError:
                    utils_MNIST.create_pMNIST_PCA_dataset(trainset, testset, 784, input_dim)
                    print("Saved dataset")
                    trainset = torch.load(f'./data/MNIST_PCA_{input_dim}_train.pt')
                    testset = torch.load(f'./data/MNIST_PCA_{input_dim}_test.pt')
            if sample_size != 60000:
                # random_list=random.sample(range(60000), args.sample_size)
                ordered_list = list(range(0, sample_size - 1))
                trainset = torch.utils.data.Subset(trainset, ordered_list)

        if dataset == 'XOR':
            try:
                trainset = torch.load(
                    f'./data/XOR_{input_dim}_dimension_{signal_noise_ratio}_ratio_train_{sample_size}_samples.pt')
                testset = torch.load(f'./data/XOR_{input_dim}_dimension_{signal_noise_ratio}_ratio_test_{10000}_samples.pt')
                print('Loaded Dataset')
            except IOError:
                utils_MNIST.create_XOR_dataset(args.sample_size, 10000, input_dim, signal_noise_ratio)
                print('Saved Dataset')
                trainset = torch.load(
                    f'./data/XOR_{input_dim}_dimension_{signal_noise_ratio}_ratio_train_{sample_size}_samples.pt')
                testset = torch.load(f'./data/XOR_{input_dim}_dimension_{signal_noise_ratio}_ratio_test_{10000}_samples.pt')

        if dataset == 'teacher':
            try:
                trainset = torch.load(
                    f'./data/{dataset}_{input_dim}_dimension_teacherwidth_{teacher_width}_{signal_noise_ratio}_ratio_train_{sample_size}_samples.pt')
                testset = torch.load(
                    f'./data/{dataset}_{input_dim}_dimension_teacherwidth_{teacher_width}_{signal_noise_ratio}_ratio_test_{10000}_samples.pt')
                print('Loaded Dataset')
            except IOError:
                utils_MNIST.create_teacher_dataset(sample_size, 10000, input_dim, teacher_width, signal_noise_ratio)
                print('Saved Dataset')
                trainset = torch.load(
                    f'./data/{dataset}_{input_dim}_dimension_teacherwidth_{teacher_width}_{signal_noise_ratio}_ratio_train_{sample_size}_samples.pt')
                testset = torch.load(
                    f'./data/{dataset}_{input_dim}_dimension_teacherwidth_{teacher_width}_{signal_noise_ratio}_ratio_test_{10000}_samples.pt')

        if dataset == 'generalized_XOR':
            trainset = torch.load(f'./data/{dataset}_{square_edge}_edge_{input_dim}_dimension_{signal_noise_ratio}_ratio_'
                                  f'train_{sample_size}_samples.pt')
            testset = torch.load(f'./data/generalized_XOR_{square_edge}_edge_{input_dim}_dimension_'
                                 f'{signal_noise_ratio}_ratio_test_{10000}_samples.pt')
    else:
        if dataset == 'pMNIST':
            if input_dim == 784:
                transform = transforms.Compose(
                    [transforms.ToTensor(),
                     transforms.Normalize(0.5, 0.5)])
                trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                                      download=True, transform=transform,
                                                      target_transform=utils_MNIST.make_binary())
                testset = torchvision.datasets.MNIST(root='./data', train=False,
                                                     download=True, transform=transform, target_transform=utils_MNIST.make_binary())
            if input_dim != 784:
                trainset = torch.load(f'./data/MNIST_PCA_{input_dim}_train.pt')
                testset = torch.load(f'./data/MNIST_PCA_{input_dim}_test.pt')
            if sample_size != 60000:
                ordered_list = list(range(0, sample_size - 1))
                # random_list = random.sample(range(60000), args.sample_size)
                trainset = torch.utils.data.Subset(trainset, ordered_list)

        if dataset == 'XOR':
            trainset = torch.load(
                f'./data/{dataset}_{input_dim}_dimension_{signal_noise_ratio}_ratio_train_{sample_size}_samples.pt')
            testset = torch.load(
                f'./data/{dataset}_{input_dim}_dimension_{signal_noise_ratio}_ratio_test_{10000}_samples.pt')

        if dataset == 'teacher':
            trainset = torch.load(f'./data/{dataset}_{input_dim}_dimension_'
                                  f'teacherwidth_{teacher_width}_{signal_noise_ratio}_ratio_train_{sample_size}_samples.pt')
            testset = torch.load(f'./data/{dataset}_{input_dim}_dimension_'
                                 f'teacherwidth_{teacher_width}_{signal_noise_ratio}_ratio_test_{10000}_samples.pt')
        if dataset == 'generalized_XOR':
            trainset = torch.load(
                f'./data/{dataset}_{square_edge}_edge_{input_dim}_dimension_{signal_noise_ratio}_ratio_'
                f'train_{sample_size}_samples.pt')
            testset = torch.load(f'./data/generalized_XOR_{square_edge}_edge_{input_dim}_dimension_'
                                 f'{signal_noise_ratio}_ratio_test_{10000}_samples.pt')
    return trainset,testset






def save_weights(big_net,dataset,depth,W,k,learning_rate,weight_decay,input_dim, optimizer_choice, epochs,square_edge,signal_noise_ratio,teacher_width):
    if epochs==500:
        if dataset=='pMNIST':
            torch.save(big_net.state_dict(), f'./nets/{dataset}_trained_{depth}_layer_{W}_net_{k+1}_lr_{learning_rate}_wd_{weight_decay}_'
                                             f'inputdim_{input_dim}_{optimizer_choice}.pth')
        if dataset=='XOR':
            torch.save(big_net.state_dict(),f'./nets/{dataset}_trained_{depth}_layer_{W}_net_{k + 1}_lr_{learning_rate}_wd_{weight_decay}'
                                            f'_inputdim_{input_dim}_ratio_{signal_noise_ratio}_{optimizer_choice}.pth')
        if dataset=='teacher':
            torch.save(big_net.state_dict(),f'./nets/{dataset}_trained_{depth}_layer_{W}_net_{k + 1}_lr_{learning_rate}_wd_{weight_decay}_'
                                            f'inputdim_{input_dim}_teacherwidth_{teacher_width}_ratio_{signal_noise_ratio}_{optimizer_choice}.pth')
        if dataset=='generalized_XOR':
            torch.save(big_net.state_dict(),
                       f'./nets/{dataset}_trained_{depth}_layer_{W}_net_{k + 1}_lr_{learning_rate}_wd_{weight_decay}'
                       f'_inputdim_{input_dim}_edge_{square_edge}_ratio_{signal_noise_ratio}_{optimizer_choice}.pth')
    else:
        if dataset=='pMNIST':
            torch.save(big_net.state_dict(), f'./nets/{dataset}_trained_{depth}_layer_{W}_net_{k+1}_lr_{learning_rate}_wd_{weight_decay}_'
                                             f'inputdim_{input_dim}_{optimizer_choice}_{epochs}_epochs.pth')
        if dataset=='XOR':
            torch.save(big_net.state_dict(),f'./nets/{dataset}_trained_{depth}_layer_{W}_net_{k + 1}_lr_{learning_rate}_wd_{weight_decay}'
                                            f'_inputdim_{input_dim}_ratio_{signal_noise_ratio}_{optimizer_choice}_{epochs}_epochs.pth')
        if dataset=='teacher':
            torch.save(big_net.state_dict(),f'./nets/{dataset}_trained_{depth}_layer_{W}_net_{k + 1}_lr_{learning_rate}_wd_{weight_decay}_'
                                            f'inputdim_{input_dim}_teacherwidth_{teacher_width}_ratio_{signal_noise_ratio}_{optimizer_choice}'
                                            f'_{epochs}_epochs.pth')

        if dataset=='generalized_XOR':
            torch.save(big_net.state_dict(),
                       f'./nets/{dataset}_trained_{depth}_layer_{W}_net_{k + 1}_lr_{learning_rate}_wd_{weight_decay}'
                       f'_inputdim_{input_dim}_edge_{square_edge}_ratio_{signal_noise_ratio}_{optimizer_choice}_{epochs}_epochs.pth')






def save_accuracy_inf(accuracy_inf,dataset,depth,W,input_dim,learning_rate,weight_decay,square_edge,signal_noise_ratio,teacher_width,optimizer_choice,epochs):
    if epochs==500:
        if dataset == 'pMNIST':
            with open(f'./testerrors/ensemble_testerror_{dataset}_{depth}_layer_{W}_inputdim_{input_dim}_lr_{learning_rate}_wd_{weight_decay}_{optimizer_choice}.npy', 'wb') as f:
                np.save(f, accuracy_inf)
        if dataset == 'XOR':
            with open(f'./testerrors/ensemble_testerror_{dataset}_{depth}_layer_{W}_inputdim_{input_dim}_lr_{learning_rate}_wd_{weight_decay}_ratio_{signal_noise_ratio}_{optimizer_choice}.npy',
                    'wb') as f:
                np.save(f, accuracy_inf)
        if dataset == 'teacher':
            with open(f'./testerrors/ensemble_testerror_{dataset}_{depth}_layer_{W}_inputdim_{input_dim}_lr_{learning_rate}_wd_{weight_decay}_'
                      f'teacherwidth_{teacher_width}_ratio_{signal_noise_ratio}_{optimizer_choice}.npy', 'wb') as f:
                np.save(f, accuracy_inf)
        if dataset == 'generalized_XOR':
            with open(f'./testerrors/ensemble_testerror_{dataset}_{depth}_layer_{W}_inputdim_{input_dim}_lr_{learning_rate}_wd_{weight_decay}_'
                      f'edge_{square_edge}_ratio_{signal_noise_ratio}_{optimizer_choice}.npy','wb') as f:
                np.save(f, accuracy_inf)
    else:
        if dataset == 'pMNIST':
            with open(f'./testerrors/ensemble_testerror_{dataset}_{depth}_layer_{W}_inputdim_{input_dim}_lr_{learning_rate}_wd_{weight_decay}_{optimizer_choice}_{epochs}.npy', 'wb') as f:
                np.save(f, accuracy_inf)
        if dataset == 'XOR':
            with open(f'./testerrors/ensemble_testerror_{dataset}_{depth}_layer_{W}_inputdim_{input_dim}_lr_{learning_rate}'
                      f'_wd_{weight_decay}_ratio_{signal_noise_ratio}_{optimizer_choice}_{epochs}.npy',
                    'wb') as f:
                np.save(f, accuracy_inf)
        if dataset == 'teacher':
            with open(f'./testerrors/ensemble_testerror_{dataset}_{depth}_layer_{W}_inputdim_{input_dim}_lr_{learning_rate}_wd_{weight_decay}_'
                      f'teacherwidth_{teacher_width}_ratio_{signal_noise_ratio}_{optimizer_choice}_{epochs}.npy', 'wb') as f:
                np.save(f, accuracy_inf)
        if dataset == 'generalized_XOR':
            with open(f'./testerrors/ensemble_testerror_{dataset}_{depth}_layer_{W}_inputdim_{input_dim}_lr_{learning_rate}_wd_{weight_decay}_'
                      f'edge_{square_edge}_ratio_{signal_noise_ratio}_{optimizer_choice}_{epochs}.npy','wb') as f:
                np.save(f, accuracy_inf)





def save_average_acc(average,dataset,depth,W,input_dim,learning_rate,weight_decay,square_edge,signal_noise_ratio,teacher_width,optimizer_choice,epochs):
    if epochs==500:
        if dataset == 'pMNIST':
            with open(f'./testerrors/average_testerror_{dataset}_{depth}_layer_{W}_inputdim_{input_dim}_lr_{learning_rate}_wd_{weight_decay}_{optimizer_choice}.npy', 'wb') as f:
                np.save(f, average)
        if dataset == 'XOR':
            with open(f'./testerrors/average_testerror_{dataset}_{depth}_layer_{W}_inputdim_{input_dim}_lr_{learning_rate}_wd_{weight_decay}_ratio_{signal_noise_ratio}_{optimizer_choice}.npy',
                    'wb') as f:
                np.save(f, average)
        if dataset == 'teacher':
            with open(f'./testerrors/average_testerror_{dataset}_{depth}_layer_{W}_inputdim_{input_dim}_lr_{learning_rate}_wd_{weight_decay}_'
                      f'teacherwidth_{teacher_width}_ratio_{signal_noise_ratio}_{optimizer_choice}.npy', 'wb') as f:
                np.save(f, average)
        if dataset == 'generalized_XOR':
            with open(f'./testerrors/average_testerror_{dataset}_{depth}_layer_{W}_inputdim_{input_dim}_lr_{learning_rate}_wd_{weight_decay}_'
                      f'edge_{square_edge}_ratio_{signal_noise_ratio}_{optimizer_choice}.npy','wb') as f:
                np.save(f, average)
    else:
        if dataset == 'pMNIST':
            with open(f'./testerrors/average_testerror_{dataset}_{depth}_layer_{W}_inputdim_{input_dim}_lr_'
                      f'{learning_rate}_wd_{weight_decay}_{optimizer_choice}_{epochs}.npy', 'wb') as f:
                np.save(f, average)
        if dataset == 'XOR':
            with open(f'./testerrors/average_testerror_{dataset}_{depth}_layer_{W}_inputdim_{input_dim}_lr_'
                      f'{learning_rate}_wd_{weight_decay}_ratio_{signal_noise_ratio}_{optimizer_choice}_{epochs}.npy',
                    'wb') as f:
                np.save(f, average)
        if dataset == 'teacher':
            with open(f'./testerrors/average_testerror_{dataset}_{depth}_layer_{W}_inputdim_{input_dim}_lr_{learning_rate}_wd_{weight_decay}_'
                      f'teacherwidth_{teacher_width}_ratio_{signal_noise_ratio}_{optimizer_choice}_{epochs}.npy', 'wb') as f:
                np.save(f, average)
        if dataset == 'generalized_XOR':
            with open(f'./testerrors/average_testerror_{dataset}_{depth}_layer_{W}_inputdim_{input_dim}_lr_{learning_rate}_wd_{weight_decay}_'
                      f'edge_{square_edge}_ratio_{signal_noise_ratio}_{optimizer_choice}_{epochs}.npy','wb') as f:
                np.save(f, average)





def save_delta_error(delta_error,dataset,depth,W,input_dim,learning_rate,weight_decay,optimizer_choice,epochs,signal_noise_ratio,teacher_width,square_edge):
    if epochs==500:
        if dataset == 'pMNIST':
            with open(f'./deltaerrors/ensemble_deltaerror_{dataset}_{depth}_layer_{W}_inputdim_{input_dim}_lr_{learning_rate}_wd_{weight_decay}_{optimizer_choice}.npy', 'wb') as f:
                np.save(f, delta_error)
        if dataset == 'XOR':
            with open(f'./deltaerrors/ensemble_deltaerror_{dataset}_{depth}_layer_{W}_inputdim_{input_dim}_lr_{learning_rate}_wd_{weight_decay}_ratio_{signal_noise_ratio}_{optimizer_choice}.npy',
                    'wb') as f:
                np.save(f, delta_error)
                print("Saved XOR accuracy")
        if dataset == 'teacher':
            with open(f'./deltaerrors/ensemble_deltaerror_{dataset}_{depth}_layer_{W}_inputdim_{input_dim}_lr_{learning_rate}_wd_{weight_decay}_'
                      f'teacherwidth_{teacher_width}_ratio_{signal_noise_ratio}_{optimizer_choice}.npy', 'wb') as f:
                np.save(f, delta_error)
        if dataset == 'generalized_XOR':
            with open(f'./deltaerrors/ensemble_deltaerror_{dataset}_{depth}_layer_{W}_inputdim_{input_dim}_lr_{learning_rate}_wd_{weight_decay}_'
                      f'edge_{square_edge}_ratio_{signal_noise_ratio}_{optimizer_choice}.npy',
                    'wb') as f:
                np.save(f, delta_error)
    else:
        if dataset == 'pMNIST':
            with open(f'./deltaerrors/ensemble_deltaerror_{dataset}_{depth}_layer_{W}_inputdim_{input_dim}_lr_'
                      f'{learning_rate}_wd_{weight_decay}_{optimizer_choice}_{epochs}.npy', 'wb') as f:
                np.save(f, delta_error)
        if dataset == 'XOR':
            with open(f'./deltaerrors/ensemble_deltaerror_{dataset}_{depth}_layer_{W}_inputdim_{input_dim}_lr_'
                      f'{learning_rate}_wd_{weight_decay}_ratio_{signal_noise_ratio}_{optimizer_choice}_{epochs}.npy',
                    'wb') as f:
                np.save(f, delta_error)
                print("Saved XOR accuracy")
        if dataset == 'teacher':
            with open(f'./deltaerrors/ensemble_deltaerror_{dataset}_{depth}_layer_{W}_inputdim_{input_dim}_lr_{learning_rate}_wd_{weight_decay}_'
                      f'teacherwidth_{teacher_width}_ratio_{signal_noise_ratio}_{optimizer_choice}_{epochs}.npy', 'wb') as f:
                np.save(f, delta_error)
        if dataset == 'generalized_XOR':
            with open(f'./deltaerrors/ensemble_deltaerror_{dataset}_{depth}_layer_{W}_inputdim_{input_dim}_lr_{learning_rate}_wd_{weight_decay}_'
                      f'edge_{square_edge}_ratio_{signal_noise_ratio}_{optimizer_choice}_{epochs}.npy',
                    'wb') as f:
                np.save(f, delta_error)





def save_train_loss(train_counter, train_losses,dataset,depth,W,input_dim,optimizer_choice,epochs,signal_noise_ratio,teacher_width,square_edge):
    fig = plt.figure()
    plt.plot(train_counter, train_losses, color='blue')
    plt.legend(['Train Loss'], loc='upper right')
    plt.xlabel('number of training examples seen')
    plt.ylabel('training loss')
    if epochs==500:
        if dataset == 'pMNIST':
            plt.savefig(
                f'./training_losses_plots/{dataset}_{depth}_layer_{W}_inputdim_{input_dim}_{optimizer_choice}_training_loss.png')
        if dataset == 'XOR':
            plt.savefig(
                f'./training_losses_plots/{dataset}_{depth}_layer_{W}_inputdim_{input_dim}_ratio_{signal_noise_ratio}_{optimizer_choice}_training_loss.png')
        if dataset == 'teacher':
            plt.savefig(
                f'./training_losses_plots/{dataset}_{depth}_layer_{W}_inputdim_{input_dim}_teacherwidth_{teacher_width}_ratio_{signal_noise_ratio}'
                f'{optimizer_choice}_training_loss.png')
        if dataset == 'generalized_XOR':
            plt.savefig(f'./training_losses_plots/{dataset}_{depth}_layer_{W}_inputdim_{input_dim}_'
                        f'edge_{square_edge}_ratio_{signal_noise_ratio}_{optimizer_choice}_training_loss.png')
    else:
        if dataset == 'pMNIST':
            plt.savefig(
                f'./training_losses_plots/{dataset}_{depth}_layer_{W}_inputdim_{input_dim}_{optimizer_choice}_epochs_{epochs}_training_loss.png')
        if dataset == 'XOR':
            plt.savefig(
                f'./training_losses_plots/{dataset}_{depth}_layer_{W}_inputdim_{input_dim}_ratio_{signal_noise_ratio}_'
                f'{optimizer_choice}_epochs_{epochs}__training_loss.png')
        if dataset == 'teacher':
            plt.savefig(
                f'./training_losses_plots/{dataset}_{depth}_layer_{W}_inputdim_{input_dim}_teacherwidth_{teacher_width}_ratio_{signal_noise_ratio}'
                f'{optimizer_choice}_epochs_{epochs}__training_loss.png')
        if dataset == 'generalized_XOR':
            plt.savefig(f'./training_losses_plots/{dataset}_{depth}_layer_{W}_inputdim_{input_dim}_'
                        f'edge_{square_edge}_ratio_{signal_noise_ratio}_{optimizer_choice}_epochs_{epochs}__training_loss.png')





def load_accuracy_inf(dataset,depth,W,input_dim,learning_rate,weight_decay,optimizer_choice,epochs,signal_noise_ratio,teacher_width,square_edge):
    if epochs==500:
        if dataset == 'pMNIST':
            accuracy_inf = np.load(
                f'./testerrors/ensemble_testerror_{dataset}_{depth}_layer_{W}_inputdim_{input_dim}_lr_{learning_rate}_wd_{weight_decay}'
                f'_{optimizer_choice}.npy')
        if dataset == 'XOR':
            accuracy_inf = np.load(f'./testerrors/ensemble_testerror_{dataset}_{depth}_layer_{W}_inputdim_'
                                   f'{input_dim}_lr_{learning_rate}_wd_{weight_decay}_ratio_{signal_noise_ratio}_{optimizer_choice}.npy')
        if dataset == 'teacher':
            accuracy_inf = np.load(f'./testerrors/ensemble_testerror_{dataset}_{depth}_layer_{W}_inputdim_{input_dim}_lr_'
                                   f'{learning_rate}_wd_{weight_decay}_teacherwidth_{teacher_width}_ratio_{signal_noise_ratio}_{optimizer_choice}.npy')
        if dataset == 'generalized_XOR':
            accuracy_inf = np.load(f'./testerrors/ensemble_testerror_{dataset}_{depth}_layer_{W}_inputdim_{input_dim}_lr_'
                                   f'{learning_rate}_wd_{weight_decay}_edge_{square_edge}_ratio_{signal_noise_ratio}_{optimizer_choice}.npy')
    else:
        if dataset == 'pMNIST':
            accuracy_inf = np.load(
                f'./testerrors/ensemble_testerror_{dataset}_{depth}_layer_{W}_inputdim_{input_dim}_lr_{learning_rate}_wd_{weight_decay}'
                f'_{optimizer_choice}_{epochs}.npy')
        if dataset == 'XOR':
            accuracy_inf = np.load(f'./testerrors/ensemble_testerror_{dataset}_{depth}_layer_{W}_inputdim_'
                                   f'{input_dim}_lr_{learning_rate}_wd_{weight_decay}_ratio_{signal_noise_ratio}_{optimizer_choice}_{epochs}.npy')
        if dataset == 'teacher':
            accuracy_inf = np.load(f'./testerrors/ensemble_testerror_{dataset}_{depth}_layer_{W}_inputdim_{input_dim}_lr_'
                                   f'{learning_rate}_wd_{weight_decay}_teacherwidth_{teacher_width}_ratio_{signal_noise_ratio}_{optimizer_choice}_{epochs}.npy')
        if dataset == 'generalized_XOR':
            accuracy_inf = np.load(f'./testerrors/ensemble_testerror_{dataset}_{depth}_layer_{W}_inputdim_{input_dim}_lr_'
                                   f'{learning_rate}_wd_{weight_decay}_edge_{square_edge}_ratio_{signal_noise_ratio}_{optimizer_choice}_{epochs}.npy')
    return accuracy_inf





def load_weights(dataset,depth,W,i,learning_rate,weight_decay,input_dim,optimizer_choice,epochs,signal_noise_ratio,teacher_width,square_edge):
    if epochs==500:
        if dataset == 'pMNIST':
            PATH = f'./nets/pMNIST_trained_{depth}_layer_{W}_net_{i + 1}_lr_{learning_rate}_wd_{weight_decay}_inputdim_{input_dim}_{optimizer_choice}.pth'
        if dataset == 'XOR':
            PATH = f'./nets/{dataset}_trained_{depth}_layer_{W}_net_{i + 1}_lr_{learning_rate}_wd_{weight_decay}_inputdim_' \
                   f'{input_dim}_ratio_{signal_noise_ratio}_{optimizer_choice}.pth'
        if dataset == 'teacher':
            PATH = f'./nets/{dataset}_trained_{depth}_layer_{W}_net_{i + 1}_lr_{learning_rate}_wd_{weight_decay}_' \
                   f'inputdim_{input_dim}_teacherwidth_{teacher_width}_ratio_{signal_noise_ratio}_{optimizer_choice}.pth'
        if dataset == 'generalized_XOR':
            PATH = f'./nets/{dataset}_trained_{depth}_layer_{W}_net_{i + 1}_lr_{learning_rate}_wd_{weight_decay}_inputdim_' \
                   f'{input_dim}_edge_{square_edge}_ratio_{signal_noise_ratio}_{optimizer_choice}.pth'
    else:
        if dataset == 'pMNIST':
            PATH = f'./nets/pMNIST_trained_{depth}_layer_{W}_net_{i + 1}_lr_{learning_rate}_wd_{weight_decay}_inputdim_' \
                   f'{input_dim}_{optimizer_choice}_{epochs}_epochs.pth'
        if dataset == 'XOR':
            PATH = f'./nets/{dataset}_trained_{depth}_layer_{W}_net_{i + 1}_lr_{learning_rate}_wd_{weight_decay}_inputdim_' \
                   f'{input_dim}_ratio_{signal_noise_ratio}_{optimizer_choice}_{epochs}_epochs.pth'
        if dataset == 'teacher':
            PATH = f'./nets/{dataset}_trained_{depth}_layer_{W}_net_{i + 1}_lr_{learning_rate}_wd_{weight_decay}_' \
                   f'inputdim_{input_dim}_teacherwidth_{teacher_width}_ratio_{signal_noise_ratio}_{optimizer_choice}_{epochs}_epochs.pth'
        if dataset == 'generalized_XOR':
            PATH = f'./nets/{dataset}_trained_{depth}_layer_{W}_net_{i + 1}_lr_{learning_rate}_wd_{weight_decay}_inputdim_' \
                   f'{input_dim}_edge_{square_edge}_ratio_{signal_noise_ratio}_{optimizer_choice}_{epochs}_epochs.pth'
    return PATH





def save_train_acc(arr1, train_accs, train_acc_errors, dataset, depth, W, input_dim, weight_decay, learning_rate, optimizer_choice, n_epochs, signal_noise_ratio, teacher_width, square_edge):
    plt.clf()
    ax = plt.gca()
    ax.set_xlim([2, 1.2 * W])
    train_errors = 100 - train_accs
    ax.set_xscale('log')
    for i in range(1,len(train_errors)):
        if train_errors[i]==0:
            train_errors[i]=train_errors[i-1]
    ax.set_ylim([train_errors[-1]*0.8,train_errors[0]*1.2])
    ax.errorbar(arr1, train_errors, yerr=train_acc_errors, fmt="bo")
    x = np.linspace(2, 1.2 * arr1[-1], 1000)

    y = np.full(1000, train_errors[-1])
    ax.plot(x, y, label='constant')
    ax.set_title(f'FCN of depth {depth}, input dimension={input_dim} and wd={weight_decay}')
    plt.xlabel('chunks width')
    plt.ylabel('train set accuracy')
    if dataset == 'pMNIST':
        plt.savefig(
            f'./plots/train_acc_plots/{dataset}_{depth}_layer_{W}_lr_{learning_rate}_wd_{weight_decay}_inputdim_{input_dim}_{optimizer_choice}_{n_epochs}_train_acc.png')
    if dataset == 'XOR':
        plt.savefig(
            f'./plots/train_acc_plots/{dataset}_{depth}_layer_{W}_lr_{learning_rate}_wd_{weight_decay}_inputdim_{input_dim}_'
            f'ratio_{signal_noise_ratio}_{optimizer_choice}_{n_epochs}_train_acc.png')
    if dataset == 'teacher':
        plt.savefig(
            f'./plots/train_acc_plots/{dataset}_{depth}_layer_{W}_lr_{learning_rate}_wd_{weight_decay}_inputdim_{input_dim}_'
            f'teacherwidth_{teacher_width}_ratio_{signal_noise_ratio}_{optimizer_choice}_{n_epochs}_train_acc.png')
    if dataset == 'generalized_XOR':
        plt.savefig(
            f'./plots/train_acc_plots/{dataset}_{depth}_layer_{W}_lr_{learning_rate}_wd_{weight_decay}_inputdim_{input_dim}_'
            f'edge_{square_edge}_ratio_{signal_noise_ratio}_{optimizer_choice}_{n_epochs}_train_acc.png')



def save_all_arrays(dataset,n_epochs,depth,W,input_dim,learning_rate,weight_decay,optimizer_choice,arr1,arr5,acc_errors,
                    train_accs,train_acc_errors,signal_noise_ratio,teacher_width,square_edge):
    if n_epochs == 500:
        if dataset == 'pMNIST':
            with open(
                    f'./arrays/sizes_{depth}_layer_{W}_inputdim_{input_dim}_lr_{learning_rate}_wd_{weight_decay}_{optimizer_choice}.npy',
                    'wb') as f:
                np.save(f, arr1)
            with open(
                    f'./arrays/error_{depth}_layer_{W}_inputdim_{input_dim}_lr_{learning_rate}_wd_{weight_decay}_{optimizer_choice}.npy',
                    'wb') as f:
                np.save(f, arr5)
            with open(
                    f'./arrays/error_on_error_{depth}_layer_{W}_inputdim_{input_dim}_lr_{learning_rate}_wd_{weight_decay}_{optimizer_choice}.npy',
                    'wb') as f:
                np.save(f, acc_errors)
            with open(
                    f'./arrays/train_error_{depth}_layer_{W}_inputdim_{input_dim}_lr_{learning_rate}_wd_{weight_decay}_{optimizer_choice}.npy',
                    'wb') as f:
                np.save(f, train_accs)
            with open(
                    f'./arrays/train_acc_error_{depth}_layer_{W}_inputdim_{input_dim}_lr_{learning_rate}_wd_{weight_decay}_{optimizer_choice}.npy',
                    'wb') as f:
                np.save(f, train_acc_errors)
        if dataset == 'XOR':
            with open(
                    f'./arrays/sizes_{dataset}_{depth}_layer_{W}_inputdim_{input_dim}_lr_{learning_rate}_wd_{weight_decay}_'
                    f'ratio_{signal_noise_ratio}_{optimizer_choice}.npy', 'wb') as f:
                np.save(f, arr1)
            with open(
                    f'./arrays/error_{dataset}_{depth}_layer_{W}_inputdim_{input_dim}_lr_{learning_rate}_wd_{weight_decay}_'
                    f'ratio_{signal_noise_ratio}_{optimizer_choice}.npy', 'wb') as f:
                np.save(f, arr5)
            with open(
                    f'./arrays/error_on_error_{dataset}_{depth}_layer_{W}_inputdim_{input_dim}_lr_{learning_rate}_wd_{weight_decay}_'
                    f'ratio_{signal_noise_ratio}_{optimizer_choice}.npy', 'wb') as f:
                np.save(f, acc_errors)
            with open(
                    f'./arrays/train_error_{dataset}_{depth}_layer_{W}_inputdim_{input_dim}_lr_{learning_rate}_wd_{weight_decay}_'
                    f'ratio_{signal_noise_ratio}_{optimizer_choice}.npy', 'wb') as f:
                np.save(f, train_accs)
            with open(
                    f'./arrays/train_acc_error_{dataset}_{depth}_layer_{W}_inputdim_{input_dim}_lr_{learning_rate}_wd_{weight_decay}_'
                    f'ratio_{signal_noise_ratio}_{optimizer_choice}.npy', 'wb') as f:
                np.save(f, train_acc_errors)
        if dataset == 'teacher':
            with open(
                    f'./arrays/sizes_{dataset}_{depth}_layer_{W}_inputdim_{input_dim}_lr_{learning_rate}_wd_{weight_decay}_'
                    f'teacherwidth_{teacher_width}_ratio_{signal_noise_ratio}_{optimizer_choice}.npy', 'wb') as f:
                np.save(f, arr1)
            with open(
                    f'./arrays/error_{dataset}_{depth}_layer_{W}_inputdim_{input_dim}_lr_{learning_rate}_wd_{weight_decay}_'
                    f'teacherwidth_{teacher_width}_ratio_{signal_noise_ratio}_{optimizer_choice}.npy', 'wb') as f:
                np.save(f, arr5)
            with open(f'./arrays/error_on_error_{dataset}_{depth}_layer_{W}_inputdim_{input_dim}_lr_{learning_rate}_wd_'
                      f'{weight_decay}_{optimizer_choice}.npy', 'wb') as f:
                np.save(f, acc_errors)
            with open(f'./arrays/train_error_{dataset}_{depth}_layer_{W}_inputdim_{input_dim}_lr_{learning_rate}_wd_'
                      f'{weight_decay}_{optimizer_choice}.npy', 'wb') as f:
                np.save(f, train_accs)
            with open(
                    f'./arrays/train_acc_error_{dataset}_{depth}_layer_{W}_inputdim_{input_dim}_lr_{learning_rate}_wd_'
                    f'{weight_decay}_{optimizer_choice}.npy', 'wb') as f:
                np.save(f, train_acc_errors)
        if dataset == 'generalized_XOR':
            with open(
                    f'./arrays/sizes_{dataset}_{depth}_layer_{W}_inputdim_{input_dim}_lr_{learning_rate}_wd_{weight_decay}_'
                    f'edge_{square_edge}_ratio_{signal_noise_ratio}_{optimizer_choice}.npy', 'wb') as f:
                np.save(f, arr1)
            with open(
                    f'./arrays/error_{dataset}_{depth}_layer_{W}_inputdim_{input_dim}_lr_{learning_rate}_wd_{weight_decay}_'
                    f'edge_{square_edge}_ratio_{signal_noise_ratio}_{optimizer_choice}.npy', 'wb') as f:
                np.save(f, arr5)
            with open(
                    f'./arrays/error_on_error_{dataset}_{depth}_layer_{W}_inputdim_{input_dim}_lr_{learning_rate}_wd_{weight_decay}_'
                    f'edge_{square_edge}_ratio_{signal_noise_ratio}_{optimizer_choice}.npy', 'wb') as f:
                np.save(f, acc_errors)
            with open(
                    f'./arrays/train_error_{dataset}_{depth}_layer_{W}_inputdim_{input_dim}_lr_{learning_rate}_wd_{weight_decay}_'
                    f'edge_{square_edge}_ratio_{signal_noise_ratio}_{optimizer_choice}.npy', 'wb') as f:
                np.save(f, train_accs)
            with open(
                    f'./arrays/tran_acc_error_{dataset}_{depth}_layer_{W}_inputdim_{input_dim}_lr_{learning_rate}_wd_{weight_decay}_'
                    f'edge_{square_edge}_ratio_{signal_noise_ratio}_{optimizer_choice}.npy', 'wb') as f:
                np.save(f, train_acc_errors)
    else:
        if dataset == 'pMNIST':
            with open(f'./arrays/sizes_{depth}_layer_{W}_inputdim_{input_dim}_lr_{learning_rate}_wd_{weight_decay}_'
                      f'{optimizer_choice}_{n_epochs}.npy', 'wb') as f:
                np.save(f, arr1)
            with open(f'./arrays/error_{depth}_layer_{W}_inputdim_{input_dim}_lr_{learning_rate}_wd_{weight_decay}_'
                      f'{optimizer_choice}_{n_epochs}.npy', 'wb') as f:
                np.save(f, arr5)
            with open(
                    f'./arrays/error_on_error_{depth}_layer_{W}_inputdim_{input_dim}_lr_{learning_rate}_wd_{weight_decay}_'
                    f'{optimizer_choice}_{n_epochs}.npy', 'wb') as f:
                np.save(f, acc_errors)
            with open(
                    f'./arrays/train_error_{depth}_layer_{W}_inputdim_{input_dim}_lr_{learning_rate}_wd_{weight_decay}_'
                    f'{optimizer_choice}_{n_epochs}.npy', 'wb') as f:
                np.save(f, train_accs)
            with open(
                    f'./arrays/train_acc_error_{depth}_layer_{W}_inputdim_{input_dim}_lr_{learning_rate}_wd_{weight_decay}_'
                    f'{optimizer_choice}_{n_epochs}.npy', 'wb') as f:
                np.save(f, train_acc_errors)
        if dataset == 'XOR':
            with open(
                    f'./arrays/sizes_{dataset}_{depth}_layer_{W}_inputdim_{input_dim}_lr_{learning_rate}_wd_{weight_decay}_'
                    f'ratio_{signal_noise_ratio}_{optimizer_choice}_{n_epochs}.npy', 'wb') as f:
                np.save(f, arr1)
            with open(
                    f'./arrays/error_{dataset}_{depth}_layer_{W}_inputdim_{input_dim}_lr_{learning_rate}_wd_{weight_decay}_'
                    f'ratio_{signal_noise_ratio}_{optimizer_choice}_{n_epochs}.npy', 'wb') as f:
                np.save(f, arr5)
            with open(
                    f'./arrays/error_on_error_{dataset}_{depth}_layer_{W}_inputdim_{input_dim}_lr_{learning_rate}_wd_{weight_decay}_'
                    f'ratio_{signal_noise_ratio}_{optimizer_choice}_{n_epochs}.npy', 'wb') as f:
                np.save(f, acc_errors)
            with open(
                    f'./arrays/train_error_{dataset}_{depth}_layer_{W}_inputdim_{input_dim}_lr_{learning_rate}_wd_{weight_decay}_'
                    f'ratio_{signal_noise_ratio}_{optimizer_choice}_{n_epochs}.npy', 'wb') as f:
                np.save(f, train_accs)
            with open(
                    f'./arrays/train_acc_error_{dataset}_{depth}_layer_{W}_inputdim_{input_dim}_lr_{learning_rate}_wd_{weight_decay}_'
                    f'ratio_{signal_noise_ratio}_{optimizer_choice}_{n_epochs}.npy', 'wb') as f:
                np.save(f, train_acc_errors)
        if dataset == 'teacher':
            with open(
                    f'./arrays/sizes_{dataset}_{depth}_layer_{W}_inputdim_{input_dim}_lr_{learning_rate}_wd_{weight_decay}_'
                    f'teacherwidth_{teacher_width}_ratio_{signal_noise_ratio}_{optimizer_choice}_{n_epochs}.npy',
                    'wb') as f:
                np.save(f, arr1)
            with open(
                    f'./arrays/error_{dataset}_{depth}_layer_{W}_inputdim_{input_dim}_lr_{learning_rate}_wd_{weight_decay}_'
                    f'teacherwidth_{teacher_width}_ratio_{signal_noise_ratio}_{optimizer_choice}_{n_epochs}.npy',
                    'wb') as f:
                np.save(f, arr5)
            with open(
                    f'./arrays/error_on_error_{dataset}_{depth}_layer_{W}_inputdim_{input_dim}_lr_{learning_rate}_wd_{weight_decay}_'
                    f'teacherwidth_{teacher_width}_ratio_{signal_noise_ratio}_{optimizer_choice}_{n_epochs}.npy',
                    'wb') as f:
                np.save(f, acc_errors)
            with open(
                    f'./arrays/train_error_{dataset}_{depth}_layer_{W}_inputdim_{input_dim}_lr_{learning_rate}_wd_{weight_decay}_'
                    f'teacherwidth_{teacher_width}_ratio_{signal_noise_ratio}_{optimizer_choice}_{n_epochs}.npy',
                    'wb') as f:
                np.save(f, train_accs)
            with open(
                    f'./arrays/train_acc_error_{dataset}_{depth}_layer_{W}_inputdim_{input_dim}_lr_{learning_rate}_wd_{weight_decay}_'
                    f'teacherwidth_{teacher_width}_ratio_{signal_noise_ratio}_{optimizer_choice}_{n_epochs}.npy',
                    'wb') as f:
                np.save(f, train_acc_errors)
        if dataset == 'generalized_XOR':
            with open(
                    f'./arrays/sizes_{dataset}_{depth}_layer_{W}_inputdim_{input_dim}_lr_{learning_rate}_wd_{weight_decay}_'
                    f'edge_{square_edge}_ratio_{signal_noise_ratio}_{optimizer_choice}_{n_epochs}.npy', 'wb') as f:
                np.save(f, arr1)
            with open(
                    f'./arrays/error_{dataset}_{depth}_layer_{W}_inputdim_{input_dim}_lr_{learning_rate}_wd_{weight_decay}_'
                    f'edge_{square_edge}_ratio_{signal_noise_ratio}_{optimizer_choice}_{n_epochs}.npy', 'wb') as f:
                np.save(f, arr5)
            with open(
                    f'./arrays/error_on_error_{dataset}_{depth}_layer_{W}_inputdim_{input_dim}_lr_{learning_rate}_wd_{weight_decay}_'
                    f'edge_{square_edge}_ratio_{signal_noise_ratio}_{optimizer_choice}_{n_epochs}.npy', 'wb') as f:
                np.save(f, acc_errors)
            with open(
                    f'./arrays/train_error_{dataset}_{depth}_layer_{W}_inputdim_{input_dim}_lr_{learning_rate}_wd_{weight_decay}_'
                    f'edge_{square_edge}_ratio_{signal_noise_ratio}_{optimizer_choice}_{n_epochs}.npy', 'wb') as f:
                np.save(f, train_accs)
            with open(
                    f'./arrays/train_acc_error_{dataset}_{depth}_layer_{W}_inputdim_{input_dim}_lr_{learning_rate}_wd_{weight_decay}_'
                    f'edge_{square_edge}_ratio_{signal_noise_ratio}_{optimizer_choice}_{n_epochs}.npy', 'wb') as f:
                np.save(f, train_acc_errors)