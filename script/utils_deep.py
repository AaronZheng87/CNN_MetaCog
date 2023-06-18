#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  3 13:27:27 2022

@author: ningmei
"""
import os,torch,torchvision,gc

import numpy as np
import pandas as pd

from glob import glob
from tqdm import tqdm
from PIL import Image as pil_image
from itertools import product
from time import sleep

from torch.utils.data import Dataset,DataLoader
from torch import nn
from torchvision import models as Tmodels
from torchvision.datasets import MNIST
from torchvision.utils import make_grid
from torchvision import transforms
from torchvision.models import ResNet50_Weights, AlexNet_Weights, DenseNet169_Weights, VGG19_BN_Weights,MobileNet_V2_Weights, MobileNet_V3_Large_Weights, Inception_V3_Weights,ResNet18_Weights, SqueezeNet1_1_Weights, ResNeXt50_32X4D_Weights, GoogLeNet_Weights, ShuffleNet_V2_X0_5_Weights
from sklearn.metrics import roc_curve

from joblib import Parallel,delayed

from matplotlib import pyplot as plt

from typing import List, Callable, Union, Any, TypeVar, Tuple
###############################################################################
Tensor = TypeVar('torch.tensor')
###############################################################################
torch.manual_seed(12345)
np.random.seed(12345)

##############################################################################
def noise_func(x:Tensor,noise_level:float = 0.):
    """
    add guassian noise to the images during agumentation procedures
    Inputs
    --------------------
    x: torch.tensor, batch_size x 3 x height x width
    noise_level: float, level of noise, between 0 and 1
    """
    
    generator   = torch.distributions.HalfNormal(x.std())
    noise       = generator.sample(x.shape)
    new_x       = x * (1 - noise_level) + noise * noise_level
    new_x       = torch.clamp(new_x,x.min(),x.max(),)
    return new_x

def concatenate_transform_steps(image_resize:int = 128,
                                num_output_channels:int = 3,
                                noise_level:float = 0.,
                                rotate:float = 0.,
                                fill_empty_space:int = 255,
                                grayscale:bool = True,
                                center_crop:bool = False,
                                center_crop_size:Tuple = (1024,1024),
                                ):
    """
    from image to tensors

    Parameters
    ----------
    image_resize : int, optional
        DESCRIPTION. The default is 128.
    num_output_channels : int, optional
        DESCRIPTION. The default is 3.
    noise_level : float, optional
        DESCRIPTION. The default is 0..
    rotate : float, optional
        DESCRIPTION. The default is 0.,
    fill_empty_space : int, optional
        DESCRIPTION. The defaultis 130.
    grayscale: bool, optional
        DESCRIPTION. The default is True.
    center_crop : bool, optional
        DESCRIPTION. The default is False.
    center_crop_size : Tuple, optional
        DESCRIPTION. The default is (1024, 1024)

    Returns
    -------
    transformer_steps : TYPE
        DESCRIPTION.

    """
    transformer_steps = []
    # crop the image - for grid like layout
    if center_crop:
        transformer_steps.append(transforms.CenterCrop(center_crop_size))
    # resize
    transformer_steps.append(transforms.Resize((image_resize,image_resize)))
    # rotation
    if rotate > 0.:
        transformer_steps.append(transforms.RandomHorizontalFlip(p = .5))
        transformer_steps.append(transforms.RandomVerticalFlip(p = .5))
        transformer_steps.append(transforms.RandomRotation(degrees = rotate,
                                                           fill = fill_empty_space,
                                                           ))
    # grayscale
    if grayscale:
        transformer_steps.append(# it needs to be 3 if we want to use pretrained CV models
                                transforms.Grayscale(num_output_channels = num_output_channels)
                                )
    # rescale to [0,1] from int8
    transformer_steps.append(transforms.ToTensor())
    # add noise
    if noise_level > 0:
        transformer_steps.append(transforms.Lambda(lambda x:noise_func(x,noise_level)))
    # normalization
    transformer_steps.append(transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                  std=[0.229, 0.224, 0.225])
                             )
    transformer_steps = transforms.Compose(transformer_steps)
    return transformer_steps

def append_to_dict_list(df:pd.core.frame.DataFrame,
                        attribute,
                        variable,
                        ):
    try:
        idx_column = int(attribute[-1]) - 1
    except:
        idx_column = 0
    temp = variable.detach().cpu().numpy()[:,idx_column]
    [df[attribute].append(item) for item in temp]
    return df

def psychometric_curve(x,a,b,c,d):
    return a / (1. + np.exp(-c * (x - d))) + b


#candidate models
def candidates(model_name:str,) -> nn.Module:
    """
    A simple loader for the CNN backbone models
    Parameters
    ----------
    model_name : str
        DESCRIPTION.
    Returns
    -------
    nn.Module
        A pretrained CNN model.
    """
def candidates(model_name:str,) -> nn.Module:
    """
    A simple loader for the CNN backbone models
    Parameters
    ----------
    model_name : str
        DESCRIPTION.
    Returns
    -------
    nn.Module
        A pretrained CNN model.
    """
    picked_models = dict(
            # resnet18        = Tmodels.resnet18(weights              = "IMAGENET1K_V1",
            #                                    progress             = False,),
            alexnet         = Tmodels.alexnet(weights              = AlexNet_Weights.IMAGENET1K_V1,
                                              progress               = True,),
            # squeezenet      = Tmodels.squeezenet1_1(weights              = "IMAGENET1K_V1",
            #                                        progress         = False,),
            vgg19           = Tmodels.vgg19_bn(weights              = VGG19_BN_Weights.IMAGENET1K_V1,
                                              progress              = True,),
            densenet169     = Tmodels.densenet169(weights           = DenseNet169_Weights.IMAGENET1K_V1,
                                                  progress          = True,),
            # inception       = Tmodels.inception_v3(weights              = "IMAGENET1K_V1",
            #                                       progress          = False,),
            # googlenet       = Tmodels.googlenet(weights              = "IMAGENET1K_V1",
            #                                    progress             = False,),
            # shufflenet      = Tmodels.shufflenet_v2_x0_5(weights              = "IMAGENET1K_V1",
            #                                             progress    = False,),
            mobilenet       = Tmodels.mobilenet_v2(weights          = MobileNet_V2_Weights.IMAGENET1K_V1,
                                                  progress          = True,),
            # mobilenet_v3_l  = Tmodels.mobilenet_v3_large(weights              = "IMAGENET1K_V1",
            #                                              progress   = False,),
            # resnext50_32x4d = Tmodels.resnext50_32x4d(weights              = "IMAGENET1K_V1",
            #                                          progress       = False,),
            resnet50        = Tmodels.resnet50(weights              = ResNet50_Weights.IMAGENET1K_V1,
                                              progress              = True,),
            )
    return picked_models[model_name]

def define_type(model_name:str) -> str:
    """
    We define the type of the pretrained CNN models for easier transfer learning
    Parameters
    ----------
    model_name : str
        DESCRIPTION.
    Returns
    -------
    str
        DESCRIPTION.
    """
    model_type          = dict(
            alexnet     = 'simple',
            vgg19       = 'simple',
            densenet169 = 'simple',
            inception   = 'inception',
            mobilenet   = 'simple',
            resnet18    = 'resnet',
            resnet50    = 'resnet',
            )
    return model_type[model_name]

def hidden_activation_functions(activation_func_name:str,num_parameters:int=3) -> nn.Module:
    """
    A simple loader for some of the nonlinear activation functions
    Parameters
    Parameters
    ----------
    activation_func_name : str
        DESCRIPTION.
    num_parameters : int
        I don't know how to use this yet.
    Returns
    -------
    nn.Module
        The activation function.
    """
    funcs = dict(relu       = nn.ReLU(),
                 selu       = nn.SELU(),
                 elu        = nn.ELU(),
                 celu       = nn.CELU(),
                 gelu       = nn.GELU(),
                 silu       = nn.SiLU(),
                 sigmoid    = nn.Sigmoid(),
                 tanh       = nn.Tanh(),
                 linear     = None,
                 leaky_relu = nn.LeakyReLU(),
                 hardshrink = nn.Hardshrink(lambd = .1),
                 softshrink = nn.Softshrink(lambd = .1),
                 tanhshrink = nn.Tanhshrink(),
                 # weight decay should not be used when learning aa for good performance.
                 prelu      = nn.PReLU(num_parameters=num_parameters,),
                 )
    return funcs[activation_func_name]

def compute_image_loss(image_loss_func:Callable,
                       image_category:Tensor,
                       labels:Tensor,
                       device:str,
                       n_noise:int      = 0,
                       num_classes:int  = 2,
                       ) -> Tensor:
    """
    Compute the loss of predicting the image categories
    Parameters
    ----------
    image_loss_func : Callable
        DESCRIPTION.
    image_category : Tensor
        DESCRIPTION.
    labels : Tensor
        DESCRIPTION.
    device : str
        DESCRIPTION.
    n_noise : int, optional
        DESCRIPTION. The default is 0.
    num_classes : int, optional
        DESCRIPTION. The default is 10.
    Returns
    -------
    image_loss: Tensor
        DESCRIPTION.
    """
    if "Binary Cross Entropy" in image_loss_func.__doc__:
        labels = labels.float()
        if n_noise > 0:
            noisy_labels    = torch.ones(labels.shape) * (1/num_classes)
            noisy_labels    = noisy_labels[:n_noise]
            labels          = torch.cat([labels.to(device),noisy_labels.to(device)])
        # print(image_category.shape,labels.shape)
        image_loss = image_loss_func(image_category.to(device),
                                     labels.view(image_category.shape).to(device)
                                     )
    elif "negative log likelihood loss" in image_loss_func.__doc__:
        temp                    = torch.vstack([1-labels.argmax(1),labels.argmax(1)]).T
        labels                  = temp.detach().clone()
        labels                  = labels.argmax(1)
        labels                  = labels.long()
        if n_noise > 0:
            image_category      = image_category[:-n_noise]
        image_loss              = image_loss_func(torch.log(image_category).to(device),
                                                  labels.to(device))
    elif "Kullback-Leibler divergence loss" in image_loss_func.__doc__:
        image_loss_func.reduction  = 'batchmean'
        image_loss_func.log_target = True
        if n_noise > 0:
            noisy_labels    = torch.ones(labels.shape) * (1/num_classes)
            noisy_labels    = noisy_labels[:n_noise]
            labels          = torch.cat([labels.to(device),noisy_labels.to(device)])
        image_loss = image_loss_func(torch.log(image_category).to(device),
                                     labels.to(device))
    return image_loss

invTrans = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
                                                     std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                                transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],
                                                     std = [ 1., 1., 1. ]),
                               ])
normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])
def train_valid_cnn_classifier(net:nn.Module,
                               dataloader:torch.utils.data.dataloader.DataLoader,
                               optimizer:torch.optim,
                               classification_loss:nn.Module,
                               idx_epoch:int    = 0,
                               device           = 'cpu',
                               train:bool       = True,
                               verbose          = 0,
                               n_noise:int      = 0,
                               sleep_time:int   = 0,
                               ):
    """
    

    Parameters
    ----------
    net : nn.Module
        DESCRIPTION.
    dataloader : torch.utils.data.dataloader.DataLoader
        DESCRIPTION.
    optimizer : torch.optim
        DESCRIPTION.
    classification_loss : nn.Module
        DESCRIPTION.
    idx_epoch : int, optional
        DESCRIPTION. The default is 0.
    device : string or torch.device, optional
        DESCRIPTION. The default is 'cpu'.
    train : bool, optional
        DESCRIPTION. The default is True.
    verbose : TYPE, optional
        DESCRIPTION. The default is 0.
     n_noise : int, optional
        DESCRIPTION. The default is 0.
    sleep_time : int, optional
        DESCRIPTION. The default is 0.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    if train:
        net.train(True)
    else:
        net.eval()
    loss  = 0.
    iterator    = tqdm(enumerate(dataloader))
    for idx_batch,(batch_features,batch_labels) in iterator:
        if n_noise > 0:
            noise_generator = torch.distributions.normal.Normal(batch_features.mean(),
                                                                batch_features.std(),)
            noise_features = noise_generator.sample(batch_features.shape)[:n_noise]
            # temp = invTrans(batch_features[:n_noise])
            # idx_pixels = torch.where(temp == 1)
            # temp = invTrans(noise_features)
            # temp[idx_pixels] = 1
            # noise_features = normalizer(temp)
            batch_features = torch.cat([batch_features,noise_features])
        # zero grad
        optimizer.zero_grad()
        # forward pass
        if train:
            (batch_extract_features,
             batch_hidden_representation,
             batch_prediction) = net(batch_features.to(device))
        else:
            with torch.no_grad():
                (batch_extract_features,
                 batch_hidden_representation,
                 batch_prediction) = net(batch_features.to(device))
        # compute loss
        batch_loss = compute_image_loss(classification_loss,
                                        batch_prediction,
                                        batch_labels,
                                        device,
                                        n_noise = n_noise,
                                        )
        if train:
            # backprop
            batch_loss.backward()
            # modify weights
            optimizer.step()
        # record the loss of a mini-batch
        loss += batch_loss.item()
        if verbose > 0:
            iterator.set_description(f'epoch {idx_epoch+1:3.0f}-{idx_batch + 1:4.0f}/{100*(idx_batch+1)/len(dataloader):2.3f}%,loss = {loss/(idx_batch+1):2.6f}')
    if sleep_time > 0:
        sleep(sleep_time)
    return net,loss/(idx_batch+1)

def train_valid_betting_network(net:nn.Module,
                                dataloader:torch.utils.data.dataloader.DataLoader,
                                optimizer:torch.optim,
                                classification_loss:nn.Module,
                                betting_loss:nn.Module,
                                idx_epoch:int    = 0,
                                device           = 'cpu',
                                train:bool       = True,
                                verbose          = 0,
                                sleep_time:int   = 0,
                                ):
    """
    

    Parameters
    ----------
    net : nn.Module
        DESCRIPTION.
    dataloader : torch.utils.data.dataloader.DataLoader
        DESCRIPTION.
    optimizer : torch.optim
        DESCRIPTION.
    classification_loss : nn.Module
        DESCRIPTION.
    betting_loss : nn.Module
        DESCRIPTION.
    idx_epoch : int, optional
        DESCRIPTION. The default is 0.
    device : string or torch.device, optional
        DESCRIPTION. The default is 'cpu'.
    train : bool, optional
        DESCRIPTION. The default is True.
    verbose : TYPE, optional
        DESCRIPTION. The default is 0.
    sleep_time : int, optional
        DESCRIPTION. The default is 0.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    if train:
        net.train(True)
    else:
        net.eval()
    loss        = 0.
    iterator    = tqdm(enumerate(dataloader))
    for idx_batch,(batch_image1,batch_label1,
                   batch_image2,batch_label2,
                   batch_correct_bet) in iterator:
        # zero grad
        optimizer.zero_grad()
        # forward pass
        if train:
            (batch_betting_output,
             (batch_features1,
              batch_hidden_representation1,
              batch_prediction1),
             (batch_features2,
              batch_hidden_representation2,
              batch_prediction2)) = net(batch_image1.to(device),
                                     batch_image2.to(device))
        else:
            with torch.no_grad():
                (batch_betting_output,
                 (batch_features1,
                  batch_hidden_representation1,
                  batch_prediction1),
                 (batch_features2,
                  batch_hidden_representation2,
                  batch_prediction2)) = net(batch_image1.to(device),
                                         batch_image2.to(device))
        # compute loss
        batch_loss = compute_image_loss(
                                    betting_loss,
                                    batch_betting_output,
                                    batch_correct_bet,
                                    device,
                                    )
        # loss_image = compute_image_loss(
        #                             classification_loss,
        #                             batch_prediction1,
        #                             batch_label1,
        #                             device,
        #                             )
        # loss_image += compute_image_loss(
        #                             classification_loss,
        #                             batch_prediction2,
        #                             batch_label2,
        #                             device,
        #                             )
        # batch_loss += loss_image / 2.0
        if train:
            # backprop
            batch_loss.backward()
            # modify weights
            optimizer.step()
        # record the loss of a mini-batch
        loss += batch_loss.item()
        if verbose > 0:
            iterator.set_description(f'epoch {idx_epoch+1:3.0f}-{idx_batch + 1:4.0f}/{100*(idx_batch+1)/len(dataloader):2.3f}%,loss = {loss/(idx_batch+1):2.6f}')
    if sleep_time > 0:
        sleep(sleep_time)
    return net,loss/(idx_batch+1)

def determine_training_stops(net,
                             idx_epoch:int,
                             warmup_epochs:int,
                             valid_loss:Tensor,
                             counts: int        = 0,
                             device             = 'cpu',
                             best_valid_loss    = np.inf,
                             tol:float          = 1e-4,
                             f_name:str         = 'temp.h5',
                             ) -> Tuple[Tensor,int]:
    """
    A function in validation determining whether to stop training
    It only works after the warmup 
    Parameters
    ----------
    net : nn.Module
        DESCRIPTION.
    idx_epoch : int
        DESCRIPTION.
    warmup_epochs : int
        DESCRIPTION.
    valid_loss : Tensor
        DESCRIPTION.
    counts : int, optional
        DESCRIPTION. The default is 0.
    device : TYPE, optional
        DESCRIPTION. The default is 'cpu'.
    best_valid_loss : TYPE, optional
        DESCRIPTION. The default is np.inf.
    tol : float, optional
        DESCRIPTION. The default is 1e-4.
    f_name : str, optional
        DESCRIPTION. The default is 'temp.h5'.
    Returns
    -------
    best_valid_loss: Tensor
        DESCRIPTION.
    counts:int
        used for determine when to stop training
    """
    if idx_epoch >= warmup_epochs: # warming up
        temp = valid_loss
        if np.logical_and(temp < best_valid_loss,np.abs(best_valid_loss - temp) >= tol):
            best_valid_loss = valid_loss
            torch.save(net.state_dict(),f_name)# why do i need state_dict()?
            counts = 0
        else:
            counts += 1
    return best_valid_loss,counts

def train_valid_loop(net:nn.Module,
                     dataloader_train:torch.utils.data.dataloader.DataLoader,
                     dataloader_valid:torch.utils.data.dataloader.DataLoader,
                     optimizer:torch.optim,
                     classification_loss= nn.BCELoss(),
                     betting_loss       = nn.BCELoss(),
                     scheduler          = None,
                     device             = 'cpu',
                     verbose            = 0,
                     n_epochs:int       = 1000,
                     warmup_epochs:int  = 5,
                     patience:int       = 5,
                     tol:float          = 1e-4,
                     f_name:str         = 'temp.h5',
                     model_stage:str    = 'cnn',
                     n_noise:int        = 0,
                     sleep_time:int     = 0,
                     ):
    """
    

    Parameters
    ----------
    net : nn.Module
        DESCRIPTION.
    dataloader_train : torch.utils.data.dataloader.DataLoader
        DESCRIPTION.
    dataloader_valid : torch.utils.data.dataloader.DataLoader
        DESCRIPTION.
    optimizer : torch.optim
        DESCRIPTION.
    classification_loss : TYPE, optional
        DESCRIPTION. The default is nn.BCELoss().
    betting_loss : TYPE, optional
        DESCRIPTION. The default is nn.BCELoss().
    scheduler : TYPE, optional
        DESCRIPTION. The default is None.
    device : TYPE, optional
        DESCRIPTION. The default is 'cpu'.
    verbose : TYPE, optional
        DESCRIPTION. The default is 0.
    n_epochs : int, optional
        DESCRIPTION. The default is 1000.
    warmup_epochs : int, optional
        DESCRIPTION. The default is 5.
    patience : int, optional
        DESCRIPTION. The default is 5.
    tol : float, optional
        DESCRIPTION. The default is 1e-4.
    f_name : str, optional
        DESCRIPTION. The default is 'temp.h5'.
    model_stage : str, optional
        DESCRIPTION. The default is 'cnn'.
    n_noise : int, optional
        DESCRIPTION. The default is 0.
    sleep_time : int, optional
        DESCRIPTION. The default is 0.
        
    Returns
    -------
    net : TYPE
        DESCRIPTION.
    losses : TYPE
        DESCRIPTION.

    """
    best_valid_loss         = np.inf
    losses                  = []
    counts                  = 0
    for idx_epoch in range(n_epochs):
        print('\ntraining...')
        if model_stage == 'cnn':
            net,train_loss = train_valid_cnn_classifier(net,
                                                        dataloader_train,
                                                        optimizer,
                                                        classification_loss,
                                                        idx_epoch   = idx_epoch,
                                                        device      = device,
                                                        train       = True,
                                                        verbose     = verbose,
                                                        n_noise     = n_noise,
                                                        sleep_time  = sleep_time,
                                                        )
            print('\nvalidating...')
            net,valid_loss = train_valid_cnn_classifier(net,
                                                        dataloader_valid,
                                                        optimizer,
                                                        classification_loss,
                                                        idx_epoch   = idx_epoch,
                                                        device      = device,
                                                        train       = False,
                                                        verbose     = verbose,
                                                        sleep_time  = sleep_time,
                                                        )
        elif model_stage == 'betting':
            net,train_loss = train_valid_betting_network(net,
                                                         dataloader_train,
                                                         optimizer,
                                                         classification_loss,
                                                         betting_loss,
                                                         idx_epoch   = idx_epoch,
                                                         device      = device,
                                                         train       = True,
                                                         verbose     = verbose,
                                                         sleep_time  = sleep_time,
                                                         )
            print('\nvalidating...')
            net,valid_loss = train_valid_betting_network(net,
                                                         dataloader_valid,
                                                         optimizer,
                                                         classification_loss,
                                                         betting_loss,
                                                         idx_epoch   = idx_epoch,
                                                         device      = device,
                                                         train       = False,
                                                         verbose     = verbose,
                                                         sleep_time  = sleep_time,
                                                         )
        if scheduler != None and idx_epoch >= warmup_epochs:
            scheduler.step(valid_loss)
        best_valid_loss,counts = determine_training_stops(net,
                                                          idx_epoch,
                                                          warmup_epochs,
                                                          valid_loss,
                                                          counts            = counts,
                                                          device            = device,
                                                          best_valid_loss   = best_valid_loss,
                                                          tol               = tol,
                                                          f_name            = f_name,)
        if counts >= patience:#(len(losses) > patience) and (len(set(losses[-patience:])) == 1):
            break
        else:
            print(f'\nepoch {idx_epoch + 1}, best valid loss = {best_valid_loss:.8f},count = {counts}')
        losses.append(best_valid_loss)
    return net,losses

def A(y_true:np.ndarray,y_pred:np.ndarray) -> float:
    """
    

    Parameters
    ----------
    y_true : np.ndarray
        DESCRIPTION.
    y_pred : np.ndaray
        DESCRIPTION.

    Returns
    -------
    float
        DESCRIPTION.

    """
    fpr,tpr,thres = roc_curve(y_true, y_pred)
    fpr = fpr[1]
    tpr = tpr[1]
    if  fpr > tpr:
        A = 1/2 + ((fpr - tpr)*(1 + fpr - tpr))/((4 * fpr)*(1 - tpr))
    elif fpr <= tpr:
        A = 1/2 + ((tpr - fpr)*(1 + tpr - fpr))/((4 * tpr)*(1 - fpr))
    return A

def compute_A(h:float,f:float) -> float:
    """
    

    Parameters
    ----------
    h : float
        DESCRIPTION.
    f : float
        DESCRIPTION.

    Returns
    -------
    float
        DESCRIPTION.

    """
    if (.5 >= f) and (h >= .5):
        a = .75 + (h - f) / 4 - f * (1 - h)
    elif (h >= f) and (.5 >= h):
        a = .75 + (h - f) / 4 - f / (4 * h)
    else:
        a = .75 + (h - f) / 4 - (1 - h) / (4 * (1 - f))
    return a
def check_nan(temp):
    if np.isnan(temp[1]):
        return 0
    else:
        return temp[1]
def binary_response_score_func(y_true:np.ndarray, y_pred:np.ndarray):
    """
    

    Parameters
    ----------
    y_true : np.ndarray
        DESCRIPTION.
    y_pred : np.ndarray
        DESCRIPTION.

    Returns
    -------
    a : TYPE
        DESCRIPTION.

    """
    fpr,tpr,thresholds = roc_curve(y_true, y_pred)
    tpr = check_nan(tpr)
    fpr = check_nan(fpr)
    a = compute_A(tpr,fpr)
    return a
def collect_data_on_test(y_true:np.ndarray,y_pred:np.ndarray,
                         grid_size:int,) -> pd.core.frame.DataFrame:
    """
    

    Parameters
    ----------
    y_true : np.ndarray
        DESCRIPTION.
    y_pred : np.ndarray
        DESCRIPTION.
    grid_size : int
        DESCRIPTION.

    Returns
    -------
    df_temp : TYPE
        DESCRIPTION.

    """
    groups = ['{}-{}'.format(*np.sort(row)) for row in y_true]
    df_temp = pd.DataFrame(y_true,columns = ['class_0','class_1'])
    df_temp['prob_0'] = y_pred[:,0]
    df_temp['prob_1'] = y_pred[:,1]
    df_temp['correct answer'] = np.array(df_temp['class_1'].values >= 0.5,dtype = np.int64)
    df_temp['response'] = np.array(df_temp['prob_1'].values >= 0.5,dtype = np.int64)
    df_temp['group'] = groups
    df_temp['grid_size'] = grid_size
    return df_temp

###############################################################################
# old functions for loading old benckmarks ####################################
###############################################################################
class CustomImageDataset(Dataset):
    """
    
    """
    def __init__(self,
                 img_dir:str,
                 transform:torchvision.transforms.transforms.Compose        = None,
                 sparse_target:bool                                         = True):
        self.img_dir            = img_dir
        self.transform          = transform
        self.sparse_target      = sparse_target
        
        self.images = glob(os.path.join(img_dir,'*','*.jpg'))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path    = self.images[idx]
        image,label = lock_and_load(img_path,
                                    self.transform,
                                    self.sparse_target,
                                    )
        return image, label

def lock_and_load(img_path:str,
                  transformer_steps:torchvision.transforms.transforms.Compose,
                  sparse_target:bool):
    """
    

    Parameters
    ----------
    img_path : str
        DESCRIPTION.
    transformer_steps : torchvision.transforms.transforms.Compose
        DESCRIPTION.
    sparse_target : bool
        DESCRIPTION.

    Returns
    -------
    image : TYPE
        DESCRIPTION.
    label : TYPE
        DESCRIPTION.

    """
    image       = pil_image.open(img_path)
    label       = img_path.split('/')[-2]
    label       = torch.tensor([int(label.split('-')[0]),int(label.split('-')[1])])
    label       = label / label.sum() # hard max
    if sparse_target and label.detach().numpy()[0] != 0.5:
        label   = torch.vstack([1 - label.argmax(),label.argmax()]).T
    
    image = transformer_steps(image)
    return image,label

class betting_network_dataloader(Dataset):
    def __init__(self, 
                 dataframe:pd.core.frame.DataFrame,
                 transformer_steps = None,
                 noise_level:float = 0.,
                 sparse_target:bool = False,
                 ):
        """
        

        Parameters
        ----------
        dataframe : pd.core.frame.DataFrame
            DESCRIPTION.
        transformer_steps : TYPE, optional
            DESCRIPTION. The default is None.
        noise_level : float, optional
            DESCRIPTION. The default is 0..
        sparse_target : bool, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        None.

        """
        self.dataframe = dataframe
        if transformer_steps == None:
            self.transformer_steps = concatenate_transform_steps(
                                    128,
                                    noise_level = noise_level,)
        else:
            self.transformer_steps = transformer_steps
        self.noise_level = noise_level,
        self.sparse_target = sparse_target
    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        image1,label1 = lock_and_load(row['image1'],
                                      self.transformer_steps,
                                      self.sparse_target,)
        image2,label2 = lock_and_load(row['image2'],
                                      self.transformer_steps,
                                      self.sparse_target,)
        if self.sparse_target:
            correct_bet = torch.tensor([row['sparse_label'],1-row['sparse_label']])
        else:
            correct_bet = torch.tensor([row['correct_bet1'],
                                        row['correct_bet2']])
        return image1,label1,image2,label2,correct_bet

def append_to_list(df:pd.core.frame.DataFrame,
                   image1,
                   image2):
    """
    

    Parameters
    ----------
    df : pd.core.frame.DataFrame
        DESCRIPTION.
    image1 : TYPE
        DESCRIPTION.
    image2 : TYPE
        DESCRIPTION.

    Returns
    -------
    df : TYPE
        DESCRIPTION.

    """
    df['image1'].append(image1)
    df['image2'].append(image2)
    return df

def build_betting_dataloader(img_dir:str,
                             transformer_steps  = None,
                             batch_size:int     = 16,
                             shuffle:bool       = True,
                             num_workers:int    = 2,
                             sparse_target:bool = False,
                             noise_level:float  = 0.,
                             memory_samples:int = 10,
                             random_state       = None,
                             ):
    """
    

    Parameters
    ----------
    img_dir : str
        DESCRIPTION.
    transformer_steps : TYPE, optional
        DESCRIPTION. The default is None.
    batch_size : int, optional
        DESCRIPTION. The default is 16.
    shuffle : bool, optional
        DESCRIPTION. The default is True.
    num_workers : int, optional
        DESCRIPTION. The default is 2.
    sparse_target : bool, optional
        DESCRIPTION. The default is False.
    noise_level : float, optional
        DESCRIPTION. The default is 0..
    memory_samples : int, optional
        DESCRIPTION. The default is 100.
    random_state : None or int, optional
        DESCRIPTION. The default is None

    Returns
    -------
    dataloader : TYPE
        DESCRIPTION.

    """
    
    all_images = glob(os.path.join(img_dir,'*','*.jpg'))
    df_images = pd.DataFrame(all_images,columns = ['image_path'])
    df_images['group'] = df_images['image_path'].apply(lambda x:x.split('/')[-2])
    df_images['group'] = df_images['group'].apply(compute_ratio_from_group)
    df = dict(image1=[],image2=[])
    for group1,df_sub1 in df_images.groupby('group'):
        for group2,df_sub2 in df_images.groupby('group'):
            if group1 != group2:
                if memory_samples != None:
                    df_sub1 = df_sub1.sample(memory_samples,replace = False,random_state = random_state)
                    df_sub2 = df_sub2.sample(memory_samples,replace = False,random_state = random_state)
                pairs = product(df_sub1['image_path'],df_sub2['image_path'])
                [append_to_list(df, image1, image2) for image1,image2 in pairs]
                
    df = pd.DataFrame(df)
    df['group1'] = df['image1'].apply(lambda x:x.split('/')[-2])
    df['group2'] = df['image2'].apply(lambda x:x.split('/')[-2])
    
    df['image1_ratio1'] = df['group1'].apply(lambda x:float(x.split('-')[0]))
    df['image1_ratio2'] = df['group1'].apply(lambda x:float(x.split('-')[1]))
    df['image2_ratio1'] = df['group2'].apply(lambda x:float(x.split('-')[0]))
    df['image2_ratio2'] = df['group2'].apply(lambda x:float(x.split('-')[1]))
    
    group1 = np.sort(df[['image1_ratio1', 'image1_ratio2']].values,axis = 1)
    group1 = group1[:,0] / group1[:,1]
    group2 = np.sort(df[['image2_ratio1', 'image2_ratio2']].values,axis = 1)
    group2 = group2[:,0] / group2[:,1]
    
    df['difficulty1'] = group1
    df['difficulty2'] = group2
    
    df['sparse_label'] = np.array(df['difficulty1'].values > df['difficulty2'].values,
                           dtype = np.float64
                           )
    
    temp = df[['difficulty1','difficulty2']].values
    temp = temp / temp.sum(1).reshape(-1,1)
    temp = 1 - temp
    df['correct_bet1'] = temp[:,0]
    df['correct_bet2'] = temp[:,1]
    
    dataset             = betting_network_dataloader(
                                            dataframe = df,
                                            transformer_steps = transformer_steps,
                                            noise_level = noise_level,
                                            sparse_target = sparse_target,
                                            )
    dataloader          = DataLoader(dataset,
                                     batch_size         = batch_size,
                                     shuffle            = shuffle,
                                     num_workers        = num_workers,
                                     )
    return dataloader

def compute_ratio_from_group(x):
    """
    This function can only apply to dataframes

    Parameters
    ----------
    x : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    x = np.sort([float(x.split('-')[0]), float(x.split('-')[1])])
    return x[0] / x[1]

def build_dataloader(img_dir:str,
                     transformer_steps  = None,
                     batch_size:int     = 16,
                     shuffle:bool       = True,
                     num_workers:int    = 2,
                     sparse_target:bool = False,
                     ):
    """
    build a dataloader for batch feeding

    Parameters
    ----------
    img_dir : str
        DESCRIPTION.
    transformer_steps : TYPE, optional
        DESCRIPTION. The default is None.
    batch_size : int, optional
        DESCRIPTION. The default is 16.
    shuffle : bool, optional
        DESCRIPTION. The default is True.
    num_workers : int, optional
        DESCRIPTION. The default is 2.
    sparse_target : bool, optional
        DESCRIPTION. The default is False.
    Returns
    -------
    dataloader : TYPE
        DESCRIPTION.

    """
    dataset             = CustomImageDataset(img_dir       = img_dir,
                                             transform     = transformer_steps,
                                             sparse_target = sparse_target,
                                             )
    dataloader          = DataLoader(dataset,
                                     batch_size         = batch_size,
                                     shuffle            = shuffle,
                                     num_workers        = num_workers,
                                     )
    return dataloader

##############################################################################

if __name__ == "__main__":
    pass