#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  3 13:28:08 2022

@author: ningmei
"""
import numpy as np
import torch,os
from torch import nn
from torchvision import models

from utils_deep import candidates,define_type

from typing import List, Callable, Union, Any, TypeVar, Tuple
###############################################################################
Tensor = TypeVar('torch.tensor')
###############################################################################

def create_hidden_layer(
                        layer_type:str = 'linear',
                        input_units:int = 2,
                        output_units:int = 2,
                        output_dropout:float = 0.,
                        output_activation:nn.Module = nn.ReLU(),
                        out_channels:int = 1280,
                        kernel_size:int = 3,
                        device = 'cpu',
                        ):
    """
    create a linear hidden layer
    
    Inputs
    ---
    layer_type: str, default = "linear", well, I want to implement recurrent layers
    input_units: int, in_features of the layer
    output_units: int, out_features of the layer
    output_drop: float, between 0 and 1
    output_activation: nn.Module or None, torch activation functions, None = linear activation function
    device: str or torch.device
    
    Outputs
    ---
    hidden_layer: nn.Sequential module
    """
    if layer_type == 'linear':
        latent_layer     = nn.Linear(input_units,output_units).to(device)
        dropout          = nn.Dropout(p = output_dropout).to(device)
        
        if output_activation is not None:
            hidden_layer = nn.Sequential(
                                latent_layer,
                                nn.BatchNorm1d(output_units,),
                                output_activation,
                                dropout)
        else:
            hidden_layer = nn.Sequential(
                                latent_layer,
                                nn.BatchNorm1d(output_units),
                                dropout)
        return hidden_layer
    elif layer_type == 'convolutional':
        latent_layer1    = nn.Conv2d(input_units, out_channels, kernel_size)
        latent_layer2    = nn.Conv2d(out_channels,out_channels, kernel_size)
        dropout          = nn.Dropout(p = output_dropout).to(device)
        if output_activation is not None:
            hidden_layer = nn.Sequential(
                                latent_layer1,
                                nn.BatchNorm2d(num_features = out_channels,),
                                output_activation,
                                latent_layer2,
                                nn.BatchNorm2d(num_features = out_channels,),
                                output_activation,
                                dropout)
        else:
            hidden_layer = nn.Sequential(
                                latent_layer1,
                                nn.BatchNorm2d(num_features = out_channels,),
                                output_activation,
                                latent_layer2,
                                dropout)
        return hidden_layer
        
    elif layer_type == 'recurrent':
        raise NotImplementedError
    else:
        raise NotImplementedError

def CNN_feature_extractor(pretrained_model_name,
                          retrain_encoder:bool,
                          in_shape:Tuple = (1,3,128,128),
                          device = 'cpu',
                          ):
    """
    
    Parameters
    ----------
    pretrained_model_name : TYPE
        DESCRIPTION.
    retrain_encoder : bool
        DESCRIPTION.
    in_shape : Tuple, optional
        DESCRIPTION. The default is (1,3,128,128).
    device : str or torch.device, optional
        DESCRIPTION. The default is 'cpu'.
    Returns
    -------
    in_features : int
        DESCRIPTION.
    feature_extractor : nn.Module
        DESCRIPTION.
    """
    pretrained_model       = candidates(pretrained_model_name)
    # freeze the pretrained model
    if not retrain_encoder:
        for params in pretrained_model.parameters():
            params.requires_grad = False
    # get the dimensionof the CNN features
    if define_type(pretrained_model_name) == 'simple':
        in_features                = nn.AdaptiveAvgPool2d((1,1))(
                                            pretrained_model.features(
                                                    torch.rand(*in_shape))).shape[1]
        feature_extractor          = easy_model(pretrained_model = pretrained_model,).to(device)
    elif define_type(pretrained_model_name) == 'resnet':
        in_features                = pretrained_model.fc.in_features
        feature_extractor          = resnet_model(pretrained_model = pretrained_model,).to(device)
    return in_features,feature_extractor

def cnn_block(in_channels:int = 3,
              out_channels:int = 256,
              kernel_size:Tuple = (3,3),
              stride:Tuple = (1,1),
              padding:Tuple = (1,1),
              pool_kernel_size:int = 2,
              pool_stride:int = 1,
              pool_padding:int = 0,
              ) -> nn.Module:
    """
    

    Parameters
    ----------
    in_channels : int, optional
        DESCRIPTION. The default is 3.
    out_channels : int, optional
        DESCRIPTION. The default is 256.
    kernel_size : Tuple, optional
        DESCRIPTION. The default is (3,3).
    stride : Tuple, optional
        DESCRIPTION. The default is (1,1).
    padding : Tuple, optional
        DESCRIPTION. The default is (1,1).
    pool_kernel_size : int, optional
        DESCRIPTION. The default is 2.
    pool_stride : int, optional
        DESCRIPTION. The default is 1.
    pool_padding : int, optional
        DESCRIPTION. The default is 0.

    Returns
    -------
    block : nn.Module
        DESCRIPTION.

    """
    cnn = nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding,)
    batchnorm = nn.BatchNorm2d(out_channels,)
    activation = nn.ReLU(inplace = True)
    pooling = nn.MaxPool2d(kernel_size = pool_kernel_size,
                           stride = pool_stride,
                           padding = pool_padding,
                           )
    block = nn.Sequential(cnn,batchnorm,activation,pooling)
    return block
    
##############################################################################
# class CNN_model_from_pretrain_vgg(nn.Module):
#     def __init__(self,
#                  n_cnn_layers:int = 1,
#                  in_shape:Tuple = (1,3,128,128),
#                  hidden_layer_type:str = 'linear',
#                  hidden_layer_size:int = 300,
#                  hidden_activation:nn.Module = nn.ReLU(),
#                  hidden_dropout:float = 0,
#                  output_layer_size:int = 2,
#                  device = 'cpu',
#                  ):
#         super(CNN_model_from_pretrain_vgg,self).__init()
#         torch.manual_seed(12345)
#         pretrained_vgg = models.vgg19_bn(weights              = "IMAGENET1K_V1",
#                                          progress              = False,)


class CNN_model(nn.Module):
    def __init__(self,
                 cnn_layers:List = [256,],
                 in_shape = (1,3,128,128),
                 hidden_layer_type:str = 'linear',
                 hidden_layer_size:int = 300,
                 hidden_activation:nn.Module = nn.ReLU(),
                 hidden_dropout:float = 0,
                 output_layer_size:int = 2,
                 device = 'cpu',
                 ):
        super(CNN_model,self).__init__()
        torch.manual_seed(12345)
        self.cnn_layers = cnn_layers
        self.device = device
        
        self.first_cnn = cnn_block(in_channels=3,
                                   out_channels=cnn_layers[0],
                                   ).to(self.device)
        if len(self.cnn_layers) > 1:
            self.more_cnn = []
            for in_channels,out_channels in zip(cnn_layers[:-1],
                                                cnn_layers[1:], ):
                self.more_cnn.append(cnn_block(in_channels,out_channels).to(self.device))
            self.more_cnn = nn.ModuleList(self.more_cnn)
        
        self.global_pooling = nn.AdaptiveAvgPool2d((1,1)).to(self.device)
        self.hidden_layer = create_hidden_layer(
                                    layer_type          = hidden_layer_type,
                                    input_units         = cnn_layers[-1],
                                    output_units        = hidden_layer_size,
                                    output_activation   = hidden_activation,
                                    output_dropout      = hidden_dropout,
                                    device              = device,
                                    ).to(self.device)
        self.output_layer = nn.Sequential(nn.Linear(hidden_layer_size,output_layer_size),
                                          nn.Softmax(dim = -1)
                                          ).to(self.device)
    def forward(self,x):
        cnn_features = self.first_cnn(x)
        if len(self.cnn_layers) > 1:
            for cnn_feature_layer in self.more_cnn:
                cnn_features = cnn_feature_layer(cnn_features)
        cnn_features = self.global_pooling(cnn_features)
        features = torch.squeeze(torch.squeeze(cnn_features,3),2)
        hidden_representation = self.hidden_layer(features)
        prediction = self.output_layer(hidden_representation)
        return features,hidden_representation,prediction

class easy_model(nn.Module):
    """
    Models are not created equally
    Some pretrained models are composed by a {feature} and a {classifier} component
    thus, they are very easy to modify and transfer learning
    Inputs
    --------------------
    pretrained_model: nn.Module, pretrained model object
    in_shape: Tuple, input image dimensions (1,n_channels,height,width)
    Outputs
    --------------------
    out: torch.tensor, the pooled CNN features
    """
    def __init__(self,
                 pretrained_model:nn.Module,
                 in_shape = (1,3,128,128),
                 ) -> None:
        super(easy_model,self).__init__()
        torch.manual_seed(12345)
        self.in_features    = nn.AdaptiveAvgPool2d((1,1))(pretrained_model.features(torch.rand(*in_shape))).shape[1]
        
        # print(f'feature dim = {self.in_features}')
        self.features       = pretrained_model.features
    def forward(self,x:Tensor) -> Tensor:
        out                 = self.features(x)
        return out

class resnet_model(nn.Module):
    """
    Models are not created equally
    Some pretrained models are composed by a {feature} and a {fc} component
    thus, they are very easy to modify and transfer learning
    Inputs
    --------------------
    pretrained_model: nn.Module, pretrained model object
    Outputs
    --------------------
    out: torch.tensor, the pooled CNN features
    """

    def __init__(self,
                 pretrained_model:nn.Module,
                 ) -> None:
        super(resnet_model,self).__init__()
        torch.manual_seed(12345)
        
        self.in_features    = pretrained_model.fc.in_features
        res_net             = torch.nn.Sequential(*list(pretrained_model.children())[:-2])
        # print(f'feature dim = {self.in_features}')
        
        self.features       = res_net
        
    def forward(self,x:Tensor) -> Tensor:
        out                 = self.features(x)
        return out

def add_noise_to_layer(x:Tensor,
                       noise_level:float = 0,
                       noise_type:str = 'gaussian',
                       device = 'cpu',
                       ):
    """
    add noise to the a layer output

    Parameters
    ----------
    x : Tensor
        DESCRIPTION.
    noise_level : float, optional
        DESCRIPTION. The default is 0.
    noise_type : str, optional
        DESCRIPTION. The default is 'gaussian'. Another option is 'reduction'
    device : TYPE, optional
        DESCRIPTION. The default is 'cpu'.

    Returns
    -------
    x : TYPE
        DESCRIPTION.

    """
    if noise_type == 'gaussian':
        # get the min and max of the layer output
        x_min,x_max = x.min(),x.max()
        # standard normal noise
        noise_generator = torch.distributions.Normal(0,1)
        # generate noise
        noise = noise_generator.sample(x.shape).to(device)
        # add noise
        x = x * (1 - noise_level) + noise * noise_level
        # clip the noise output use min max
        x = torch.clamp(x,x_min,x_max,)
    elif noise_type == 'reduction': # reduce the signal strength
        x = x * noise_level
    else:
        raise NotImplementedError
    return x

class perceptual_network(nn.Module):
    def __init__(self,
                 pretrained_model_name:str      = 'mobilenet',
                 hidden_layer_size:int          = 300,
                 hidden_activation:nn.Module    = nn.ReLU(),
                 hidden_dropout:float           = 0.,
                 hidden_layer_type:str          = 'linear',
                 output_layer_size:int          = 2,
                 confidence_layer_size:int      = 2,
                 in_shape:Tuple                 = (1,3,128,128),
                 retrain_encoder:bool           = False,
                 device                         = 'cpu',
                 batch_size:int                 = 8,
                 ):
        super(perceptual_network,self).__init__()
        torch.manual_seed(12345)
        self.batch_size = batch_size
        self.device     = device
        # 1. calculate the input feature size for the dense layer
        # 2. extract the CNN layers from the pretrained network
        # 3. decide whether to fine tune the CNN layers
        in_features,feature_extractor    = CNN_feature_extractor(
                                            pretrained_model_name   = pretrained_model_name,
                                            retrain_encoder         = retrain_encoder,
                                            in_shape                = in_shape,
                                            device                  = device,
                                                               )
        # unblock the sequential model
        layers_of_feature_extractor      = list(feature_extractor.features.children())
        self.layers_of_feature_extractor = nn.ModuleList(layers_of_feature_extractor).to(device)
        # self.feature_extractor           = feature_extractor.to(device)
        # adaptive pooling layer to vectorize the CNN layer outpus
        self.avgpool                     = nn.AdaptiveAvgPool2d((1,1))
        # hidden layer
        self.hidden_layer                = create_hidden_layer(
                                            layer_type          = hidden_layer_type,
                                            input_units         = in_features,
                                            output_units        = hidden_layer_size,
                                            output_activation   = hidden_activation,
                                            output_dropout      = hidden_dropout,
                                            device              = device,
                                            ).to(device)
        # decision layer
        self.decision_layer                = nn.Sequential(
                                            nn.Linear(hidden_layer_size,output_layer_size),
                                            nn.Softmax(dim = -1)
                                            ).to(device)
        self.confidence_layer            = nn.Sequential(
                                            nn.Linear(hidden_layer_size,confidence_layer_size),
                                            nn.Softmax(dim = -1)
                                            ).to(device)
    
    def forward(self,x,
                idx_layer:int       = 0, 
                noise_level:float   = 0,
                noise_type:str      = 'gaussian',
                ):
        """
        The forward function can be used for adding noise to the 
        convolutional layers and the dense layers
        
        if the "idx_layer" matches the index of the convolutional layer, and
        the value is positive, we can add noise to the correspondent output
        of the convolutional layer
        
        if the "idex_layer" is negative, it refers to the dense layers:
            -2 is the hidden layer
            -1 is the decision layer (we probably wont add noise here)
        
        noise_level must be between 0 and 1, where 0 means no noise, and
        1 means 100% noise
        
        Parameters
        ----------
        x : TYPE
            DESCRIPTION.
        idx_layer : int, optional
            DESCRIPTION. The default is 0.
        noise_level : float, optional
            DESCRIPTION. The default is 0.
        noise_type : str, optional
            DESCRIPTION. The default is 'gaussian'. Another option is 'reduction'.
            'gaussion' means add Gaussian noise to the layer outputs
            'reduction' means multiplying a factor to reduce the layer outputs

        Returns
        -------
        features : TYPE
            DESCRIPTION.
        hidden_representation : TYPE
            DESCRIPTION.
        prediction : TYPE
            DESCRIPTION.

        """
        #  通过cnn层处理图片
        for ii,layer in enumerate(self.layers_of_feature_extractor):
            x = layer(x)
        # it is very difficult to deal with the dimension
        features = self.avgpool(x)
        features = torch.squeeze(torch.squeeze(features,3),2)
        if self.batch_size == 1:
            features = features[None,:]
        # hidden layer
        hidden_representation = self.hidden_layer(features)
        # decision layer
        prediction = self.decision_layer(hidden_representation)
        # confidence layer
        confidence = self.confidence_layer(hidden_representation)

        return features,hidden_representation,prediction, confidence

class betting_network(nn.Module):
    def __init__(self,
                 # for the CNN network
                 pretrained_model_name:str      = 'mobilenet',
                 hidden_layer_size:int          = 300,
                 hidden_activation:nn.Module    = nn.ReLU(),
                 hidden_dropout:float           = 0.,
                 hidden_layer_type:str          = 'linear',
                 output_layer_size:int          = 2,
                 in_shape:Tuple                 = (1,3,128,128),
                 retrain_encoder:bool           = False,
                 cnn_weight_path:str            = 'temp.h5',
                 # for the betting network
                 betting_layers:List            = [256,],
                 signal_source:str              = 'hidden',
                 device                         = 'cpu',
                 batch_size:int                 = 8,
                 ):
        super(betting_network,self,).__init__()
        torch.manual_seed(12345)
        # make a CNN network and load the trained weights
        cnn_classifier_args = dict(
                pretrained_model_name   = pretrained_model_name,
                hidden_layer_size       = hidden_layer_size,
                hidden_activation       = hidden_activation,
                hidden_dropout          = hidden_dropout,
                hidden_layer_type       = hidden_layer_type,
                output_layer_size       = output_layer_size,
                in_shape                = in_shape,
                retrain_encoder         = retrain_encoder,
                device                  = device,
                )
        self.device = device
        self.batch_size = batch_size
        ## load trained weights
        self.simple_classifier = perceptual_network(**cnn_classifier_args).to(device)
        if os.path.exists(cnn_weight_path):
            self.simple_classifier.load_state_dict(torch.load(cnn_weight_path))
            self.simple_classifier.eval()
            ## freeze the CNN model
            for p in self.simple_classifier.parameters():p.requires_grad = False
        else:
            print('pretrained perceptual network not existed, training from scratch')
            raise NotImplementedError
        # make a betting network
        self.signal_source = signal_source
        if signal_source == 'hidden':
            input_units = hidden_layer_size * 2
        elif signal_source == 'prediction':
            input_units = output_layer_size * 2
        elif signal_source == 'both':
            input_units = (hidden_layer_size + output_layer_size) * 2
        else:
            raise NotImplementedError
        layers = [create_hidden_layer(
                    layer_type          = hidden_layer_type,
                    input_units         = input_units,
                    output_units        = betting_layers[0],
                    output_activation   = hidden_activation,
                    output_dropout      = hidden_dropout,
                    device              = device,
                    ).to(device)]
        if len(betting_layers) > 1:
            for in_features,out_features in zip(betting_layers[:-1],
                                                betting_layers[1:],):
                layers.append(create_hidden_layer(
                    layer_type          = hidden_layer_type,
                    input_units         = in_features,
                    output_units        = out_features,
                    output_activation   = hidden_activation,
                    output_dropout      = hidden_dropout,
                    device              = device,
                    ).to(device))
        self.betting_hidden = nn.Sequential(*layers).to(device)
        self.betting_output = nn.Sequential(nn.Linear(betting_layers[-1],
                                                      2,),
                                            nn.Softmax(dim = -1)).to(device)
        
    def forward(self,image1:Tensor,image2:Tensor,
                idx_layer_perception:int = 0,
                idx_layer_betting:int = 0,
                noise_level:float = 0,
                add_noise_to:str = 'perception',
                noise_type:str = 'gaussian',
                ):
        """
        

        Parameters
        ----------
        image1 : Tensor
            DESCRIPTION.
        image2 : Tensor
            DESCRIPTION.
        idx_layer_perception : int, optional
            DESCRIPTION. The default is 0.
        idx_layer_betting : int, optional
            DESCRIPTION. The default is 0.
        noise_level : float, optional
            DESCRIPTION. The default is 0. Must be between 0 and 1
        add_noise_to : str, optional
            DESCRIPTION. The default is 'perception'. Or 'betting'
        noise_type : str, optional
            DESCRIPTION. The default is 'gaussian'. Another option is 'reduction'.
            'gaussion' means add Gaussian noise to the layer outputs
            'reduction' means multiplying a factor to reduce the layer outputs

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        if add_noise_to == 'perception' and noise_level > 0: #add noise
            (features1,
             hidden_representation1,
             prediction1) = self.simple_classifier(image1,
                                                   idx_layer    = idx_layer_perception,
                                                   noise_level  = noise_level,
                                                   )
            (features2,
             hidden_representation2,
             prediction2) = self.simple_classifier(image2,
                                                   idx_layer    = idx_layer_perception,
                                                   noise_level  = noise_level,
                                                   )
        else: # don't add noise
            (features1,
             hidden_representation1,
             prediction1) = self.simple_classifier(image1)
            (features2,
             hidden_representation2,
             prediction2) = self.simple_classifier(image2)
        
        # concatenate the hidden representations of the images
        if self.signal_source == 'hidden':
            betting_input = torch.cat((hidden_representation1,
                                       hidden_representation2,
                                       ), 1)
        elif self.signal_source == 'prediction':
            betting_input = torch.cat((prediction1,
                                       prediction2,
                                       ), 1)
        elif self.signal_source == 'both':
            betting_input = torch.cat((hidden_representation1,
                                       hidden_representation2,
                                       prediction1,
                                       prediction2,
                                       ), 1)
        else:
            raise NotImplementedError
        
        # add noise to the image representations if needed
        if (add_noise_to == 'betting_input') and (noise_level > 0):
            if noise_type == 'gaussian':
                x_min,x_max = betting_input.min(),betting_input.max()
                # standard nornal noise
                noise_generator = torch.distributions.Normal(0,1)
                noise = noise_generator.sample(betting_input.shape).to(self.device)
                betting_input = betting_input * (1 - noise_level) + noise * noise_level
                betting_input = torch.clamp(betting_input,x_min,x_max,)
            elif noise_type == 'reduction':
                betting_input = betting_input * noise_level
            else:
                raise NotImplementedError
        
        # feed to the betting network
        betting_hidden = self.betting_hidden(betting_input,)
        if (add_noise_to == 'betting') and (noise_level > 0):
            if noise_type == 'gaussian':
                x_min,x_max = betting_hidden.min(),betting_hidden.max()
                # standard nornal noise
                noise_generator = torch.distributions.Normal(0,1)
                noise = noise_generator.sample(betting_hidden.shape).to(self.device)
                betting_hidden = betting_hidden * (1 - noise_level) + noise * noise_level
                betting_hidden = torch.clamp(betting_hidden,x_min,x_max,)
            elif noise_type == 'reduction':
                betting_hidden = betting_hidden * noise_level
            else:
                raise NotImplementedError
        betting_output = self.betting_output(betting_hidden,)
        return betting_output,(features1,hidden_representation1,prediction1),(features2,hidden_representation2,prediction2)
"""
class simple_cnn_classifier(nn.Module):
    def __init__(self,
                 pretrained_model_name:str = 'mobilenet',
                 hidden_layer_size:int = 300,
                 hidden_activation:nn.Module = nn.ReLU(),
                 hidden_dropout:float = 0.,
                 hidden_layer_type:str = 'linear',
                 output_layer_size:int = 2,
                 in_shape:Tuple = (1,3,128,128),
                 retrain_encoder:bool = False,
                 device = 'cpu',
                 ):
        super(simple_cnn_classifier,self).__init__()
        torch.manual_seed(12345)
        
        in_features,feature_extractor  = CNN_feature_extractor(
                                            pretrained_model_name   = pretrained_model_name,
                                            retrain_encoder         = retrain_encoder,
                                            in_shape                = in_shape,
                                            device                  = device,
                                                               )
        self.feature_extractor = feature_extractor.to(device)
        self.avgpool           = nn.AdaptiveAvgPool2d((1,1))
        self.hidden_layer      = create_hidden_layer(
                                            layer_type          = hidden_layer_type,
                                            input_units         = in_features,
                                            output_units        = hidden_layer_size,
                                            output_activation   = hidden_activation,
                                            output_dropout      = hidden_dropout,
                                            device              = device,
                                            ).to(device)
        self.output_layer = nn.Sequential(nn.Linear(hidden_layer_size,output_layer_size),
                                          nn.Softmax(dim = -1)
                                          ).to(device)
    
    def forward(self,x):
        features = self.feature_extractor(x)
        features = self.avgpool(features)
        features = torch.squeeze(torch.squeeze(features,3),2)
        hidden_representation = self.hidden_layer(features)
        prediction = self.output_layer(hidden_representation)
        return features,hidden_representation,prediction

class deeper_cnn_classifier(nn.Module):
    def __init__(self,
                 pretrained_model_name:str = 'mobilenet',
                 additional_cnn:int = 3,
                 hidden_layer_size:int = 300,
                 hidden_activation:nn.Module = nn.ReLU(),
                 hidden_dropout:float = 0.,
                 hidden_layer_type:str = 'linear',
                 output_layer_size:int = 2,
                 in_shape:Tuple = (1,3,128,128),
                 retrain_encoder:bool = False,
                 device = 'cpu',
                 ):
        super(deeper_cnn_classifier,self).__init__()
        torch.manual_seed(12345)
        
        in_features,feature_extractor  = CNN_feature_extractor(
                                            pretrained_model_name   = pretrained_model_name,
                                            retrain_encoder         = retrain_encoder,
                                            in_shape                = in_shape,
                                            device                  = device,
                                                               )
        self.feature_extractor = feature_extractor.to(device)
        cnn_list = []
        for _ in range(additional_cnn):
            cnn_list.append(create_hidden_layer(layer_type = 'convolutional',
                                                input_units = in_features,
                                                out_channels = in_features,
                                                kernel_size = 3,
                                                output_dropout = hidden_dropout,))
        self.cnn_layers = nn.Sequential(*cnn_list)
        
        self.avgpool           = nn.AdaptiveAvgPool2d((1,1))
        self.hidden_layer = create_hidden_layer(
                                            layer_type          = hidden_layer_type,
                                            input_units         = in_features,
                                            output_units        = hidden_layer_size,
                                            output_activation   = hidden_activation,
                                            output_dropout      = hidden_dropout,
                                            device              = device,
                                            ).to(device)
        self.output_layer = nn.Sequential(nn.Linear(hidden_layer_size,
                                                    output_layer_size),
                                          nn.Softmax(dim = -1)
                                          ).to(device)
    
    def forward(self,x):
        features = self.feature_extractor(x)
        features = self.cnn_layers(features)
        features = self.avgpool(features)
        features = torch.squeeze(torch.squeeze(features))
        hidden_representation = self.hidden_layer(features)
        prediction = self.output_layer(hidden_representation)
        return features,hidden_representation,prediction


class betting_network(nn.Module):
    def __init__(self,
                 # for the CNN network
                 pretrained_model_name:str = 'mobilenet',
                 hidden_layer_size:int = 300,
                 hidden_activation:nn.Module = nn.ReLU(),
                 hidden_dropout:float = 0.,
                 hidden_layer_type:str = 'linear',
                 output_layer_size:int = 2,
                 in_shape:Tuple = (1,3,128,128),
                 retrain_encoder:bool = False,
                 cnn_weight_path:str = 'temp.h5',
                 # for the betting network
                 betting_layers:List = [256,],
                 signal_source:str = 'hidden',
                 device = 'cpu',):
        super(betting_network,self,).__init__()
        torch.manual_seed(12345)
        # make a CNN network and load the trained weights
        cnn_classifier_args = dict(
                pretrained_model_name   = pretrained_model_name,
                hidden_layer_size       = hidden_layer_size,
                hidden_activation       = hidden_activation,
                hidden_dropout          = hidden_dropout,
                hidden_layer_type       = hidden_layer_type,
                output_layer_size       = output_layer_size,
                in_shape                = in_shape,
                retrain_encoder         = retrain_encoder,
                device                  = device,
                )
        ## load trained weights
        self.simple_classifier = simple_cnn_classifier(**cnn_classifier_args).to(device)
        if os.path.exists(cnn_weight_path):
            self.simple_classifier.load_state_dict(torch.load(cnn_weight_path))
            self.simple_classifier.eval()
            ## freeze the CNN model
            for p in self.simple_classifier.parameters():p.requires_grad = False
        else:
            print('pretrained perceptual network not existed, training from scratch')
            raise NotImplementedError
        # make a betting network
        self.signal_source = signal_source
        if signal_source == 'hidden':
            input_units = hidden_layer_size * 2
        elif signal_source == 'prediction':
            input_units = output_layer_size * 2
        elif signal_source == 'both':
            input_units = (hidden_layer_size + output_layer_size) * 2
        else:
            raise NotImplementedError
        layers = [create_hidden_layer(
                    layer_type          = hidden_layer_type,
                    input_units         = input_units,
                    output_units        = betting_layers[0],
                    output_activation   = hidden_activation,
                    output_dropout      = hidden_dropout,
                    device              = device,
                    ).to(device)]
        if len(betting_layers) > 1:
            for in_features,out_features in zip(betting_layers[:-1],
                                                betting_layers[1:],):
                layers.append(create_hidden_layer(
                    layer_type          = hidden_layer_type,
                    input_units         = in_features,
                    output_units        = out_features,
                    output_activation   = hidden_activation,
                    output_dropout      = hidden_dropout,
                    device              = device,
                    ).to(device))
        self.betting_hidden = nn.Sequential(*layers).to(device)
        self.betting_output = nn.Sequential(nn.Linear(betting_layers[-1],
                                                      2,),
                                            nn.Softmax(dim = -1)).to(device)
        
    def forward(self,image1,image2):
        features1,hidden_representation1,prediction1 = self.simple_classifier(image1)
        features2,hidden_representation2,prediction2 = self.simple_classifier(image2)
        # concatenate the hidden representations of the images
        if self.signal_source == 'hidden':
            betting_input = torch.cat((hidden_representation1,
                                       hidden_representation2,
                                       ), 1)
        elif self.signal_source == 'prediction':
            betting_input = torch.cat((prediction1,
                                       prediction2,
                                       ), 1)
        elif self.signal_source == 'both':
            betting_input = torch.cat((hidden_representation1,
                                       hidden_representation2,
                                       prediction1,
                                       prediction2,
                                       ), 1)
        else:
            raise NotImplementedError
        # print(betting_input.shape)
        # feed to the betting network
        betting_hidden = self.betting_hidden(betting_input,)
        betting_output = self.betting_output(betting_hidden)
        return betting_output,(features1,hidden_representation1,prediction1),(features2,hidden_representation2,prediction2)

"""
import torch
import torch.nn as nn

class Encoder(nn.Module):
	def __init__(self):
		super(Encoder, self).__init__()
		# Convolutional encoder
		self.conv1 = nn.Conv2d(1, 32, 3, stride=2, padding=1)
		self.conv1_BN = nn.BatchNorm2d(32)
		self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
		self.conv2_BN = nn.BatchNorm2d(32)
		self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
		self.conv3_BN = nn.BatchNorm2d(32)
		# MLP encoder
		self.fc1 = nn.Linear(512, 256)
		self.fc1_BN = nn.BatchNorm1d(256)
		self.fc2 = nn.Linear(256, 128)
		self.fc2_BN = nn.BatchNorm1d(128)
		self.latent_dim = 64
		self.z_out = nn.Linear(128, self.latent_dim)
		# Nonlinearities
		self.leaky_relu = nn.LeakyReLU()
	def forward(self, x):
		# Convolutional encoder
		conv1_out = self.leaky_relu(self.conv1_BN(self.conv1(x)))
		conv2_out = self.leaky_relu(self.conv2_BN(self.conv2(conv1_out)))
		conv3_out = self.leaky_relu(self.conv3_BN(self.conv3(conv2_out)))
		# Flatten output of conv. net
		conv3_out_flat = torch.flatten(conv3_out, 1)
		# MLP encoder
		fc1_out = self.leaky_relu(self.fc1_BN(self.fc1(conv3_out_flat)))
		fc2_out = self.leaky_relu(self.fc2_BN(self.fc2(fc1_out)))
		z = self.z_out(fc2_out)
		# Create dict with flattened states (for decoding analyses)
		all_layers = {'conv1': torch.flatten(conv1_out, 1),
					  'conv2': torch.flatten(conv2_out, 1),
					  'conv3': conv3_out_flat,
					  'fc1': fc1_out,
					  'fc2': fc2_out,
					  'z': z}
		return z, all_layers

class Class_out(nn.Module):
	def __init__(self):
		super(Class_out, self).__init__()
		# Feedforward layer
		self.fc = nn.Linear(64, 1)
		# Nonlinearities
		self.softmax = nn.Softmax(dim=-1)
	def forward(self, z):
		y = self.softmax(self.fc(z))
		return y

class Conf_out(nn.Module):
	def __init__(self):
		super(Conf_out, self).__init__()
		# Feedforward layer
		self.fc = nn.Linear(64, 1)
		# Nonlinearities
		self.softmax = nn.Softmax(dim=-1)
	def forward(self, z):
		conf = self.softmax(self.fc(z))
		return conf

class Decoder(nn.Module):
	def __init__(self, input_size):
		super(Decoder, self).__init__()
		# Feedforward layer
		self.fc = nn.Linear(input_size, 1, bias=False)
		# Nonlinearities
		self.softmax = nn.Softmax(dim=-1)
	def forward(self, x):
		y_linear = self.fc(x)
		y = self.softmax(y_linear)
		return y, y_linear

if __name__ == "__main__":
    pass