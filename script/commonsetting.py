from torch import nn
import torch
import os
path = os.getcwd()
#training_dir = os.path.join(path, "data", "img", "train")
training_dir = "../data/img/validation"
#val_dir = os.path.join(path, "data", "img", "validation")
val_dir = "../data/img/sub_train"
#test_dir = os.path.join(path, "data", "img", "test")
test_dir = "../data/img/sub_test"

noise_level_train = 3
noise_level_val = 1
noise_level_test = 0.5

batch_size = 32#改
image_resize = 128#改
num_workers = 4
learning_rate = 1e-4#改

label_map = dict(circle=[1, 0, 0], triangle=[0, 1, 0], square=[0, 0, 1])

tol = 1e-4

patience = 20#改
warmup_epochs = 3

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

# model configurations
pretrained_model_name   = 'resnet50'
hidden_layer_size       = 300
hidden_activation_name  = 'selu'
hidden_activation       = hidden_activation_functions(hidden_activation_name)
hidden_dropout          = 0.
use_object_cnn          = False # single trained
hidden_layer_type       = 'linear'
output_layer_size       = 3
confidence_layer_size   = 2
in_shape                = (1, 3, image_resize, image_resize)
retrain_encoder         = False
device                  = torch.device('cuda' if torch.cuda.is_available() else 'cpu');#print(f'working on {device}')
if __name__ == "__main__":
    pass