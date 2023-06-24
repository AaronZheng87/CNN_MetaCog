import torch
from torchvision.datasets import ImageFolder, DatasetFolder
import commonsetting
import os
from glob import glob
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image



#建立自己的dataset
def noise_func(x, noise_level:float = 0.):
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
                                center_crop_size:tuple = (1024,1024),
                                ):
    """
    https://github.com/nmningmei/ensemble_perception_simulation/blob/main/scripts/utils_deep.py

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

class CustomImageDataset(Dataset):
    """
    
    """
    def __init__(self,
                 img_dir:str,
                 label_map,
                 transform = None):
        self.img_dir            = img_dir
        self.label_map         = label_map
        self.transform          = transform

        
        self.images = glob(os.path.join(img_dir,'*','*.png'))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path   = self.images[idx]
        image = Image.open(img_path)
        image = self.transform(image)
        label = img_path.split("/")[-2]
        label = self.label_map[label]
        return image, label

if __name__ == "__main__":
    tranformer_steps = concatenate_transform_steps(image_resize=commonsetting.image_resize, rotate=45)
    dataset_train = CustomImageDataset(commonsetting.training_dir,label_map=commonsetting.label_map , transform=tranformer_steps)
    dataloader_train = DataLoader(dataset_train, batch_size=commonsetting.batch_size, shuffle=True, num_workers=commonsetting.num_workers)
    

    a,d = next(iter(dataloader_train))
