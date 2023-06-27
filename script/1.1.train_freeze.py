import commonsetting
from models import perceptual_network, Encoder, Class_out, Conf_out
from dataloader import CustomImageDataset, concatenate_transform_steps
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch import nn
import torch
from tqdm import tqdm
import numpy as np



def determine_training_stops(net,
                             idx_epoch:int,
                             warmup_epochs:int,
                             valid_loss,
                             counts: int        = 0,
                             device             = commonsetting.device,
                             best_valid_loss    = np.inf,
                             tol:float          = 1e-4,
                             f_name:str         = 'temp.h5',
                             ):
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

def training_loop(dataloader_train, device, model, loss_function, optimizer):
    model.train(True)
    dataloader_train = tqdm(dataloader_train)
    train_loss = 0.

    for idx_batch, (batch_image, batch_label) in enumerate(dataloader_train):

        batch_label = torch.vstack(batch_label).T.float()
        #记得每一次处理数据之前要做这一步
        optmizer.zero_grad()

        features,hidden_representation,prediction, confidence = model(batch_image.to(device))

        class_loss = loss_function(prediction.float(), batch_label.float())
        
        correct_preds = batch_label.clone().detach().argmax(1)==prediction.clone().detach().argmax(1)
        correct_preds = correct_preds.float()

        correct_preds = torch.vstack([1-correct_preds, correct_preds]).T.float()
        


        conf_loss = loss_function(confidence.float(), correct_preds.float())

        combined_loss = class_loss + conf_loss
        train_loss = train_loss + combined_loss.item()
        combined_loss.backward()
        optmizer.step()
        dataloader_train.set_description(f"train loss = {train_loss/(idx_batch + 1):2.6f}")
    
    return model, train_loss


def validation_loop(dataloader_val, device, model, loss_function, optimizer):

    model.eval()
    dataloader_val = tqdm(dataloader_val)
    val_loss = 0.

    with torch.no_grad():
        for idx_batch, (batch_image, batch_label) in enumerate(dataloader_val):
            batch_label = torch.vstack(batch_label).T.float()
            #记得每一次处理数据之前要做这一步

            features,hidden_representation,prediction, confidence = model(batch_image.to(device))

            class_loss = loss_function(prediction.float(), batch_label.float())

        
            correct_preds = batch_label.clone().detach().argmax(1)==prediction.clone().detach().argmax(1)
            correct_preds = correct_preds.float()

            correct_preds = torch.vstack([1-correct_preds, correct_preds]).T.float()
            

            conf_loss = loss_function(confidence.float(), correct_preds.float())

            combined_loss = class_loss + conf_loss
            val_loss = val_loss + combined_loss.item()
            dataloader_val.set_description(f"validation loss = {val_loss/(idx_batch + 1):2.6f}")
    return model, val_loss


if __name__ == "__main__":
    tranformer_steps = concatenate_transform_steps(image_resize=commonsetting.image_resize, rotate=45)
    dataset_train = CustomImageDataset(commonsetting.training_dir,label_map=commonsetting.label_map , transform=tranformer_steps)
    dataloader_train = DataLoader(dataset_train, batch_size=commonsetting.batch_size, shuffle=True, num_workers=commonsetting.num_workers)
    dataset_val = CustomImageDataset(commonsetting.val_dir,label_map=commonsetting.label_map , transform=tranformer_steps)
    dataloader_val = DataLoader(dataset_val, batch_size=commonsetting.batch_size, shuffle=True, num_workers=commonsetting.num_workers)
    SimpleCNN = perceptual_network(pretrained_model_name=commonsetting.pretrained_model_name, 
                                   hidden_layer_size=commonsetting.hidden_layer_size, hidden_activation=commonsetting.hidden_activation,
                                   hidden_dropout=commonsetting.hidden_dropout, hidden_layer_type=commonsetting.hidden_layer_type, output_layer_size=commonsetting.output_layer_size, 
                                   confidence_layer_size = commonsetting.confidence_layer_size, in_shape=commonsetting.in_shape, retrain_encoder=commonsetting.retrain_encoder, 
                                   )


    SimpleCNN = SimpleCNN.to(commonsetting.device)
    for p in SimpleCNN.parameters():
        p.requires_grad = False

    for p in SimpleCNN.hidden_layer.parameters():
        p.requires_grad = True

    for p in SimpleCNN.decision_layer.parameters():
        p.requires_grad = True

    for p in SimpleCNN.confidence_layer.parameters():
        p.requires_grad = False# if want to train the confidence layer, use True

    params = [{"params": SimpleCNN.hidden_layer.parameters(),
               "lr": commonsetting.learning_rate,
               },
               {
                "params": SimpleCNN.decision_layer.parameters(),
               "lr": commonsetting.learning_rate,
               }''', 
               {
                "params": SimpleCNN.confidence_layer.parameters(),
               "lr": commonsetting.learning_rate,
               }'''
             ]

    optmizer = Adam(params, lr=commonsetting.learning_rate)
    loss_fun = nn.BCELoss()
    counts = 0
    best_valid_loss = np.inf
    for epoch in range(1000):
        SimpleCNN, train_loss = training_loop(dataloader_train, commonsetting.device, SimpleCNN, loss_fun, optmizer)
        SimpleCNN, val_loss = validation_loop(dataloader_val, commonsetting.device, SimpleCNN, loss_fun, optmizer)
        best_valid_loss, counts = determine_training_stops(SimpleCNN, epoch, warmup_epochs=commonsetting.warmup_epochs, valid_loss=val_loss, counts=counts, 
                                 device=commonsetting.device, best_valid_loss=best_valid_loss, tol=commonsetting.tol, 
                                 f_name="../models/simplecnn_bs32e4i224h200.h5")
        if counts >= commonsetting.patience:#(len(losses) > patience) and (len(set(losses[-patience:])) == 1):
            break
        else:
            print(f'\nepoch {epoch + 1}, best valid loss = {best_valid_loss:.8f},count = {counts}')





