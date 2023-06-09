import os
import numpy as np
from PIL import Image
from typing import List, Union, Tuple, Any

import time

import torch
import torch.nn as nn
from torch import optim
from einops import rearrange
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

import torchmetrics

# To use config
from config import LSTMConfig

# To use Paths class
from utils.file_utils import create_output_dirs, PathsContainer
from argparse import ArgumentParser, Namespace

from dataset.video_dataset import VideoFrameDataset, ImglistToTensor, MaskDataset
from dataset.dataset_utils import load_data

from models.Seq2Seq import Seq2Seq
from utils.early_stop import EarlyStop
import losses





def parse_args() -> Namespace:
    parser = ArgumentParser("ConvLSTM")
    parser.add_argument("--config-file-name", required=True, type=str, help="Name of json file with config")
    parser.add_argument("--output-dir", help="Name of dir to save the checkpoints to", required=True, type=str)
    parser.add_argument("--run-id", help="Name of the run", required=True, type=str)
    parser.add_argument("--resume", help="In case training was not completed resume from last epoch", default=False, type=bool)

    return parser.parse_args()

def collate(batch):

    batch = torch.cat(batch)

    # Split input and target
    # Randomly pick 10 frames as input, 11th frame is target
    rand = np.random.randint(11,21)
    input = batch[:,rand-11:rand, :, :]
    target = batch[:,rand,:,:]
    input = one_hot_encoding(input, 49)                             
                   
    return input, target 


def one_hot_encoding(input_tensor, num_classes=49):
    one_hot = F.one_hot(input_tensor.long(), num_classes)
    one_hot = rearrange(one_hot, 'b f h w c -> b c f h w')  # Change the dimensions to (batch_size, num_classes, height, width)
    return one_hot


def train(epoch, model, train_loader, val_loader, criterion, optimizer, scheduler, early_stop, num_epochs, device, output_dir):

    while epoch < num_epochs:
        print(f'Starting epoch {epoch}')
        start_time = time.time()
        model.train()
        train_loss = 0                                            
        for idx, data in enumerate(train_loader):
            input, target = data
            input, target = input.to(device), target.to(device)

            optimizer.zero_grad()
            
            output = model(input)
                                        
            loss = criterion(output, target)       
            loss.backward()                                            
            optimizer.step()      

            train_loss += loss.item()
            if scheduler is not None:
                scheduler.step()

            if idx % 50 == 0 and epoch < 5:
              print(f"Loss on current batch: {loss.item()}") # we can just take a sample, don't need to average it
            
        avg_epoch_loss = train_loss / len(train_loader)
        end_time = time.time()

        model.eval()
        predicted_masks_22 = []
        actual_masks_22 = []
        val_loss = 0
        with torch.no_grad():
            for idx, data in enumerate(val_loader):
                input, label, masks = data
                input, label, masks = input.to(device), label.to(device), masks.to(device)

                context = one_hot_encoding(masks[:,:11], 49)
                for frame in range(11):
                    pred = model(context)

                    if frame == 0: #criterion on first frame
                        loss = criterion(pred, masks[:,11].long())
                        val_loss += loss.item()

                    pred = torch.argmax(pred, dim=1)
                    
                    if frame == 10:
                        predicted_masks_22.append(pred.cpu())
                        actual_masks_22.append(masks[:,21].cpu())
                    
                    pred = one_hot_encoding(pred.unsqueeze(1))
                    context = context[:,:,-10:]
                    context = torch.cat([context, pred], dim=2)

            avg_epoch_val_loss = val_loss / len(val_loader)
            
            predicted_masks_22 = torch.cat(predicted_masks_22, dim=0)
            actual_masks_22 = torch.cat(actual_masks_22, dim=0)
            jaccard = compute_jaccard(actual_masks_22, predicted_masks_22)

        print(f"Epoch: {epoch}, Time for training epoch {end_time - start_time}, Average epoch loss: {avg_epoch_loss:.4f}, Average epoch val loss: {avg_epoch_val_loss:.4f}, Jaccard Score: {jaccard}")

        plot_dir = os.path.join(output_dir, 'training_plots')
        plot_model_example(output[0], target[0], plot_dir, f'example_epoch_{epoch}')

        # Early Stopping
        if jaccard > early_stop.best_value:
            torch.save(model.module.state_dict() if torch.cuda.device_count() > 1 else model.state_dict(), os.path.join(output_dir, 'models/best',"best_model.pkl"))

        early_stop.step(jaccard, epoch)
        if early_stop.stop_training(epoch):
            print(
                "early stopping at epoch {} since valdiation loss didn't improve from epoch no {}. Best value {}, current value {}".format(
                    epoch, early_stop.best_epoch, early_stop.best_value, jaccard
                ))
            break

        # Used this approach (while and epoch increase) so that we can get back to training the loaded model from checkpoint
        epoch += 1

        # Checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.module.state_dict() if torch.cuda.device_count() > 1 else model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
            'early_stop': early_stop,
            }, os.path.join(output_dir, 'models/partial', "checkpoint.pkl"))

    return {
        "epochs": epoch,
        "train_loss": train_loss,
        "model": model
            }


def compute_jaccard(ground_truth_mask, predicted_mask):
    jaccard = torchmetrics.JaccardIndex(task="multiclass", num_classes=49) #.to(device)
    return jaccard(torch.Tensor(ground_truth_mask), torch.Tensor(predicted_mask))


def validate_model(model, valloader, index_to_show):
    model.eval()
    predicted_masks_22 = []
    actual_masks_22 = []
    predicted_masks_show = []
    with torch.no_grad():
        for batch_num, data in enumerate(valloader):
            input, label, masks = data
            input, label, masks = input.to(device), label.to(device), masks.to(device)

            if batch_num == index_to_show:
                actual_masks_show = masks[0,11:].cpu()

            context = one_hot_encoding(masks[:,:11], 49)
            for frame in range(11):
                pred = model(context)
                pred = torch.argmax(pred, dim=1)
                if frame == 10:
                    predicted_masks_22.append(pred[:].cpu())
                    actual_masks_22.append(masks[:,21].cpu())
                if batch_num == index_to_show:
                    predicted_masks_show.append(pred[0].cpu())
                
                pred = one_hot_encoding(pred.unsqueeze(1))
                context = context[:,:,-10:]
                context = torch.cat([context, pred], dim=2)
        
        predicted_masks_22 = torch.cat(predicted_masks_22, dim=0)
        actual_masks_22 = torch.cat(actual_masks_22, dim=0)
        jaccard = compute_jaccard(actual_masks_22, predicted_masks_22)
        print(f"Jaccard Score: {jaccard}")
    return predicted_masks_show, actual_masks_show


def plot_model_example(prediction, target, output_dir, fig_name='example'):
    prediction = torch.argmax(prediction, dim=0)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    ax1.imshow(prediction.cpu().numpy())
    ax1.set_title("Predicted Next Frame Mask")
    ax2.imshow(target.cpu().numpy())
    ax2.set_title("Actual Next Frame Mask")
    plt.savefig(f'{output_dir}/{fig_name}.pdf')


def plot_model_result(pred, actual, output_dir, fig_name='example'):
    """
    Plot and save figure
    """
    num_frames = len(pred)
    fig, ax = plt.subplots(2, num_frames, figsize = (num_frames*2, 4))
    fig.subplots_adjust(wspace=0.01, hspace = 0.1)

    for j in range(num_frames):
      ax[0,j].set_axis_off()
      ax[1,j].set_axis_off()
      
      img_pred = pred[j]
      img_actual = actual[j]

      ax[0,j].imshow(img_pred)
      ax[1,j].imshow(img_actual)
    fig.savefig(f'{output_dir}/{fig_name}.pdf', bbox_inches = 'tight')


if __name__ == "__main__":

    args = parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    paths = PathsContainer.from_args(args.output_dir, args.run_id, args.config_file_name)
    create_output_dirs(paths.save_dir)

    if args.resume:
        print("Resuming training...")
        config = LSTMConfig.from_json(os.path.join(paths.save_dir, 'used_config.json'))
        config_input = LSTMConfig.from_json(paths.config_path)
        if config_input != config:
            print("Input config differs from used config: loading used config")
    else:
        config = LSTMConfig.from_json(paths.config_path)
        output_config_path = os.path.join(paths.save_dir, "used_config.json")
        os.system("cp {} {}".format(paths.config_path, output_config_path))

    num_epochs = config.training.epochs

    print('Loading train data...')
    dataset_unlabeled = MaskDataset(save_dir=config.data.train.path)
    dataloader_unlabeled = DataLoader(dataset_unlabeled, batch_size=config.data.train.batch_size, shuffle=True, collate_fn=collate)

    print('Loading val data...')
    val_data_dir  = os.path.join(config.data.val.path, 'data')
    val_annotation_dir = os.path.join(config.data.val.path, 'annotations.txt')
    dataloader_val = load_data(val_data_dir, val_annotation_dir, config.data.val.batch_size, mask=True)

    epoch = 0

    model = Seq2Seq(**config.lstm_model.args, device=device)
    model.to(device)
    print('LSTM Loaded.')

    optimizer = getattr(optim, config.optimizer.name)(params=model.parameters(), **config.optimizer.args)

    total_steps = num_epochs * len(dataloader_unlabeled)

    scheduler = getattr(optim.lr_scheduler, config.lr_scheduler.name)(optimizer, T_max=total_steps, **config.lr_scheduler.args)

    class_weights = torch.ones(49)
    class_weights[0] = config.criterion.background_weight
    class_weights = class_weights.to(device)

    if config.criterion.name in ['CEandDiceLoss', 'MaskLoss']:
        criterion = getattr(losses, config.criterion.name)(weight=class_weights, **config.criterion.args)
    else:
        criterion = getattr(nn, config.criterion.name)(weight=class_weights, **config.criterion.args)

    criterion.to(device)

    early_stop = EarlyStop(config.training.early_stopping_patience)


    if args.resume:
        print("Attempting to find existing checkpoint")
        path_partials = os.path.join(paths.save_dir, "models/partial")
        try:
            checkpoint = torch.load(os.path.join(path_partials, "checkpoint.pkl"), map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if scheduler is not None:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            epoch = checkpoint['epoch']
            early_stop = checkpoint['early_stop']
            print(f'Resuming training from epoch {epoch}')
        except :
            print("Couldn't load model from checkpoint, starting again training from epoch 0")
    
    # Pass model to DataParallel (Could improve by using DistributedDataParallel)
    if torch.cuda.device_count() > 1:
        print("Model training will be distributed to {} GPUs.".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)
    model.to(device)

    print('Start training model...')
    results = train(epoch, model, dataloader_unlabeled, dataloader_val, criterion, optimizer, scheduler, early_stop, num_epochs, device, paths.save_dir)

    print(f'Model training finshed at epoch {results["epochs"]}, trainig loss: {results["train_loss"]}')


