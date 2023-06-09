import os
import numpy as np
from PIL import Image
from typing import List, Union, Tuple, Any

import time
import random

import torch
import torch.nn as nn
from torch import optim
from einops import rearrange
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision import transforms

import torchmetrics

# To use config
from config import FineTuneConfig
from functools import partial

# To use Paths class
from utils.file_utils import create_output_dirs, PathsContainer
from argparse import ArgumentParser, Namespace

from dataset.video_dataset import VideoFrameDataset, ImglistToTensor, MaskDataset
from dataset.dataset_utils import load_data

from models.Seq2Seq import Seq2Seq
from models.UNet import UNet
from utils.early_stop import EarlyStop
import losses


def parse_args() -> Namespace:
    parser = ArgumentParser("ConvLSTM")
    parser.add_argument("--config-file-name", required=True, type=str, help="Name of json file with config")
    parser.add_argument("--output-dir", help="Name of dir to save the checkpoints to", required=True, type=str)
    parser.add_argument("--run-id", help="Name of the run", required=True, type=str)
    parser.add_argument("--resume", help="In case training was not completed resume from last epoch", default=False, type=bool)

    return parser.parse_args()


def one_hot_encoding(input_tensor, num_classes=49):
    one_hot = F.one_hot(input_tensor.long(), num_classes)
    one_hot = rearrange(one_hot, 'b f h w c -> b c f h w')  # Change the dimensions to (batch_size, num_classes, height, width)
    return one_hot


def finetune(epoch, model, masker, train_loader, val_loader, criterion, model_optimizer, model_scheduler, early_stop, num_epochs, device, output_dir):

    while epoch < num_epochs:
        print(f'Starting epoch {epoch}')
        start_time = time.time()
        model.train()
        masker.eval()
        train_loss = 0                                       
        for idx, data in enumerate(train_loader):
            frames, labels, masks = data
            frames, labels, masks = frames.to(device), labels.to(device), masks.to(device)

            B = frames.shape[0]

            model_optimizer.zero_grad()

            frames = rearrange(frames, 'b t c h w -> (b t) c h w')

            gen_masks = masker(frames)
            gen_masks = torch.argmax(gen_masks, dim=1)
            gen_masks = rearrange(gen_masks, '(b t) h w -> b t h w', b=B)

            # Fine-tune on predicting the 22nd frame
            context = one_hot_encoding(gen_masks[:,:11], 49)

            for frame in range(11):
                model_optimizer.zero_grad()
                    
                pred = model(context)

                loss = criterion(pred, masks[:, 11 + frame].long())
                loss.backward()

                model_optimizer.step()

                train_loss += loss.item()

                pred = torch.argmax(pred, dim=1)
                pred = one_hot_encoding(pred.unsqueeze(1))
                context = context[:,:,-10:]
                context = torch.cat([context, pred], dim=2)

            if model_scheduler is not None:
                model_scheduler.step()

            if idx % 50 == 0 and epoch < 5:
              print(f"Loss on current batch: {loss.item()}") # we can just take a sample, don't need to average it
            
        avg_epoch_loss = train_loss / len(train_loader * 11)
        end_time = time.time()

        model.eval()
        predicted_masks_22 = []
        actual_masks_22 = []
        predicted_masks_show = []
        show_idx = random.randint(0, len(val_loader))

        val_loss = 0
        #valbar = tqdm(enumerate(valloader), total=len(valloader))
        with torch.no_grad():
            for idx, data in enumerate(val_loader):
                input, label, masks = data
                input, label, masks = input.to(device), label.to(device), masks.to(device)

                context = one_hot_encoding(masks[:,:11], 49)

                if idx == show_idx:
                    actual_masks_show = masks[0,11:].cpu()
                for frame in range(11):
                    pred = model(context)

                    if frame == 0: #criterion on first frame
                        loss = criterion(pred, masks[:,11].long())
                        val_loss += loss.item()

                    pred = torch.argmax(pred, dim=1)

                    if idx == show_idx:
                        predicted_masks_show.append(pred[0].cpu())
                    
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
        plot_model_example(pred[0, 0], masks[0, 0], plot_dir, f'example_epoch_{epoch}')
        plot_model_result(predicted_masks_show, actual_masks_show, plot_dir, f'predictions_12_to_22_epoch_{epoch}')

        # Early Stopping
        if jaccard > early_stop.best_value:
            torch.save(model.module.state_dict() if torch.cuda.device_count() > 1 else model.state_dict(), os.path.join(output_dir, 'models/best',"best_model.pkl"))

        early_stop.step(jaccard, epoch)
        if early_stop.stop_training(epoch):
            print(
                "early stopping at epoch {} since valdiation loss didn't improve from epoch no {}. Best jaccard {}, current jaccard {}".format(
                    epoch, early_stop.best_epoch, early_stop.best_value, jaccard
                ))
            break

        # Used this approach (while and epoch increase) so that we can get back to training the loaded model from checkpoint
        epoch += 1

        # Checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.module.state_dict() if torch.cuda.device_count() > 1 else model.state_dict(),
            'masker_state_dict': masker.module.state_dict() if torch.cuda.device_count() > 1 else masker.state_dict(),
            'model_optimizer_state_dict': model_optimizer.state_dict(),
            'model_scheduler_state_dict': model_scheduler.state_dict() if model_scheduler is not None else None,
            'early_stop': early_stop,
            }, os.path.join(output_dir, 'models/partial', "checkpoint.pkl"))

    return {
        "epochs": epoch,
        "train_loss": train_loss
            }

def compute_jaccard(ground_truth_mask, predicted_mask):
    jaccard = torchmetrics.JaccardIndex(task="multiclass", num_classes=49) #.to(device)
    return jaccard(torch.Tensor(ground_truth_mask), torch.Tensor(predicted_mask))


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
        config = FineTuneConfig.from_json(os.path.join(paths.save_dir, 'used_config.json'))
        config_input = FineTuneConfig.from_json(paths.config_path)
        if config_input != config:
            print("Input config differs from used config: loading used config")
    else:
        config = FineTuneConfig.from_json(paths.config_path)
        output_config_path = os.path.join(paths.save_dir, "used_config.json")
        os.system("cp {} {}".format(paths.config_path, output_config_path))

    num_epochs = config.training.epochs

    print('Loading train data...')
    train_data_dir  = os.path.join(config.data.train.path, 'data')
    train_annotation_dir = os.path.join(config.data.train.path, 'annotations.txt')
    dataloader_train = load_data(train_data_dir, train_annotation_dir, config.data.train.batch_size, mask=True)

    print('Loading val data...')
    val_data_dir  = os.path.join(config.data.val.path, 'data')
    val_annotation_dir = os.path.join(config.data.val.path, 'annotations.txt')
    dataloader_val = load_data(val_data_dir, val_annotation_dir, config.data.val.batch_size, mask=True)

    epoch = 0

    model = Seq2Seq(**config.lstm_model.args, device=device)
    model_checkpoint = torch.load(config.lstm_model.path, map_location=device)
    model.load_state_dict(model_checkpoint)
    model.to(device)
    print('LSTM Loaded.')

    masker = UNet(**config.masker_model.args)
    unet_checkpoint = torch.load(config.masker_model.path, map_location=device)
    masker.load_state_dict(unet_checkpoint)
    masker.to(device)
    print('UNet Loaded.')

    model_optimizer = getattr(optim, config.lstm_optimizer.name)(params=model.parameters(), **config.lstm_optimizer.args)

    total_steps = num_epochs * len(dataloader_train)

    model_scheduler = getattr(optim.lr_scheduler, config.lstm_lr_scheduler.name)(model_optimizer, T_max=total_steps, **config.lstm_lr_scheduler.args)

    class_weights = torch.ones(config.masker_model.args["n_class"])
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
            model_optimizer.load_state_dict(checkpoint['model_optimizer_state_dict'])
            if model_scheduler is not None:
                model_scheduler.load_state_dict(checkpoint['model_scheduler_state_dict'])
            epoch = checkpoint['epoch']
            early_stop = checkpoint['early_stop']
            print(f'Resuming finetuning from epoch {epoch}')
        except :
            print("Couldn't load model from checkpoint, starting again finetuning from epoch 0")
    
    # Pass model to DataParallel (Could improve by using DistributedDataParallel)
    if torch.cuda.device_count() > 1:
        print("Model training will be distributed to {} GPUs.".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)
        masker = nn.DataParallel(masker)
    model.to(device)
    masker.to(device)


    print('Start finetuning model...')
    results = finetune(epoch, model, masker, dataloader_train, dataloader_val, criterion, model_optimizer, model_scheduler, early_stop, num_epochs, device, paths.save_dir)

    print(f'Model finetuning finshed at epoch {results["epochs"]}, trainig loss: {results["train_loss"]}')

