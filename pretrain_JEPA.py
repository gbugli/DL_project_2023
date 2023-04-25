import os
import gc
import torch
import torch.nn as nn
from einops import rearrange
import time
# import imageio.v3 as iio
import numpy as np
import copy
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import math
import torch.nn.functional as F
import warnings
from video_dataset import VideoFrameDataset, ImglistToTensor
from argparse import ArgumentParser, Namespace

from models import IJEPA_base, CustomDataParallel, EarlyStop

def parse_args() -> Namespace:
    parser = ArgumentParser("JEPA")
    # parser.add_argument("--config-file-name", required=True, type=str, help="Name of json file with config")
    parser.add_argument("--root", help="Name of dir with data", required=True, type=str)
    parser.add_argument("--val-dir", help="Name of dir with validation data", required=True, type=str)
    parser.add_argument("--output-dir", help="Name of dir to save the checkpoints to", required=True, type=str)
    parser.add_argument("--run-id", help="Name of the run", required=True, type=str)
    parser.add_argument("--resume", help="In case training was not completed resume from last epoch", default=False, type=bool)

    return parser.parse_args()

def load_data(root, annotation_file, batch_size=2):
    preprocess = transforms.Compose([
            ImglistToTensor(),  # list of PIL images to (FRAMES x CHANNELS x HEIGHT x WIDTH) tensor
            # transforms.Resize(299),  # image batch, resize smaller edge to 299
            transforms.Resize((128,128)),
            # transforms.CenterCrop(299),  # image batch, center crop to square 299x299
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    dataset = VideoFrameDataset(
        root_path=root,
        annotationfile_path=annotation_file,
        num_segments=1,
        frames_per_segment=22,
        imagefile_template='image_{:d}.png',
        transform=preprocess,
        test_mode=False
    )

    dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=12,
            pin_memory=True
        )
    return dataloader

def load_validation_data(val_dir, val_annotation_file, batch_size=2):
    preprocess = transforms.Compose([
            ImglistToTensor(),  # list of PIL images to (FRAMES x CHANNELS x HEIGHT x WIDTH) tensor
            # transforms.Resize(299),  # image batch, resize smaller edge to 299
            transforms.Resize((128,128)),
            # transforms.CenterCrop(299),  # image batch, center crop to square 299x299
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    dataset = VideoFrameDataset(
        root_path=val_dir,
        annotationfile_path=val_annotation_file,
        num_segments=1,
        frames_per_segment=22,
        imagefile_template='image_{:d}.png',
        transform=preprocess,
        test_mode=False
    )

    dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=12,
            pin_memory=True
        )
    return dataloader

# Train the model
def train_model(epoch, model, criterion, optimizer, scheduler, dataloader, val_dataloader, num_epochs, output_dir, device, early_stop):
    while epoch < num_epochs:
        print(f'Starting epoch {epoch + 1}')
        start_time = time.time()
        model.train()
        train_loss = 0
        for i, data in enumerate(dataloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            prediction_blocks, target_blocks = model(inputs.transpose(1, 2))
            loss = criterion(prediction_blocks, target_blocks)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()

            if i % 50 == 0 and epoch < 5:
              print(f"Current loss: {loss.item()}") # we can just take a sample, don't need to average it

            # Update the learning rate using the scheduler
            if scheduler is not None:
                scheduler.step()
        
        avg_epoch_loss = train_loss / len(dataloader)
        end_time = time.time()
        #scheduler.step()

        ### Validation
        model.eval()
        with torch.no_grad():
          val_loss = 0
          for i, data in enumerate(val_dataloader, 0):
              inputs, labels = data
              inputs, labels = inputs.to(device), labels.to(device)

              prediction_blocks, target_blocks = model(inputs.transpose(1, 2))
              loss = criterion(prediction_blocks, target_blocks)
              val_loss += loss.item()
          
          avg_epoch_val_loss = val_loss / len(val_dataloader)
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch: {epoch + 1}, Time for training epoch {end_time - start_time}, Learning Rate: {current_lr:.6f}, Average epoch loss: {avg_epoch_loss:.4f}, Average epoch val loss: {avg_epoch_val_loss:.4f}")

        # Early Stopping
        if avg_epoch_val_loss < early_stop.best_value:
            torch.save(model.module.state_dict() if torch.cuda.device_count() > 1 else model.state_dict(), os.path.join(output_dir, 'models/best',"best_model.pkl"))
            # if torch.cuda.device_count() > 1:
            #     torch.save(model.module.state_dict(), os.path.join(output_dir, 'models/best',"best_model.pkl"))
            # else:
            #     torch.save(model.state_dict(), os.path.join(output_dir, 'models/best',"best_model.pkl"))

        early_stop.step(avg_epoch_val_loss, epoch)
        if early_stop.stop_training(epoch):
            print(
                "early stopping at epoch {} since valdiation loss didn't improve from epoch no {}. Best value {}, current value {}".format(
                    epoch, early_stop.best_epoch, early_stop.best_value, avg_epoch_val_loss
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

if __name__ == "__main__":

    torch.cuda.empty_cache()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # All these hyperparamters we might want to have a config file to choose them and use a custom Config class to parse
    num_epochs = 100
    div_factor = 10 # max_lr/div_factor = initial lr
    final_div_factor = 100 # final lr is initial_lr/final_div_factor 
    batch_size = 8
    patience = 10

    args = parse_args()

    # Make run dir
    if not os.path.exists(os.path.join(args.output_dir,args.run_id)):
        os.makedirs(os.path.join(args.output_dir,args.run_id), exist_ok=True)
    
    save_dir = os.path.join(args.output_dir,args.run_id)
    os.makedirs(os.path.join(save_dir, "models/partial"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "models/best"), exist_ok=True)

    # Load unlabeled data and validation data
    unlabeled_data_dir  = os.path.join(args.root, 'data')
    unlabeled_annotation_dir = os.path.join(args.root, 'annotations.txt')
    val_data_dir  = os.path.join(args.val_dir, 'data')
    val_annotation_dir = os.path.join(args.val_dir, 'annotations.txt')

    print('Loading train data...')
    dataloader = load_data(unlabeled_data_dir, unlabeled_annotation_dir, batch_size)
    print('Loading val data...')
    val_dataloader = load_validation_data(val_data_dir, val_annotation_dir, batch_size)

    # Used this approach so that we can get back to training the loaded model from checkpoint
    epoch = 0

    # get these params from a global config?
    model = IJEPA_base(img_size=128, patch_size=8, in_chans=3, norm_layer=nn.LayerNorm, num_frames=22, attention_type='divided_space_time', dropout=0.1, mode="train", M=4, embed_dim=384, device=device,
                        # encoder parameters
                        enc_depth=10,
                        enc_num_heads=6,
                        enc_mlp_ratio=4.,
                        enc_qkv_bias=False,
                        enc_qk_scale=None,
                        enc_drop_rate=0.,
                        enc_attn_drop_rate=0.,
                        enc_drop_path_rate=0.1,
                        # predictor parameters
                        pred_depth=10,
                        pred_num_heads=6,
                        pred_mlp_ratio=4.,
                        pred_qkv_bias=False,
                        pred_qk_scale=None,
                        pred_drop_rate=0.1,
                        pred_attn_drop_rate=0.1,
                        pred_drop_path_rate=0.1,
                        # positional and spacial embedding parameters
                        pos_drop_rate=0.1,
                        time_drop_rate=0.1)
    # Leave this to load properly the optimizer dict
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.005)

    # Define One Cycle LR Scheduler
    total_steps = num_epochs * len(dataloader)
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.003, total_steps=total_steps, div_factor=div_factor, final_div_factor=final_div_factor)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=0.000001)
    # scheduler = None
    early_stop = EarlyStop(patience, loss=True)

    if args.resume:
        print("Attempting to find existing checkpoint")
        path_partials = os.path.join(save_dir, "models/partial")
        if os.path.exists(path_partials):
            checkpoint = torch.load(os.path.join(path_partials, "checkpoint.pkl"), map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if scheduler is not None:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            epoch = checkpoint['epoch']
            early_stop = checkpoint['early_stop']
            print(f'Resuming training from epoch {epoch}')
    
    if torch.cuda.device_count() > 1:
        # model = CustomDataParallel(model)
        print("Model training will be distributed to {} GPUs.".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)
    model.to(device)

    print('Start training model...')
    results = train_model(epoch, model, criterion, optimizer, scheduler, dataloader, val_dataloader, num_epochs, save_dir, device, early_stop)
    torch.save(model.module.state_dict() if torch.cuda.device_count() > 1 else model.state_dict(), os.path.join(save_dir, 'models', "final_model.pkl"))
    print(f'Model training finshed at epoch {results["epoch"]}, trainig loss: {results["train_loss"]}')
