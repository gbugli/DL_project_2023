import os
import torch
import torch.nn as nn
from einops import rearrange
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

from models import IJEPA_base

def parse_args() -> Namespace:
    parser = ArgumentParser("JEPA")
    # parser.add_argument("--config-file-name", required=True, type=str, help="Name of json file with config")
    parser.add_argument("--root", help="Name of dir with data", required=True, type=str)
    parser.add_argument("--annotation-file", help="Name of dir with annotation files",
                        required=True, type=str)
    parser.add_argument("--output-dir", required=True, type=str, help="Name of dir to save the checkpoints to")
    parser.add_argument("--resume", default=False, type=bool, help="In case training was not completed resume from last epoch")

    return parser.parse_args()

def load_data(root, annotation_file):
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
            batch_size=2,
            shuffle=True,
            num_workers=1,
            pin_memory=True
        )
    return dataloader

# Train the model
def train_model(epoch, model, criterion, optimizer, scheduler, dataloader, num_epochs, output_dir, device):
    while epoch < num_epochs:
        model.train()
        train_loss = 0
        for i, data in enumerate(dataloader, 0):
            inputs, labels = data
            #inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            prediction_blocks, target_blocks = model(inputs.transpose(1, 2))
            loss = criterion(prediction_blocks, target_blocks)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()

            # Update the learning rate using the scheduler
            scheduler.step()
        
        avg_epoch_loss = train_loss / len(dataloader)
        #scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch: {epoch + 1}, Learning Rate: {current_lr:.6f}, Average epoch loss: {avg_epoch_loss:.4f}")

        # TO DO: Implement Early Stopping?? Based on what?

        # Used this approach (while and epoch increase) so that we can get back to training the loaded model from checkpoint
        epoch += 1

        # Checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.module.state_dict() if torch.cuda.device_count() > 1 else model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            #'early_stop': early_stop,
            }, os.path.join(output_dir, 'models/partial', "checkpoint.pkl"))

    return {
        "epochs": epoch,
        "train_loss": train_loss,
        "model": model
            }

if __name__ == "__main__":

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    args = parse_args()
    dataloader = load_data(args.root,args.annotation_file)

    # All these hyperparamters we might want to have a config file to choose them and use a custom Config class to parse
    num_epochs = 10
    total_steps = num_epochs * len(dataloader)
    div_factor = 5 # max_lr/div_factor = initial lr
    final_div_factor = 10 # final lr is initial_lr/final_div_factor 

    # Used this approach so that we can get back to training the loaded model from checkpoint
    epoch = 0

    # get these params from a global config?
    model = IJEPA_base(img_size=128, patch_size=8, enc_depth=6, pred_depth=6, num_heads=8)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.003, total_steps=total_steps, div_factor=div_factor, final_div_factor=final_div_factor)

    if args.resume:
        print("Attempting to find existing checkpoint")
        path_partials = os.path.join(args.output_dir, "models/partial")
        if os.path.exists(path_partials):
            checkpoint = torch.load(os.path.join(path_partials, "checkpoint.pkl"), map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            epoch = checkpoint['epoch']

    results = train_model(epoch, model, criterion, optimizer, scheduler, dataloader, num_epochs, args.output_dir, device)
    print(f'Model training finshed at epoch {results["epoch"]}, trainig loss: {results["train_loss"]}')
