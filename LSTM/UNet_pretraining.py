import os
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
import copy

import os
import os.path
import numpy as np
from PIL import Image
from torchvision import transforms
import torch
from typing import List, Union, Tuple, Any


from torch.optim.lr_scheduler import CosineAnnealingLR
from torch import optim
import torchmetrics
# import segmentation_models_pytorch as smp

from einops import rearrange
from argparse import ArgumentParser, Namespace

from video_dataset import VideoFrameDataset, ImglistToTensor

# def parse_args() -> Namespace:
#     parser = ArgumentParser("JEPA")
#     # parser.add_argument("--config-file-name", required=True, type=str, help="Name of json file with config")
#     parser.add_argument("--root", help="Name of dir with data", required=True, type=str)
#     parser.add_argument("--val-dir", help="Name of dir with validation data", required=True, type=str)
#     parser.add_argument("--output-dir", help="Name of dir to save the checkpoints to", required=True, type=str)
#     parser.add_argument("--run-id", help="Name of the run", required=True, type=str)
#     parser.add_argument("--resume", help="In case training was not completed resume from last epoch", default=False, type=bool)

#     return parser.parse_args()

def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )   


class UNet(nn.Module):

    def __init__(self, n_class):
        super().__init__()
                
        self.dconv_down1 = double_conv(3, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv(256, 512)        

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        
        
        self.dconv_up3 = double_conv(256 + 512, 256)
        self.dconv_up2 = double_conv(128 + 256, 128)
        self.dconv_up1 = double_conv(128 + 64, 64)
        
        self.conv_last = nn.Conv2d(64, n_class, 1)
        
        
    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        
        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)   
        
        x = self.dconv_down4(x)
        
        x = self.upsample(x)        
        x = torch.cat([x, conv3], dim=1)
        
        x = self.dconv_up3(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv2], dim=1)       

        x = self.dconv_up2(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv1], dim=1)   
        
        x = self.dconv_up1(x)
        
        out = self.conv_last(x)
        
        return out
    

from torch.nn.modules.loss import CrossEntropyLoss

class CEandDiceLoss(nn.Module):
    def __init__(self, ce_class_weights, ce_weight=0.5, dice_weight=0.5):
        super(CEandDiceLoss, self).__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.ce_loss = nn.CrossEntropyLoss()
        # self.dice_loss = smp.losses.DiceLoss(mode="multiclass")
        self.dice_loss = torchmetrics.Dice()
    
    def forward(self, pred, target):
        ce_loss = self.ce_loss(pred, target)
        dice_loss = self.dice_loss(pred, target)
        combined_loss = self.ce_weight * ce_loss + self.dice_weight * dice_loss

        return combined_loss
    
def compute_jaccard(ground_truth_mask, predicted_mask):
  jaccard = torchmetrics.JaccardIndex(task="multiclass", num_classes=49) #.to(device)
  return jaccard(torch.Tensor(ground_truth_mask), torch.Tensor(predicted_mask))



def load_data(root, annotation_file, batch_size=2):
    preprocess = transforms.Compose([
            ImglistToTensor(),  # list of PIL images to (FRAMES x CHANNELS x HEIGHT x WIDTH) tensor
            # transforms.Resize(299),  # image batch, resize smaller edge to 299
            # transforms.Resize((160,240)),
            # transforms.CenterCrop(299),  # image batch, center crop to square 299x299
            transforms.Normalize((0.61749697, 0.6050092, 0.52180636), (2.1824553, 2.1553133, 1.9115673)),
        ])

    dataset = VideoFrameDataset(
        root_path=root,
        annotationfile_path=annotation_file,
        num_segments=1,
        frames_per_segment=22,
        imagefile_template='image_{:d}.png',
        transform=preprocess,
        mask=True,
        test_mode=False
    )

    dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2, # arbitrarily chosen
            pin_memory=True
        )
    return dataloader

def train_model(model, trainloader, valloader, optimizer, criterion, device, epochs):
  # Training loop
  for epoch in range(epochs):
      # Training
    model.train()
    train_loss = 0
      #train_tqdm = tqdm(trainloader, desc=f"Epoch {epoch+1}/{num_epochs} (Train)")
    #   for frames, labels, masks in train_tqdm:
    for frames, labels, masks in trainloader:
        frames, labels, masks = frames.to(device), labels.to(device), masks.to(device)

        optimizer.zero_grad()
        frames = rearrange(frames, 'b t c h w -> (b t) c h w')
        masks = rearrange(masks, 'b t h w -> (b t) h w')
        #print(frames.shape)
        outputs = model(frames)
        #print(outputs.shape)
        #print(masks.shape)
        loss = criterion(outputs, masks.long())
        loss.backward()
        optimizer.step()
        scheduler.step()

        train_loss += loss.item()
    print(f'train_loss: {train_loss / (len(trainloader) + 1)}, learning_rate: {optimizer.param_groups[0]["lr"]}')
        #train_tqdm.set_postfix({'train_loss': train_loss / (train_tqdm.n + 1), 'learning_rate': optimizer.param_groups[0]['lr']})

      # Validation
    model.eval()
    val_loss = 0
    # val_tqdm = tqdm(valloader, desc=f"Epoch {epoch+1}/{num_epochs} (Val)")
    with torch.no_grad():
        val_predicted = []
        val_truth = []
        for frames, labels, masks in valloader:
            frames, labels, masks = frames.to(device), labels.to(device), masks.to(device)

            frames = rearrange(frames, 'b t c h w -> (b t) c h w')
            masks = rearrange(masks, 'b t h w -> (b t) h w')
            outputs = model(frames)
            loss = criterion(outputs, masks.long())

            outputs = torch.argmax(outputs, dim=1)
            val_predicted.append(outputs[[21,43]].cpu())
            val_truth.append(masks[[21,43]].cpu())

            val_loss += loss.item()
        print(f'val_loss: {val_loss / (len(valloader) + 1)}')
            #val_tqdm.set_postfix({'val_loss': val_loss / (val_tqdm.n + 1)})

        # Convert the tensors to numpy arrays for visualization
        outputs = outputs.cpu().numpy()
        masks = masks.cpu().numpy()
        # Choose the first sample in the batch to visualize
        sample_id = 0

        # Plot the predicted mask and real mask
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        ax1.imshow(outputs[sample_id])
        ax1.set_title("Predicted Mask")
        ax2.imshow(masks[sample_id])
        ax2.set_title("Real Mask")
        plt.savefig(f'example_{epoch}')

        results_mask_tensor = torch.cat(val_predicted, dim=0)  # Concatenate tensors
        actual_mask_tensor = torch.cat(val_truth, dim=0)
        jaccard = compute_jaccard(results_mask_tensor, actual_mask_tensor)
        print(f"jaccard_score: {jaccard}")

if __name__ == '__main__':
    ### Load the train and validation datasets
    batch_size = 2
    trainloader = load_data('/train/data/', '/train/annotations.txt', batch_size)
    valloader = load_data('/val/data/', '/val/annotations.txt', batch_size)

    model_dir = '/scratch/gb2572/DL_project_2023/output/unet/best_model_15_epochs.pkl'
    print('Datasets Loaded')

    Unet_masker = UNet(                 # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        n_class=49,                      # model output channels (number of classes in your dataset)
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    num_epochs = 5
    #criterion = smp.losses.JaccardLoss(mode="multiclass")
    weight = torch.ones(49)
    weight[0] = 0.3
    weight = weight.to(device)
    steps = num_epochs * len(trainloader)
    criterion = CEandDiceLoss(weight, 0.3, 0.7) #smp.losses.DiceLoss(mode="multiclass")#CrossEntropyLoss(weight=weight)
    optimizer = optim.Adam(Unet_masker.parameters(), lr=0.000001)
    scheduler = CosineAnnealingLR(optimizer, T_max=steps, eta_min=0.00000001)

    if model_dir is not None:
        unet_saved_data = torch.load(model_dir, map_location=device)
        Unet_masker.load_state_dict(unet_saved_data)
        Unet_masker.to(device)

    Unet_masker.to(device)
    criterion.to(device)

    ### To see a couple of predicted masks
    # for idx, data in enumerate(valloader):
    #     Unet_masker.eval()
    #     frames, labels, masks = data
    #     frames, labels, masks = frames.to(device), labels.to(device), masks.to(device)
    #     frames = rearrange(frames, 'b t c h w -> (b t) c h w')
    #     masks = rearrange(masks, 'b t h w -> (b t) h w')
    #     outputs = Unet_masker(frames)
    #     loss = criterion(outputs, masks.long())

    #     outputs = torch.argmax(outputs, dim=1)
    # # Plot the predicted mask and real mask
    #     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    #     ax1.imshow(outputs[0].cpu().numpy())
    #     ax1.set_title("Predicted Mask")
    #     ax2.imshow(masks[0].cpu().numpy())
    #     ax2.set_title("Real Mask")
    #     plt.savefig(f'example_{idx}')

    #     if idx == 5:
    #         break


    train_model(Unet_masker, trainloader, valloader, optimizer, criterion, device, num_epochs)