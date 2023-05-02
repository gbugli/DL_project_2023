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
from UNet_model import UNet

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

from torch.nn.modules.loss import CrossEntropyLoss

from losses import MaskLoss
    
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
  for epoch in range(epochs):

    # Training
    model.train()
    train_loss = 0
    for frames, labels, masks in trainloader:
      frames, labels, masks = frames.to(device), labels.to(device), masks.to(device)

      optimizer.zero_grad()
      
      # reshape inputs
      frames = rearrange(frames, 'b t c h w -> (b t) c h w')
      masks = rearrange(masks, 'b t h w -> (b t) h w')

      # forward pass
      outputs = model(frames)

      # compute loss
      loss = criterion(outputs, masks.long())
      loss.backward()

      # update weights
      optimizer.step()

      # update learning rate
      scheduler.step()

      train_loss += loss.item()

    print(f'train_loss: {train_loss / (len(trainloader) + 1)}, learning_rate: {optimizer.param_groups[0]["lr"]}')

    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
      val_predicted = []
      val_truth = []
      for frames, labels, masks in valloader:
        frames, labels, masks = frames.to(device), labels.to(device), masks.to(device)

        # reshape inputs
        frames = rearrange(frames, 'b t c h w -> (b t) c h w')
        masks = rearrange(masks, 'b t h w -> (b t) h w')

        # forward pass
        outputs = model(frames)

        # compute loss
        loss = criterion(outputs, masks.long())

        # convert outputs to predicted masks
        outputs = torch.argmax(outputs, dim=1)

        # save predicted and real masks for jaccard index
        # BEWARE: this only works with batch size 2
        val_predicted.append(outputs[[21,43]].cpu())
        val_truth.append(masks[[21,43]].cpu())

        val_loss += loss.item()

      print(f'val_loss: {val_loss / (len(valloader) + 1)}')

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

      # Compute the jaccard index
      results_mask_tensor = torch.cat(val_predicted, dim=0)
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

  Unet_masker = UNet(
      n_class=49,
  )

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


  num_epochs = 20
  weight = torch.ones(49)
  weight[0] = 0.3
  weight = weight.to(device)
  steps = num_epochs * len(trainloader)
  criterion = MaskLoss(weight, 0.2, 0.5, 0.3)
  optimizer = optim.Adam(Unet_masker.parameters(), lr=0.0001)
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