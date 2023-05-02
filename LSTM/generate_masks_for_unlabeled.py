from UNet_model import UNet
from video_dataset import VideoFrameDataset, ImglistToTensor
from tqdm import tqdm
from einops import rearrange
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import copy
from argparse import ArgumentParser, Namespace


def parse_args() -> Namespace:
    parser = ArgumentParser("Create Masks for Unlabeled Data")
    parser.add_argument("--unlabeled", help="Name of dir with data", required=True, type=str)
    parser.add_argument("--model", help="Path to model checkpoint", required=True, type=str)
    parser.add_argument("--output-dir", help="Name of dir to save the data", required=True, type=str)

    return parser.parse_args()


### Unlabeled Dataloader method
def load_unlabeled_data(root, annotation_file, batch_size=1):
  preprocess = transforms.Compose([
          ImglistToTensor(),  # list of PIL images to (FRAMES x CHANNELS x HEIGHT x WIDTH) tensor
          transforms.Normalize((0.61749697, 0.6050092, 0.52180636), (2.1824553, 2.1553133, 1.9115673)),
      ])

  dataset = VideoFrameDataset(
      root_path=root,
      annotationfile_path=annotation_file,
      num_segments=1,
      frames_per_segment=22,
      imagefile_template='image_{:d}.png',
      transform=preprocess,
      mask=False,
      test_mode=False
  )

  dataloader = torch.utils.data.DataLoader(
          dataset=dataset,
          batch_size=batch_size,
          shuffle=False,
          num_workers=1,
          pin_memory=True
      )
  return dataloader


def generate_unlabeled_masks(model, unlabeledloader, save_dir='output_masks', device='cuda'):
  model.eval()
  #unlabeled_tqdm = tqdm(unlabeledloader, desc=f"(Generating)")
  if not os.path.exists(save_dir):
    os.makedirs(save_dir)

  for idx, (frames, labels) in enumerate(unlabeledloader):
    frames, labels = frames.to(device), labels.to(device)

    with torch.no_grad():
      frames = rearrange(frames, 'b t c h w -> (b t) c h w')
      outputs = model(frames)
      outputs = torch.argmax(outputs, dim=1)
      output_file = os.path.join(save_dir, f'mask_{idx}.pt')
      torch.save(outputs.cpu(), output_file)


if __name__ == "__main__":
  args = parse_args()

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(f"Using device: {device}")

  Unet_masker = UNet(
    n_class=49,
  )

  unlabeled_data_dir  = os.path.join(args.unlabeled, 'data')
  unlabeled_annotation_dir = os.path.join(args.unlabeled, 'annotations.txt')

  ### Load model:
  unet_saved_data = torch.load(args.model, map_location=device) # do the correct os.join thing
  Unet_masker.load_state_dict(unet_saved_data)
  Unet_masker.to(device)

  unlabeledloader = load_unlabeled_data(unlabeled_data_dir, unlabeled_annotation_dir, batch_size=1)

  generate_unlabeled_masks(Unet_masker, unlabeledloader, args.output_dir, device)

