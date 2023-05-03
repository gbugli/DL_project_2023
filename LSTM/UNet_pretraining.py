import os
import copy
import numpy as np
from PIL import Image
from typing import List, Union, Tuple, Any

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torchvision import transforms

import torchmetrics
from einops import rearrange

import matplotlib.pyplot as plt

from config import UNetConfig
from argparse import ArgumentParser, Namespace
from utils.file_utils import create_output_dirs, PathsContainer

from dataset.video_dataset import VideoFrameDataset, ImglistToTensor

from models.UNet import UNet
from utils.early_stop import EarlyStop
import losses

# import segmentation_models_pytorch as smp



def parse_args() -> Namespace:
    parser = ArgumentParser("UNet")
    parser.add_argument("--config-file-name", required=True, type=str, help="Name of json file with config")
    parser.add_argument("--output-dir", help="Name of dir to save the checkpoints to", required=True, type=str)
    parser.add_argument("--run-id", help="Name of the run", required=True, type=str)
    parser.add_argument("--resume", help="In case training was not completed resume from last epoch", default=False, type=bool)

    return parser.parse_args()


def load_data(root, annotation_file, batch_size=2, mask=False, shuffle=True):
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
        mask=mask,
        test_mode=False
    )

    dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=2, # arbitrarily chosen
            pin_memory=True
        )
    return dataloader

def train_model(epoch, model, trainloader, valloader, optimizer, criterion, scheduler, early_stop, device, num_epochs, output_dir):
    while epoch < num_epochs:
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

        avg_epoch_loss = train_loss / len(trainloader)

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
                val_predicted.append(outputs[[(i+1)*22 -1 for i in range(frames.shape[0])]].cpu())
                val_truth.append(masks[[(i+1)*22 -1 for i in range(frames.shape[0])]].cpu())

                val_loss += loss.item()
        
        avg_epoch_val_loss = val_loss / len(valloader)

        # Compute the jaccard index
        results_mask_tensor = torch.cat(val_predicted, dim=0)
        actual_mask_tensor = torch.cat(val_truth, dim=0)
        jaccard = compute_jaccard(results_mask_tensor, actual_mask_tensor)

        print(f"Epoch: {epoch}, Average epoch loss: {avg_epoch_loss:.4f}, Average epoch val loss: {avg_epoch_val_loss:.4f}, Jaccard Score: {jaccard}")

        # Choose the first sample in the batch to visualize
        sample_id = 0

        # Plot the predicted mask and real mask
        plot_dir = os.path.join(output_dir, 'training_plots')
        plot_model_example(outputs[sample_id], masks[sample_id], plot_dir, f'example_epoch_{epoch}')

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
        "train_loss": avg_epoch_loss,
        "model": model
            }



def compute_jaccard(ground_truth_mask, predicted_mask):
    jaccard = torchmetrics.JaccardIndex(task="multiclass", num_classes=49) #.to(device)
    return jaccard(torch.Tensor(ground_truth_mask), torch.Tensor(predicted_mask))


def plot_model_example(prediction, target, output_dir, fig_name='example'):
      # prediction = torch.argmax(prediction, dim=0)
      fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
      ax1.imshow(prediction.cpu().numpy())
      ax1.set_title("Predicted Next Frame Mask")
      ax2.imshow(target.cpu().numpy())
      ax2.set_title("Actual Next Frame Mask")
      plt.savefig(f'{output_dir}/{fig_name}.pdf')


if __name__ == '__main__':

  args = parse_args()

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  paths = PathsContainer.from_args(args.output_dir, args.run_id, args.config_file_name)
  create_output_dirs(paths.save_dir)

  if args.resume:
      print("Resuming training...")
      config = UNetConfig.from_json(os.path.join(paths.save_dir, 'used_config.json'))
      config_input = UNetConfig.from_json(paths.config_path)
      if config_input != config:
          print("Input config differs from used config: loading used config")
  else:
      config = UNetConfig.from_json(paths.config_path)
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

  model = UNet(**config.unet_model.args)
  model.to(device)
  print('UNet Loaded.')

  optimizer = getattr(optim, config.optimizer.name)(params=model.parameters(), **config.optimizer.args)

  total_steps = num_epochs * len(dataloader_train)

  scheduler = getattr(optim.lr_scheduler, config.lr_scheduler.name)(optimizer, T_max=total_steps, **config.lr_scheduler.args)

  class_weights = torch.ones(config.unet_model.args["n_class"])
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

  if torch.cuda.device_count() > 1:
      print("Model training will be distributed to {} GPUs.".format(torch.cuda.device_count()))
      model = nn.DataParallel(model)
  model.to(device)

  print('Start training model...')
  train_model(epoch, model, dataloader_train, dataloader_val, optimizer, criterion, scheduler, early_stop, device, num_epochs, paths.save_dir)