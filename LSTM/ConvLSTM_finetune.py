import os
import os.path
import numpy as np
from PIL import Image
from torchvision import transforms
import torch
from typing import List, Union, Tuple, Any

import time

import torch
import torch.nn as nn

import numpy as np
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

import io
from ipywidgets import widgets, HBox
import torch.nn.functional as F
from torch.utils.data import Dataset
from einops import rearrange
import matplotlib.pyplot as plt

from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchmetrics

from argparse import ArgumentParser, Namespace

from video_dataset import VideoFrameDataset, ImglistToTensor, MaskDataset

from early_stop import EarlyStop

from Seq2Seq import Seq2Seq
from UNet_pretraining import UNet

def parse_args() -> Namespace:
    parser = ArgumentParser("ConvLSTM")
    # parser.add_argument("--config-file-name", required=True, type=str, help="Name of json file with config")
    parser.add_argument("--train-dir", help="Name of dir with labeled training data", required=True, type=str)
    parser.add_argument("--val-dir", help="Name of dir with validation data", required=True, type=str)
    parser.add_argument("--output-dir", help="Name of dir to save the checkpoints to", required=True, type=str)
    parser.add_argument("--run-id", help="Name of the run", required=True, type=str)
    parser.add_argument("--masker-dir", help="Name of dir of masker checkpoint", required=True, type=str)
    parser.add_argument("--model-dir", help="Name of dir of model checkpoint", required=True, type=str)
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

def one_hot_encoding(input_tensor, num_classes=49):
    one_hot = F.one_hot(input_tensor.long(), num_classes)
    #print(one_hot.shape)
    one_hot = rearrange(one_hot, 'b f h w c -> b c f h w')  # Change the dimensions to (batch_size, num_classes, height, width)
    return one_hot


def finetune(epoch, model, masker, train_loader, val_loader, criterion, model_optimizer, masker_optimizer, model_scheduler, masker_scheduler, early_stop, num_epochs, device, output_dir):

    while epoch < num_epochs:
        print(f'Starting epoch {epoch + 1}')
        start_time = time.time()
        model.train()
        train_loss = 0                                            
        # pbar = tqdm(enumerate(dataloader), total=len(dataloader))
        for idx, data in enumerate(train_loader):
            frames, labels, masks = data
            frames, labels, masks = frames.to(device), labels.to(device), masks.to(device)

            B = frame.shape[0]

            model_optimizer.zero_grad()
            masker_optimizer.zero_grad()

            frames = rearrange(frames, 'b t c h w -> (b t) c h w')
            masks = rearrange(masks, 'b t h w -> (b t) h w')
            
            gen_masks = masker(frames)
            gen_masks = rearrange(gen_masks, '(b t) h w -> b t h w', b=B)

            rand = np.random.randint(11,21)
            input = gen_masks[:,rand-11:rand, :, :]
            target_mask = masks[:,rand,:,:]
            input = one_hot_encoding(input, 49)   

            output = model(input)
                                        
            loss = criterion(output, target_mask)       
            loss.backward()      

            model_optimizer.step()
            masker_optimizer.step()   

            train_loss += loss.item()

            if model_scheduler is not None:
                model_scheduler.step()
            
            if masker_scheduler is not None:
                masker_scheduler.step()

            if idx % 50 == 0 and epoch < 5:
              print(f"Loss on current batch: {loss.item()}") # we can just take a sample, don't need to average it
            
        avg_epoch_loss = train_loss / len(train_loader)
        end_time = time.time()

        model.eval()
        predicted_masks_22 = []
        actual_masks_22 = []
        val_loss = 0
        #valbar = tqdm(enumerate(valloader), total=len(valloader))
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
            torch.save(model.module.state_dict() if torch.cuda.device_count() > 1 else model.state_dict(), os.path.join(output_dir, 'finetune/best',"best_model.pkl"))
            torch.save(masker.module.state_dict() if torch.cuda.device_count() > 1 else masker.state_dict(), os.path.join(output_dir, 'finetune/best',"best_masker.pkl"))

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
            'masker_state_dict': masker.module.state_dict() if torch.cuda.device_count() > 1 else model.state_dict(),
            'model_optimizer_state_dict': model_optimizer.state_dict(),
            'masker_optimizer_state_dict': masker_optimizer.state_dict(),
            'model_scheduler_state_dict': model_scheduler.state_dict() if model_scheduler is not None else None,
            'masker_scheduler_state_dict': masker_scheduler.state_dict() if masker_scheduler is not None else None,
            'early_stop': early_stop,
            }, os.path.join(output_dir, 'finetune/partial', "checkpoint.pkl"))

    return {
        "epochs": epoch,
        "train_loss": train_loss,
        "model": model
            }


if __name__ == "__main__":

    args = parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create all the paths to save models and files if don't already exist
    save_dir = os.path.join(args.output-dir,args.run_id)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, "finetune/partial"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "finetune/best"), exist_ok=True)
    # os.makedirs(os.path.join(save_dir, "training_plots"), exist_ok=True)

    # # instead of the above do the following
    # paths = PathsContainer.from_args(args.output_dir, args.run_id, args.config_file_name)
    # create_output_dirs(paths.save_dir)

    # if args.resume:
    #     print("Resuming training...")
    #     config = Config.from_json(os.path.join(paths.save_dir, 'used_config.json'))
    #     config_input = Config.from_json(paths.config_path)
    #     if config_input != config:
    #         logger.warning("Input config differs from used config: loading used config")
    # else:
    #     config = Config.from_json(paths.config_path)
    #     output_config_path = os.path.join(paths.save_dir, "used_config.json")
    #     os.system("cp {} {}".format(paths.config_path, output_config_path))

    # Parameters (TO DO: Make a config file for all of them)
    num_epochs = 20
    # div_factor = 10 # max_lr/div_factor = initial lr
    # final_div_factor = 100 # final lr is initial_lr/final_div_factor 
    batch_size = 8
    patience = 15

    print('Loading train data...')
    train_data_dir  = os.path.join(args.train-dir, 'data')
    train_annotation_dir = os.path.join(args.train-dir, 'annotations.txt')
    dataloader_train = load_data(train_data_dir, train_annotation_dir, batch_size, mask=True)

    print('Loading val data...')
    val_data_dir  = os.path.join(args.val-dir, 'data')
    val_annotation_dir = os.path.join(args.val-dir, 'annotations.txt')
    dataloader_val = load_data(val_data_dir, val_annotation_dir, batch_size, mask=True)

    # Used this approach so that we can get back to training the loaded model from checkpoint
    epoch = 0

    # Define LSTM
    model = Seq2Seq(num_channels=49, num_kernels=64, kernel_size=(3, 3), padding=(1, 1), activation="relu", frame_size=(160, 240), num_layers=3, device=device)
    model_checkpoint = torch.load(args.model-dir, map_location=device)
    model.load_state_dict(model_checkpoint)
    model.to(device)

    masker = Unet_masker = UNet(                 # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        n_class=49,                      # model output channels (number of classes in your dataset)
    )
    unet_checkpoint = torch.load(args.masker-dir, map_location=device)
    masker.load_state_dict(unet_checkpoint)
    masker.to(device)

    model_optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0.005)
    masker_optimizer = AdamW(masker.parameters(), lr=1e-4, weight_decay=0.005)

    total_steps = num_epochs * len(dataloader_train)

    model_scheduler = CosineAnnealingLR(model_optimizer, T_max=total_steps, eta_min=1e-8)
    masker_scheduler = CosineAnnealingLR(masker_optimizer, T_max=total_steps, eta_min=1e-8)

    class_weights = torch.ones(49)
    class_weights[0] = 0.5
    class_weights = class_weights.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    early_stop = EarlyStop(patience, loss=True)

    if args.resume:
        print("Attempting to find existing checkpoint")
        path_partials = os.path.join(save_dir, "finetune/partial")
        try:
            checkpoint = torch.load(os.path.join(path_partials, "checkpoint.pkl"), map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            masker.load_state_dict(checkpoint['masker_state_dict'])
            model_optimizer.load_state_dict(checkpoint['model_optimizer_state_dict'])
            masker_optimizer.load_state_dict(checkpoint['masker_optimizer_state_dict'])
            if model_scheduler is not None:
                model_scheduler.load_state_dict(checkpoint['model_scheduler_state_dict'])
            if masker_scheduler is not None:
                masker_scheduler.load_state_dict(checkpoint['masker_scheduler_state_dict'])
            epoch = checkpoint['epoch']
            early_stop = checkpoint['early_stop']
            print(f'Resuming finetuning from epoch {epoch}')
        except :
            print("Couldn't load model from checkpoint, starting again finetuning from epoch 0")
    
    # Pass model to DataParallel (Could improve by using DistributedDataParallel)
    if torch.cuda.device_count() > 1:
        print("Model training will be distributed to {} GPUs.".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)
    model.to(device)

    print('Start finetuning model...')
    results = finetune(epoch, model, masker, dataloader_train, dataloader_val, criterion, model_optimizer, masker_optimizer, model_scheduler, masker_scheduler, early_stop, num_epochs, device, save_dir)

    print(f'Model finetuning finshed at epoch {results["epochs"]}, trainig loss: {results["train_loss"]}')
