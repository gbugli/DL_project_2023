import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

import torch.distributed as dist
import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DDP

from pathlib import Path
import random
from datetime import datetime
import matplotlib.pyplot as plt
from einops import rearrange
import torchmetrics

from torch.optim.lr_scheduler import CosineAnnealingLR

from model import VPTREnc, VPTRDec, VPTRDisc, init_weights, VPTRFormerNAR, VPTRFormerFAR
from model import GDL, MSELoss, L1Loss, GANLoss, BiPatchNCE
from utils import KTHDataset, BAIRDataset, MovingMNISTDataset
from utils import get_dataloader
from utils import visualize_batch_clips, save_ckpt, load_ckpt, set_seed, AverageMeters, init_loss_dict, write_summary, resume_training, resume_training_parallel
from utils import set_seed, gather_AverageMeters

import logging
import os

from video_dataset import VideoFrameDataset, ImglistToTensor

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
    
def see_examples(masker, encoder, decoder, transformer, valloader, device):
    val_loss = 0
    with torch.no_grad():
        last_frames_predicted_masks = []
        last_frames_actual_masks = []

        for idx, data in enumerate(valloader):
            frames, labels, masks = data
            frames, labels, masks = frames.to(device), labels.to(device), masks.to(device)

            encoded_frames = encoder(frames[:,:11])
            predicted_frames = transformer(encoded_frames)
            decoded_predicted_frames = decoder(predicted_frames)

            last_frames_actual_masks.append(masks[:,10].cpu())
            decoded_predicted_frames = rearrange(decoded_predicted_frames, 'b t c h w -> (b t) c h w')
            masks = rearrange(masks[:,11:,:,:], 'b t h w -> (b t) h w')
            predicted_masks = masker(decoded_predicted_frames)

            last_frames_predicted_masks.append(predicted_masks[[10,21]].cpu())
            if idx < 5:
            
                # Convert the tensors to numpy arrays for visualization
                outputs = torch.argmax(predicted_masks, dim=1)
                outputs = outputs.cpu().numpy()
                masks = masks.cpu().numpy()
                # frames = frames.cpu().numpy()
                decoded_predicted_frames.cpu().numpy()
                # Choose the first sample in the batch to visualize
                sample_id = 10

                # Plot the predicted frame and real frame
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
                # frames = rearrange(frames, 'b t c h w -> (b t) c h w')
                pred_frame = renorm_transform(decoded_predicted_frames)
                pred_frame = torch.clamp(pred_frame[sample_id], min=0., max=1.)
                frames = renorm_transform(frames[0])
                frames = torch.clamp(frames[sample_id + 11], min=0., max=1.)
                ax1.imshow(rearrange(pred_frame, 'c h w -> h w c').cpu())
                ax1.set_title("Predicted Frame")
                
                ax2.imshow(rearrange(frames.cpu(), 'c h w -> h w c'))
                ax2.set_title("Real Frame")
                plt.savefig(f'frames_{idx}')

                # Plot the predicted mask and real mask
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
                ax1.imshow(outputs[sample_id])
                ax1.set_title("Predicted Mask")
                ax2.imshow(masks[sample_id])
                ax2.set_title("Real Mask")
                plt.savefig(f'masks_{idx}')
            
        results_mask_tensor = torch.cat(last_frames_predicted_masks, dim=0)  # Concatenate tensors
        actual_mask_tensor = torch.cat(last_frames_actual_masks, dim=0)
        jaccard = compute_jaccard(results_mask_tensor, actual_mask_tensor)
        print(f"jaccard_score: {jaccard}")


def train_vptr_decoder_model(masker, encoder, decoder, transformer, trainloader, valloader, optimizer, criterion, device, epochs):
  # Training loop
  for epoch in range(epochs):
    # Training

    ## Set masker to train
    masker.train()
    encoder.eval()
    decoder.eval()
    transformer.eval()

    train_loss = 0
    # train_tqdm = tqdm(trainloader, desc=f"Epoch {epoch+1}/{num_epochs} (Train)")

    for idx, data in enumerate(trainloader):
        frames, labels, masks = data
        frames, labels, masks = frames.to(device), labels.to(device), masks.to(device)

        optimizer.zero_grad()

        encoded_frames = encoder(frames[:,:11])
        predicted_frames = transformer(encoded_frames)
        decoded_predicted_frames = decoder(predicted_frames)

        decoded_predicted_frames = rearrange(decoded_predicted_frames, 'b t c h w -> (b t) c h w')
        masks = rearrange(masks[:,11:,:,:], 'b t h w -> (b t) h w')
        predicted_masks = masker(decoded_predicted_frames)
        loss = criterion(predicted_masks, masks.long())

        loss.backward()
        optimizer.step()
        scheduler.step()

        train_loss += loss.item()
    print(f'train_loss: {train_loss / (len(trainloader) + 1)}, learning_rate: {optimizer.param_groups[0]["lr"]}')

      # Validation

      ## Set masker to eval
    masker.eval()

    val_loss = 0
    with torch.no_grad():
        last_frames_predicted_masks = []
        last_frames_actual_masks = []

        for idx, data in enumerate(valloader):
            frames, labels, masks = data
            frames, labels, masks = frames.to(device), labels.to(device), masks.to(device)

            encoded_frames = encoder(frames[:,:11])
            predicted_frames = transformer(encoded_frames)
            decoded_predicted_frames = decoder(predicted_frames)

            last_frames_actual_masks.append(masks[:,10].cpu())
            decoded_predicted_frames = rearrange(decoded_predicted_frames, 'b t c h w -> (b t) c h w')
            masks = rearrange(masks[:,11:,:,:], 'b t h w -> (b t) h w')
            predicted_masks = masker(decoded_predicted_frames)

            last_frames_predicted_masks.append(predicted_masks[[10,21]].cpu())

            loss = criterion(predicted_masks, masks.long())

            val_loss += loss.item()
        print(f'val_loss: {val_loss / (len(valloader) + 1)}')
    


if __name__ == '__main__':
    set_seed(3407)

    ckpt_save_dir = Path('/scratch/gb2572/DL_project_2023/output/test_FAR/models/partial')
    tensorboard_save_dir = Path('/scratch/gb2572/DL_project_2023/output/test_FAR/tensorboard')
    resume_AE_ckpt = Path('/scratch/gb2572/DL_project_2023/output/test_vptr/models/partial').joinpath('epoch_4.tar')

    resume_Transformer_ckpt = ckpt_save_dir.joinpath('epoch_40.tar')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_past_frames = 11
    num_future_frames = 11
    encH, encW, encC = 6, 6, 528
    TSLMA_flag = False
    rpe = True
    model_flag = 'FAR' #'NAR' for NAR model, 'FAR' for FAR model

    img_channels = 3 # 1 for KTH and MovingMNIST, 3 for BAIR
    N = 1
    loss_name_list = ['T_MSE', 'T_GDL', 'T_gan', 'T_total', 'Dtotal', 'Dfake', 'Dreal']

    batch_size = 1
    trainloader = load_data('/train/data/', '/train/annotations.txt', batch_size)
    valloader = load_data('/val/data/', '/val/annotations.txt', batch_size)

    Unet_masker = UNet(                 # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        n_class=49,                      # model output channels (number of classes in your dataset)
    )

    #Set the padding_type to be "zero" for BAIR dataset
    VPTR_Enc = VPTREnc(img_channels, feat_dim = encC, n_downsampling = 3).to(device)
    VPTR_Dec = VPTRDec(img_channels, feat_dim = encC, n_downsampling = 3, out_layer = 'Tanh').to(device)
    VPTR_Enc = VPTR_Enc.eval()
    VPTR_Dec = VPTR_Dec.eval()

    if model_flag == 'NAR':
        VPTR_Transformer = VPTRFormerNAR(num_past_frames, num_future_frames, encH=encH, encW = encW, d_model=encC, 
                                            nhead=8, num_encoder_layers=4, num_decoder_layers=8, dropout=0.1, 
                                            window_size=4, Spatial_FFN_hidden_ratio=4, TSLMA_flag = TSLMA_flag, rpe=rpe).to(device)
    else:
        VPTR_Transformer = VPTRFormerFAR(num_past_frames, num_future_frames, encH=encH, encW = encW, d_model=encC, 
                                        nhead=8, num_encoder_layers=6, dropout=0.1, 
                                        window_size=4, Spatial_FFN_hidden_ratio=4, rpe=rpe).to(device)

    VPTR_Transformer = VPTR_Transformer.eval()

    #load the trained autoencoder, we initialize the discriminator from scratch, for a balanced training
    loss_dict, start_epoch = resume_training({'VPTR_Enc': VPTR_Enc, 'VPTR_Dec': VPTR_Dec}, {}, resume_AE_ckpt, loss_name_list, map_location=device)
    if resume_Transformer_ckpt is not None:
        loss_dict, start_epoch = resume_training({'VPTR_Transformer': VPTR_Transformer}, 
                                                {}, resume_Transformer_ckpt, loss_name_list, map_location=device)
        
    _, _, _, renorm_transform = get_dataloader('BAIR', N, '/train', '/val', test_past_frames = 11, 
                                                     test_future_frames = 11, ngpus = 1, num_workers = 1)
    

    # model_dir = '/scratch/gb2572/DL_project_2023/output/unet/best_model_15_epochs.pkl'
    model_dir = None

    if model_dir is not None:
        unet_saved_data = torch.load(model_dir, map_location=device)
        Unet_masker.load_state_dict(unet_saved_data)
        Unet_masker.to(device)
    
    num_epochs = 15
    #criterion = smp.losses.JaccardLoss(mode="multiclass")
    weight = torch.ones(49)
    weight[0] = 0.3
    weight = weight.to(device)
    steps = num_epochs * len(trainloader)
    criterion = CEandDiceLoss(weight, 0.3, 0.7) #smp.losses.DiceLoss(mode="multiclass")#CrossEntropyLoss(weight=weight)
    optimizer = optim.Adam(Unet_masker.parameters(), lr=0.000001)
    scheduler = CosineAnnealingLR(optimizer, T_max=steps, eta_min=0.00000001)

    Unet_masker.to(device)
    criterion.to(device)
    # Unet_masker.eval()

    train_vptr_decoder_model(Unet_masker, VPTR_Enc, VPTR_Dec, VPTR_Transformer, trainloader, valloader, optimizer, criterion, device, num_epochs)
    see_examples(Unet_masker, VPTR_Enc, VPTR_Dec, VPTR_Transformer, valloader, device)
