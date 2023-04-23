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

from models import Decoder
from models import IJEPA_base

from eval import compute_jaccard

def parse_args() -> Namespace:
    parser = ArgumentParser("Decoder")
    parser.add_argument("--train-dir", help="Name of dir with training data", required=True, type=str)
    parser.add_argument("--train-annotation-file", help="Name of dir with annotation files",
                        required=True, type=str)
    parser.add_argument("--val-dir", help="Name of dir with validation data", required=True, type=str)
    parser.add_argument("--val-annotation-file", help="Name of dir with annotation files",
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

def load_validation_data(val_folder, annotation_file_path):
    #same preprocessing as in training, since we are not doing any augmentation here
    preprocess = transforms.Compose([
            ImglistToTensor(),  # list of PIL images to (FRAMES x CHANNELS x HEIGHT x WIDTH) tensor
            transforms.Resize((128,128)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    dataset = VideoFrameDataset(
        root_path=val_folder,
        annotationfile_path=annotation_file_path,
        num_segments=1,
        frames_per_segment=22,
        imagefile_template='image_{:d}.png',
        transform=preprocess,
        test_mode=False
    )

    validationloader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=2,
            shuffle=True,
            num_workers=1,
            pin_memory=True
        )
    return validationloader


# Train the model
def train_model(epoch, decoder, encoder, criterion, optimizer, scheduler, dataloader, validationloader, num_epochs, output_dir, device):
    while epoch < num_epochs:
        decoder.train()
        # do we need to do encoder.eval() or something? Since we are not training it, we want to deactivate the dropouts
        train_loss = 0
        for i, data in enumerate(dataloader, 0):
            inputs, target_masks = data # does the dataloader load the masks yet?
            #inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            ### forward pass through encoder to get the embeddings
            predicted_embeddings = encoder(inputs.transpose(1, 2))
            print(predicted_embeddings.shape) # not sure the exact shape of this output

            ### forward pass through decoder to get the masks
            predicted_masks = decoder(predicted_embeddings)

            ### compute the loss and step
            loss = criterion(predicted_masks, target_masks)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()

            # Update the scheduler learning rate
            scheduler.step()
        
        avg_epoch_loss = train_loss / len(dataloader)

        # Validation loss
        decoder.eval()
        correct = 0
        total = 0
        jaccard_scores = []
        with torch.no_grad():
            for data in validationloader:
                inputs, target_masks = data
                #images, labels = images.to(device), labels.to(device)

                ### compute predictions
                predicted_embeddings = encoder(inputs.transpose(1, 2))
                predicted_masks = decoder(predicted_embeddings) # not sure if correct shape or need to rearrange first

                ## want to go from ( (b t) h w num_classes) to ((b t) h w) where each pixel is a class number
                _, predicted = torch.max(predicted_masks.data, 2) # 2 should be the dimension of the num_classes

                total += target_masks.size(0) * target_masks.size(1) * target_masks.size(2) # multiply by the height and width of the image
                correct += (predicted == target_masks).sum().item() # will this work in 3 dimensions?
                jaccard_scores.append(compute_jaccard(predicted, target_masks))
        
        # per-pixel accuracy on validation set
        # is this a good metric? Probably not
        val_acc = 100 * correct / total
        average_jaccard = sum(jaccard_scores) / len(jaccard_scores)

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch: {epoch + 1}, Learning Rate: {current_lr:.6f}, Average epoch train loss: {avg_epoch_loss:.4f}, Validation accuracy: {val_acc:.4f}, Average Jaccard score: {average_jaccard:.4f}")

        # Used this approach (while and epoch increase) so that we can get back to training the loaded model from checkpoint
        epoch += 1

        # Checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': decoder.module.state_dict() if torch.cuda.device_count() > 1 else decoder.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            #'early_stop': early_stop,
            }, os.path.join(output_dir, 'models/partial', "checkpoint.pkl"))

    return {
        "epochs": epoch,
        "train_loss": train_loss,
        "model": decoder
            }

if __name__ == "__main__":

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    args = parse_args()
    dataloader = load_data(args.train_folder, args.train_annotation_file)
    validationloader = load_validation_data(args.val_folder, args.val_annotation_file)

    num_epochs = 10
    total_steps = num_epochs * len(dataloader)

    # should these also come from global config?
    div_factor = 5 # max_lr/div_factor = initial lr
    final_div_factor = 10 # final lr is initial_lr/final_div_factor 

    # Used this approach so that we can getv back to training the loaded model from checkpoint
    epoch = 0

    # get these params from global config? to ensure that it always matches the trained IJEPA model
    encoder = IJEPA_base(img_size=128, patch_size=8, enc_depth=6, pred_depth=6, num_heads=8)
    decoder = Decoder(input_dim=768, hidden_dim=3072, num_hidden_layers=2)
    criterion = nn.CrossEntropyLoss() # since we will have label predictions?

    # Just using same optimizer and scheduler as IJEPA, will need to change later
    # probably higher lr than IJEPA
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=0.001, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.003, total_steps=total_steps, div_factor=div_factor, final_div_factor=final_div_factor)

    ### load pretrained IJEPA model -> should this just load from the latest IJEPA checkpoint?
    path_partials = os.path.join(args.output_dir, "models/partial")
    if os.path.exists(path_partials):
        checkpoint = torch.load(os.path.join(path_partials, "checkpoint.pkl"), map_location=device)
        encoder.load_state_dict(checkpoint['model_state_dict'])

    if args.resume:
        print("Attempting to find existing checkpoint")
        path_partials = os.path.join(args.output_dir, "models/partial")
        if os.path.exists(path_partials):
            checkpoint = torch.load(os.path.join(path_partials, "decoder_checkpoint.pkl"), map_location=device)
            decoder.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            epoch = checkpoint['epoch']

    results = train_model(epoch, decoder, encoder, criterion, optimizer, scheduler, dataloader, num_epochs, args.output_dir, device)
    # run full evaluation at this point?
    print(f'Decoder training finshed at epoch {results["epoch"]}, trainig loss: {results["train_loss"]}')
