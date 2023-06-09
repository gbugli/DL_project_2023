
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
from IJEPA.video_dataset import VideoFrameDataset, ImglistToTensor
from argparse import ArgumentParser, Namespace

from IJEPA.decoders import Decoder, ATMHead
from IJEPA.models import IJEPA_base, EarlyStop
from IJEPA.atm_loss import ATMLoss

from IJEPA.eval import compute_jaccard

def parse_args() -> Namespace:
    parser = ArgumentParser("Decoder")
    parser.add_argument("--train-dir", help="Name of dir with training data", required=True, type=str)
    parser.add_argument("--val-dir", help="Name of dir with validation data", required=True, type=str)
    parser.add_argument("--output-dir", required=True, type=str, help="Name of dir to save the checkpoints to")
    parser.add_argument("--run-id", help="Name of the run", required=True, type=str)
    parser.add_argument("--resume", default=False, type=bool, help="In case training was not completed resume from last epoch")

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
        mask=True,
        test_mode=False
    )

    dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,  # arbitrarily chosen
            pin_memory=True
        )
    return dataloader


# Train the model
def train_model(epoch, decoder, encoder, criterion, optimizer, scheduler, dataloader, validationloader, num_epochs, output_dir, device, early_stop):
    while epoch < num_epochs:
        decoder.train()
        # do we need to do encoder.eval() or something? Since we are not training it, we want to deactivate the dropouts
        train_loss = 0
        for i, data in enumerate(dataloader, 0):
            inputs, labels, target_masks = data 
            inputs, labels, target_masks = inputs.to(device), labels.to(device), target_masks.to(device)

            inputs = inputs[:, :11]

            optimizer.zero_grad()

            ### forward pass through encoder to get the embeddings
            predicted_embeddings = encoder(inputs.transpose(1, 2))

            # Reshape predicted embeddings to (b t) (h w) m
            predicted_embeddings = rearrange(predicted_embeddings, 'b t n m -> (b t) n m')
            target_masks = rearrange(target_masks, 'b t n m -> (b t) n m')

            ### forward pass through decoder to get the masks
            outputs = decoder(predicted_embeddings)

            # the target_mask tensor is of shape b f h w

            ### compute the loss and step
            loss = criterion(outputs, target_masks, -1)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()

            # Update the scheduler learning rate
            scheduler.step()

            if i % 50 == 0 and epoch < 5:
              print(f"Current loss: {loss.item()}")
        
        avg_epoch_loss = train_loss / len(dataloader)

        # # Validation loss
        # decoder.eval()
        # val_loss = 0
        # jaccard_scores = []
        # with torch.no_grad():
        #     for data in validationloader:
        #         inputs, labels, target_masks = data
        #         inputs, labels, target_masks = inputs.to(device), labels.to(device), target_masks.to(device)

        #         inputs = inputs[:, :11]

        #         ### compute predictions
        #         predicted_embeddings = encoder(inputs.transpose(1, 2))

        #         # Reshape predicted embeddings to (b t) (h w) m
        #         predicted_embeddings = rearrange(predicted_embeddings, 'b t n m -> (b t) n m')
        #         target_masks = rearrange(target_masks, 'b t n m -> (b t) n m')

        #         ### forward pass through decoder to get the masks
        #         outputs = decoder(predicted_embeddings)

        #         # compute loss
        #         val_loss +=  criterion(outputs, target_masks, -1)

        #         ## want to go from batch * frames x height x width x num_classes with logits to batch * frames x height x width with class predictions
        #         predicted  = torch.argmax(outputs['pred_masks'], 1)
        #         jaccard_scores.append(compute_jaccard(predicted, target_masks, device))
        
        # # per-pixel accuracy on validation set
        # # is this a good metric? Probably not
        # avg_val_loss = val_loss / len(validationloader)
        # average_jaccard = sum(jaccard_scores) / len(jaccard_scores)

        current_lr = optimizer.param_groups[0]['lr']
        # print(f"Epoch: {epoch + 1}, Learning Rate: {current_lr:.6f}, Avg train loss: {avg_epoch_loss:.4f}, Avg val loss: {avg_val_loss:.4f}, Avg Jaccard: {average_jaccard:.4f}")
        print(f"Epoch: {epoch + 1}, Learning Rate: {current_lr:.6f}, Avg train loss: {avg_epoch_loss:.4f}")

        # # Early Stopping
        # if average_jaccard > early_stop.best_value:
        #     torch.save(decoder.module.state_dict() if torch.cuda.device_count() > 1 else decoder.state_dict(), os.path.join(output_dir, 'models/decoder/best',"best_model.pkl"))

        # early_stop.step(average_jaccard, epoch)
        # if early_stop.stop_training(epoch):
        #     print(
        #         "early stopping at epoch {} since valdiation loss didn't improve from epoch no {}. Best value {}, current value {}".format(
        #             epoch, early_stop.best_epoch, early_stop.best_value, average_jaccard
        #         ))
        #     break

        # Used this approach (while and epoch increase) so that we can get back to training the loaded model from checkpoint
        epoch += 1

        # Checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': decoder.module.state_dict() if torch.cuda.device_count() > 1 else decoder.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'early_stop': early_stop,
            }, os.path.join(output_dir, 'models/decoder', "checkpoint_decoder.pkl"))

    return {
        "epochs": epoch,
        "train_loss": train_loss,
        "model": decoder
            }

if __name__ == "__main__":

    torch.cuda.empty_cache()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    args = parse_args()

    batch_size = 2

    # Make run dir
    if not os.path.exists(os.path.join(args.output_dir,args.run_id)):
        os.makedirs(os.path.join(args.output_dir,args.run_id), exist_ok=True)
    
    save_dir = os.path.join(args.output_dir,args.run_id)
    os.makedirs(os.path.join(save_dir, "models/decoder"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "models/decoder/best"), exist_ok=True)

    # Load train data and validation data
    train_data_dir  = os.path.join(args.train_dir, 'data')
    train_annotation_dir = os.path.join(args.train_dir, 'annotations.txt')
    val_data_dir  = os.path.join(args.val_dir, 'data')
    val_annotation_dir = os.path.join(args.val_dir, 'annotations.txt')

    print('Loading train data...')
    dataloader = load_data(train_data_dir, train_annotation_dir, batch_size)
    print('Loading val data...')
    validationloader = load_data(val_data_dir, val_annotation_dir, batch_size)

    num_epochs = 100
    total_steps = num_epochs * len(dataloader)

    # should these also come from global config?
    div_factor = 5 # max_lr/div_factor = initial lr
    final_div_factor = 10 # final lr is initial_lr/final_div_factor 
    patience = 10

    # Used this approach so that we can getv back to training the loaded model from checkpoint
    epoch = 0

    # get these params from global config? to ensure that it always matches the trained IJEPA model
    # load encoder
    encoder = IJEPA_base(img_size=128, patch_size=8, in_chans=3, norm_layer=nn.LayerNorm, num_frames=22, attention_type='divided_space_time', dropout=0.1, mode="test", M=4, embed_dim=384,
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

    # for k,v in encoder.state_dict().items():
    #     print(k)

    # load decoder       
    # decoder = Decoder(input_dim=768, hidden_dim=3072, num_hidden_layers=2)
    decoder = ATMHead(img_size=128, H=160, W=240, in_channels=384, use_stages=1)
    decoder.to(device)
    criterion = ATMLoss(48, 1)
    criterion.to(device)
    # criterion = nn.CrossEntropyLoss() # since we will have label predictions?

    # Just using same optimizer and scheduler as IJEPA, will need to change later
    # probably higher lr than IJEPA
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=0.001, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=0.000001)

    ### load pretrained IJEPA model -> should this just load from the latest IJEPA checkpoint?
    path_best = os.path.join(save_dir, "models/best")
    if os.path.exists(path_best):
        checkpoint = torch.load(os.path.join(path_best, "best_model.pkl"), map_location=device)
        encoder_state_dict = checkpoint # checkpoint['model_state_dict']
        # encoder_state_dict['mode'] = 'test'
        encoder.load_state_dict(encoder_state_dict)
    encoder.to(device)

    # path_partial = os.path.join(save_dir, "models/partial")
    # if os.path.exists(path_partial):
    #     checkpoint = torch.load(os.path.join(path_partial, "checkpoint.pkl"), map_location=device)
    #     encoder_state_dict = checkpoint['model_state_dict']
    #     # encoder_state_dict['mode'] = 'test'
    #     encoder.load_state_dict(encoder_state_dict)
    # encoder.to(device)


    early_stop = EarlyStop(patience)

    if args.resume:
        print("Attempting to find existing checkpoint")
        path_partials = os.path.join(save_dir, "models/decoder")
        if os.path.exists(path_partials):
            checkpoint = torch.load(os.path.join(path_partials, "checkpoint_decoder.pkl"), map_location=device)
            decoder.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            early_stop = checkpoint['early_stop']
            epoch = checkpoint['epoch']
    
    # if torch.cuda.device_count() > 1:
    #     print("Model training will be distributed to {} GPUs.".format(torch.cuda.device_count()))
    #     decoder = nn.DataParallel(decoder)
    # decoder.to(device)

    results = train_model(epoch, decoder, encoder, criterion, optimizer, scheduler, dataloader, validationloader, num_epochs, save_dir, device, early_stop)
    # run full evaluation at this point?
    print(f'Decoder training finshed at epoch {results["epochs"]}, trainig loss: {results["train_loss"]}')
