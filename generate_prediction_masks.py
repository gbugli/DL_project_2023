import os
import gc
import torch
import torch.nn as nn
import torch.nn.utils as utils
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

### Use Encoder and Decoder to predict the hidden set

# From brightspace:
# There are 2000 videos with 11 frames each in the hidden dataset.
# You need to submit a saved pytorch tensor or numpy array with the size (2000, 160, 240) 
# that each (160, 240) matrix corresponding to the mask of 22nd frame of each video in the hidden set.

### new annotation method for hidden set:
### Annotate the hidden set
import sys

def annotate(min_i,max_i,output_dir):
    with open(output_dir + 'annotations.txt', 'w') as f:
        for i in range(min_i, max_i):
            f.write(f'video_{i} 0 10 0 \n') #should be 11 frames instead of 22


annotate(15000, 17000, 'hidden/') # I think the hidden videos have


## Custom Dataloader because shuffle needs to be false for hidden set
# and mask is false

### Dataloader method
# are these the correct settings? Cannot reuse previous load_data method unless we add mask and shuffle as passable params
def load_hidden_data(root, annotation_file, batch_size=2):
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
        frames_per_segment=11,
        imagefile_template='image_{:d}.png',
        transform=preprocess,
        mask=False,
        test_mode=False
    )

    dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
    return dataloader

#predicted mask for frame 22, in order
def predict_masks_for_hidden(encoder, decoder, hiddenloader, device):
    hidden_results = []

    encoder.eval()
    decoder.eval()

    for i, data in enumerate(hiddenloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        #print(f"input shape: {inputs.shape}")

        # forward pass through encoder to get the embeddings
        predicted_embeddings = encoder(inputs.transpose(1, 2))

        #print(f"predicted_embeddings before reshape: {predicted_embeddings.shape}")

        # Reshape predicted embeddings to (b t) (h w) m
        predicted_embeddings = rearrange(predicted_embeddings, 'b t n m -> (b t) n m')

        #print(f"predicted_embeddings after reshape: {predicted_embeddings.shape}")

        # forward pass through decoder to get the masks
        outputs = decoder(predicted_embeddings)

        #print(f"outputs: {outputs.shape}")

        # use argmax to go from logits to classes
        prediction = torch.argmax(outputs, 1)

        #print(torch.max(prediction))


        hidden_results.append(prediction[21]) # this only works when batch_size = 1

    results_hidden_tensor = torch.stack(hidden_results)
    return results_hidden_tensor


### Load the train and validation datasets
batch_size = 1
hiddenloader = load_hidden_data('hidden/data/', 'hidden/annotations.txt', batch_size)

results = predict_masks_for_hidden(encoder, decoder, hiddenloader) # this needs to be changed
torch.save(results, 'results_hidden_tensor.pt')