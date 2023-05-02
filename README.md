# DL_project_2023


### How to train the ConvLSTM to reproduce our results

Our overall model contains 2 major components, and 3(?) steps to the training. We use a standard ConvLSTM, but instead of processing the raw input image, which leads to blurry and difficult-to-mask output over 11 future time steps, we first mask the input image first using a vanilla unet with custom loss, and then predict the future time steps on the mask directly. This has the advantage of us being able to 'sharpen' the output at each time step, and leading to a significantly higher jaccard on the validation set than training on the input images directly.

So the full pipeline is `context images -> generate mask with UNet masker -> predict next 11 with convLSTM`

And the training steps are:
1. train the masker with the training set
2. pretrain the convLSTM on the generated masks for the unlabeled set
3. finetune the complete network on the training set

# Custom dataset loader VideoFrameDataset
In order to change the data into the format required for loading with our custom dataloader (adapted from https://github.com/RaivoKoot/Video-Dataset-Loading-Pytorch), you need to generate an annotations file. To do this you need to:
1. unzip the Dataset_Student.zip
2. inside each `val`, `unlabeled`, and `train`, create a `data` directory and move all the video frames into it.
3. then, for each of the three sets, run the `annotations_generator.py` script, with the first two arguments being the start and final ids of the videos in the dataset, the directory of the dataset, and the number of frames.
4. For the `hidden` set, the process is the same, but the files have only 11 frames.
5. Each dataset directory should now have a `data` folder with the videos and an `annotations.txt`, which allows the VideoFrameDataset to load them.

Example usage: `python3 annotations_generator.py 0 1000 path/to/train/set 22`

# Train the UNet masker network
Our UNet model is a vanilla 4-layer UNet model, the code for which we adapted from https://github.com/usuyama/pytorch-unet. For training, we use a MaskLoss, which is a custom compound loss composed of weighted cross entropy, dice loss, and focal loss, inspired by https://arxiv.org/abs/2210.05844. 
The model is trained to predict masks from frames for 20 epochs on the training dataset, after which is scores a jaccard of ~`0.9468` on the validation set.

For replicating this model, the `UNet_pretraining.py` script can be run on an HPC machine with the command `x y z`. Alternatively, there is an ipynb notebook at `UNet_masker.ipynb`, which contains straightforward commands to train and download the state dict for later use.

# Generate masks for the unlabeled set
We provide the `generate_masks_for_unlabeled.py` script, which will load the unlabeled dataset, generate predicted masks for each of them, and save them as tensors to the provided output directory.

Example usage: `generate_masks_for_unlabeled.py --unlabeled [path/to/unlabeled/set] --model [path/to/unet/masker/model.pkl] --output-dir [path/to/output/dir]`

# Train ConvLSTM
#todo, giacomo knows best probably
Our ConvLSTM is based on the implementation by Rohit Panda at https://github.com/sladewinter/ConvLSTM, which has been extended to output 49 class logits instead of a binary classification for each pixel. Additionally, we use a weighted cross entropy loss which deemphasizes the importance of the background class 0.
To pretrain the ConvLSTM on the generated masks for the unlabeled dataset, we provide the `ConvLSTM_train.py` script.

Usage: `ConvLSTM_train.py --train-dir [path/to/masks/for/unlabeled] --val-dir [path/to/validation/set] --output-dir [where/to/put/checkpoints] --run-id [name] --resume [True/False, depending on if you are resuming a previous run]`

# Finetune ConvLSTM
#todo

# Generate predicted masks for hidden
We used a jupyter notebook in order to generate the final predictions for the hidden set, which can be found at `Notebooks/Generate_predictions_for_hidden.ipynb`.
To use the notebook, we upload the ConvLSTM files, the UNet files, along with the weights for the 

