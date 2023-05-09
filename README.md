# Deep Learning Competition Spring 2023
## Group 25: Ticino Tech Titans


### How to train the ConvLSTM to reproduce our results

Our overall model contains 2 major components, and 3 steps to do the training. We use a standard ConvLSTM, but instead of processing the raw input image, which leads to blurry and difficult-to-mask output over 11 future time steps, we first mask the input image using a vanilla unet with custom loss, and then predict the future time steps on the mask directly. This has the advantage of us being able to 'sharpen' the output at each time step, and leading to a significantly higher jaccard on the validation set than training on the input images directly.

So the full pipeline is `context images -> generate mask with UNet masker -> recursively predict next 11 with convLSTM`

And the training steps are:
1. Train the masker with the training set
2. Pretrain the convLSTM on the generated masks for the unlabeled set
3. (Optional) Finetune the complete network on the training set

# Custom dataset loader VideoFrameDataset
In order to change the data into the format required for loading with our custom dataloader (adapted from https://github.com/RaivoKoot/Video-Dataset-Loading-Pytorch), you need to generate an annotations file. To do this you need to:
1. Unzip the Dataset_Student.zip; we let `path_to_dataset` indicate the path to the unzipped `Dataset_Student` folder.
2. Inside each `val`, `unlabeled`, and `train`, create a `data` directory and move all the video frames into it. To do this follow the instrctions below.
3. Then, for each of the three sets, run the `annotations_generator.py` script, with the first two arguments being the start and final ids of the videos in the dataset, the directory of the dataset, and the number of frames.
4. For the `hidden` set, the process is the same, but the files have only 11 frames.
5. Each dataset directory should now have a `data` folder with the videos and an `annotations.txt`, which allows the VideoFrameDataset to load them.

Instructions for dataset preparation:
Train set 
- Go to `path_to_dataset/train`
- Execute the following commands to move all the video folders inside a folder named `data`:
    - `mkdir data`
    - `mv video_* data`
- Go to location of the `annotations_generator.py` script
- Execute the following command to generate the annotations file for the `train` dataset:
    - `python annotations_generator.py 0 1000 path_to_dataset/train 22`

Val set
- Go to `path_to_dataset/val`
- Execute the following commands to move all the video folders inside a folder named `data`:
    - `mkdir data`
    - `mv video_* data`
- Go to location of the `annotations_generator.py` script
- Execute the following command to generate the annotations file for the `val` dataset:
    - `python annotations_generator.py 1000 2000 path_to_dataset/val 22`

Unlabeled set
- Go to `path_to_dataset/unlabeled`
- Execute the following commands to move all the video folders inside a folder named `data`:
    - `mkdir data`
    - `mv video_* data`
- Go to location of the `annotations_generator.py` script
- Execute the following command to generate the annotations file for the `unlabeled` dataset:
    - `python annotations_generator.py 2000 15000 path_to_dataset/unlabeled 22`

Hidden set
- We let `path_to_hidden_dataset` indicate the path to the unzipped hidden dataset for leaderboard; go to `path_to_hidden_dataset`
- Execute the following commands to move all the video folders inside a folder named `data`:
    - `mkdir data`
    - `mv video_* data`
- Go to location of the `annotations_generator.py` script
- Execute the following command to generate the annotations file for the `hidden` dataset:
    - `python annotations_generator.py 15000 17000 path_to_hidden_dataset 11`

# Train the UNet masker network
Our UNet model is a vanilla 4-layer UNet model, the code for which we adapted from https://github.com/usuyama/pytorch-unet. For training, we use a MaskLoss, which is a custom compound loss composed of weighted cross entropy, dice loss, and focal loss, inspired by https://arxiv.org/abs/2210.05844. 
The model is trained to predict masks from frames for 20 epochs on the training dataset, after which it scores a jaccard of ~`0.9468` on the validation set.

For replicating this model, the `UNet_pretraining.py` script can be run on an HPC machine with the command 

`python UNet_pretrainnig.py --config-file-name configs/unet_config.json --output [/path/to/output-folder] --run-id [assign-a-run-id] (e.g. test_1) --resume [True/False, depending on if you are resuming a previous run]`. 

Make sure to change the data path in the config file to match with your current dataset path. Alternatively, there is an ipynb notebook at `UNet_masker.ipynb`, which contains straightforward commands to train and download the state dict for later use.

# Generate masks for the unlabeled set
We provide the `generate_masks_for_unlabeled.py` script, which will load the unlabeled dataset, generate predicted masks for each of them, and save them as tensors to the provided output directory.

Example usage: `python generate_masks_for_unlabeled.py --unlabeled [path/to/unlabeled/set] --model [path/to/unet/masker/model.pkl] --output-dir [path/to/output/dir]`

# Train ConvLSTM
Our ConvLSTM is based on the implementation by Rohit Panda at https://github.com/sladewinter/ConvLSTM, which has been extended to output 49 class logits instead of a binary classification for each pixel. Additionally, we use a weighted cross entropy loss which deemphasizes the importance of the background class 0.
To pretrain the ConvLSTM on the generated masks for the unlabeled dataset, we provide the `ConvLSTM_train.py` script. Change the data path file in `configs/lstm_config.json` with your current path for the unlabeled masked data generated in the previous step (`[path/to/masks/for/unlabeled]`) and the validation data (`[path/to/validation/set]`). 

Usage: `python ConvLSTM_train.py --config-file-name configs/lstm_config.json --output-dir [where/to/put/checkpoints] --run-id [assign-a-run-id] --resume [True/False, depending on if you are resuming a previous run]`

# (Optional) Finetune ConvLSTM
We perform the finetuning on the provided training set of the ConvLSTM for the specific prediction task. We finetuned the ConvLSTM pre-trained as described above using the labeled training set, containing true mask labels for each of the 22 video frames. The finetuning process works as follows
- For each batch, we first use the first 11 frames to predict the following one (12th), compute the loss and update the parameters with the optimizer step.
- Then we recursively predict the following frames building on the previous ones and update the parameters at every step. This means that to predict the 22nd frame will use the previous 10 predicted and the 11th given.
In this way the model should be finetuned for our specific prediction task.

To finetune the ConvLSTM on the training masks, we provide the `ConvLSTM_finetune.py` script. Change the data path file in `configs/finetune_config.json` with your current path for the training data (`[path/to/train/set]`) and the validation data (`[path/to/validation/set]`). 

Usage: `python ConvLSTM_finetune.py --config-file-name configs/finetune_config.json --output-dir [where/to/put/checkpoints] --run-id [assign-a-run-id] --resume [True/False, depending on if you are resuming a previous run]`

# Generate predicted masks for hidden
We used a jupyter notebook in order to generate the final predictions for the hidden set, which can be found at `Generate_predictions_for_hidden.ipynb`.
To use the notebook, make sure to have the UNet and ConvLSTM trained models as well as the python scripts with the classes used (models, dataset, utils). 

