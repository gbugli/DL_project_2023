import torch
from torchvision import transforms
from dataset.video_dataset import VideoFrameDataset, ImglistToTensor, MaskDataset

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