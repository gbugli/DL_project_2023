import torch
import torchmetrics
# evaluate the model
# generate metrics

# prepare the validation dataset
# take first 11 frames as input
# compute the loss of encoder output for 11-22 frames with the target
# pass all frames to decoder
# compute the loss of decoder output of each frame with the mask
# compute the jaccard index of the mask and the decoder output for all frames ### or should we compute separate scores for each time step directly?
# compute the jaccard index of the mask and the decoder output for the last frame

# the final metric is what we want to optimize, since we get graded on it, but the other ones will provide good insights

# this method works for any matching number of frames
def compute_jaccard(ground_truth_mask, predicted_mask, device):
  jaccard = torchmetrics.JaccardIndex(task="multiclass", num_classes=49).to(device)
  jaccard(torch.Tensor(ground_truth_mask), torch.Tensor(predicted_mask))

def evaluate_model(encoder, decoder, validationloader):
  encoder.eval()
  decoder.eval()
  jaccard_scores = []
  with torch.no_grad():
    for data in validationloader:
      inputs, target_masks = data
      #images, labels = images.to(device), labels.to(device)

      ### take only first 11 frames of each video as input
      # inputs should be b t c h w
      inputs = inputs[:, :11, :, :, :] # does this work?

      ### compute predictions
      predicted_embeddings = encoder(inputs.transpose(1, 2))
      predicted_masks = decoder(predicted_embeddings)

      ## want to go from ( (b t) h w num_classes) to ((b t) h w) where each pixel is a class number
      _, predicted = torch.max(predicted_masks.data, 2) # 2 should be the dimension of the num_classes

      jaccard_scores.append(compute_jaccard(predicted, target_masks))
  return jaccard_scores