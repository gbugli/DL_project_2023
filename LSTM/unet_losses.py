from torch.nn.modules.loss import CrossEntropyLoss

class CEandDiceLoss(nn.Module):
    def __init__(self, ce_class_weights, ce_weight=0.5, dice_weight=0.5):
        super(CEandDiceLoss, self).__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.ce_loss = nn.CrossEntropyLoss()
        self.dice_loss = smp.losses.DiceLoss(mode="multiclass")
    
    def forward(self, pred, target):
        ce_loss = self.ce_loss(pred, target)
        dice_loss = self.dice_loss(pred, target)
        combined_loss = self.ce_weight * ce_loss + self.dice_weight * dice_loss

        return combined_loss

class MaskLoss(nn.Module):
    def __init__(self, ce_class_weights, ce_weight=0.2, dice_weight=0.4, focal_weight=0.4):
        super(MaskLoss, self).__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.ce_loss = nn.CrossEntropyLoss(ce_class_weights)
        self.dice_loss = smp.losses.DiceLoss(mode="multiclass")
        self.focal_loss = smp.losses.FocalLoss(mode="multiclass")
    
    def forward(self, pred, target):
        ce_loss = self.ce_loss(pred, target)
        dice_loss = self.dice_loss(pred, target)
        focal_loss = self.focal_loss(pred, target)
        combined_loss = self.ce_weight * ce_loss + self.dice_weight * dice_loss + self.focal_weight * focal_loss

        return combined_loss