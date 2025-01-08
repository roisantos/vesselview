import torch.nn as nn
import torch


class DiceLoss(nn.Module):

    def forward(self, predict, target):
        assert predict.size() == target.size(), "the size of predict and target must be equal."
        self.epsilon = 1e-12

        pre = predict.flatten()
        tar = target.flatten()

        
        intersection = (pre * tar).sum(-1).sum()  # Multiplica el valor previsto por la etiqueta como intersección

        union = (pre + tar).sum(-1).sum()
        

        score = 1 - 2 * (intersection + self.epsilon) / (union + self.epsilon)
        
        return score
