import random
import torch.nn as nn

class RandomRotate(nn.Module):

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x):
        if random.random() < self.p:
            factor = random.randrange(0, 3)
            if factor == 0:
                return (x.flip(-1).transpose(-2, -1), factor)
            elif factor == 1:
                return (x.flip(-1).flip(-2), factor)
            elif factor == 2:
                return (x.transpose(-2, -1).flip(-1), factor)
        else:
            return (x, None)

    def inverse(self, pred, factor):
        if factor is not None:
            if factor == 0:
                return pred.transpose(-2, -1).flip(-1)
            elif factor == 1:
                return pred.flip(-1).flip(-2)
            elif factor == 2:
                return pred.flip(-1).transpose(-2, -1)
        else:
            return pred

class RandomFlip(nn.Module):

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x):
        if random.random() < self.p:
            factor = random.randrange(0, 2)
            if factor == 0:
                return (x.flip(-1), factor)
            elif factor == 1:
                return (x.flip(-2), factor)
        else:
            return (x, None)

    def inverse(self, pred, factor):
        if factor is not None:
            if factor == 0:
                return pred.flip(-1)
            elif factor == 1:
                return pred.flip(-2)
        else:
            return pred

class Rotate_and_Flip(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x, factor):
        if factor == 0:
            return x.flip(-1)
        elif factor == 1:
            return x.flip(-2)
        elif factor == 2:
            return x.flip(-1).transpose(-2, -1)
        elif factor == 3:
            return x.flip(-1).flip(-2)
        elif factor == 4:
            return x.transpose(-2, -1).flip(-1)

    def inverse(self, pred, factor):
        if factor == 0:
            return pred.flip(-1)
        elif factor == 1:
            return pred.flip(-2)
        elif factor == 2:
            return pred.transpose(-2, -1).flip(-1)
        elif factor == 3:
            return pred.flip(-1).flip(-2)
        elif factor == 4:
            return pred.flip(-1).transpose(-2, -1)
