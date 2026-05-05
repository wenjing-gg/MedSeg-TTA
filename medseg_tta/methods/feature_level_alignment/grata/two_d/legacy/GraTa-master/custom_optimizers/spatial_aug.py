def augment(x, factor):
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
    elif factor == 5:
        return x
