def normalize(val, val_mean, val_std):
    return (val - val_mean) / val_std

def unnormalize(val, val_mean, val_std):
    return val * val_std / val_mean

def from_numpy(np_arr):
    return torch.tensor(np_arr).to(device)

def to_numpy(arr):
    return arr.detach().cpu().numpy()