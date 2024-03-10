import torch
import yaml

def make_padding_mask(lengths):
    """
    パディング部のマスクを生成する関数
    """
    max_len = torch.max(lengths).item()
    range_tensor = torch.arange(0, max_len, device=lengths.device, dtype=lengths.dtype)
    mask = range_tensor >= lengths.unsqueeze(1)
    return mask

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config