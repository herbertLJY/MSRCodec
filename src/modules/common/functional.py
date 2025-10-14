import torch


@torch.no_grad()
def sequence_mask(lens, max_len=None, dtype=torch.bool):
    """sequence_mask(torch.LongTensor([5,3]))
    tensor([[ True,  True,  True,  True,  True],
            [ True,  True,  True, False, False]])
    """
    if max_len is None:
        max_len = torch.max(lens)
    scale = torch.arange(0, max_len, device=lens.device)
    mask = torch.less(scale, lens.unsqueeze(-1)).to(dtype)
    return mask



@torch.no_grad()
def mask_inf(mask):
    mask = mask.long()
    mask = mask > 0
    mask = mask.long()

    infs = torch.FloatTensor([float("-inf"), 0]).unsqueeze(0).expand(mask.size(0), -1)
    infs = infs.to(mask.device)
    mask = torch.gather(infs, -1, mask) 
    return mask


if __name__ == '__main__':
    out = sequence_mask(torch.LongTensor([5,3]), 6)
    print(out)