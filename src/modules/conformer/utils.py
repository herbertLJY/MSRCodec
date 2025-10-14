


def crop_seq(x, offsets, length):
    """Crop padded tensor with specified length.

    :param x: (torch.Tensor) The shape is (B, C, D)
    :param offsets: (list)
    :param min_len: (int)
    :return:
    """
    B, C, T = x.shape
    x_ = x.new_zeros(B, C, length)
    for i in range(B):
        x_[i, :] = x[i, :, offsets[i]: offsets[i] + length]
    return x_
