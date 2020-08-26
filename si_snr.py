def si_snr(pred, label):
    label_power = torch.pow(label, 2).sum(1, keepdim=True)
    pred_true = torch.sum(pred * label, dim=-1,
                          keepdim=True) * label / label_power
    pred_noise = pred - pred_true
    pred_true_power = torch.pow(pred_true, 2).sum(1)
    pred_noise_power = torch.pow(pred_noise, 2).sum(1)
    si_snr = 10 * torch.log10(pred_true_power / pred_noise_power)
    return si_snr
