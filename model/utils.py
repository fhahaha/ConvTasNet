from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from normalization import cLN, gLN


class Encoder(nn.Module):
    def __init__(self, win_size, win_shift, out_channels):
        super(Encoder, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=1,
                                out_channels=out_channels,
                                kernel_size=win_size,
                                stride=win_shift,
                                bias=False)

    def forward(self, input):  # [B, S]
        input = torch.unsqueeze(input, 1)  # [B, 1, S]
        input = self.conv1d(input)  # [B, N, T]  T = (S-win_size)/win_shift+1
        input = F.relu(input)
        return input


@dataclass
class SeperableConvParams:
    in_channel: int
    hidden_channel: int
    kernel_size: int
    dilation: int
    causal: bool
    skip: bool


_NORM_FN_ = {
    'cLN': cLN,
    'gLN': gLN,
}


class SeperableConv(nn.Module):
    def __init__(self, params):
        super(SeperableConv, self).__init__()
        self.params = params
        if params.causal:
            self.padding = (params.kernel_size - 1) * params.dilation
            norm = 'cLN'
        else:
            self.padding = ((params.kernel_size - 1) * params.dilation) // 2
            norm = 'gLN'
        # pre conv
        self.conv = nn.Conv1d(params.in_channel, params.hidden_channel, 1)
        self.prelu1 = nn.PReLU()
        self.norm1 = _NORM_FN_[norm](params.hidden_channel)
        # depthwise conv
        self.depth_conv = nn.Conv1d(params.hidden_channel,
                                    params.hidden_channel,
                                    params.kernel_size,
                                    dilation=params.dilation,
                                    groups=params.hidden_channel)
        self.prelu2 = nn.PReLU()
        self.norm2 = _NORM_FN_[norm](params.hidden_channel)
        # pointwise conv
        self.point_conv = nn.Conv1d(params.hidden_channel, params.in_channel,
                                    1)
        # skip connection
        if params.skip:
            self.skip_conv = nn.Conv1d(params.hidden_channel,
                                       params.in_channel, 1)

    def forward(self, input):
        input_bp = input
        input = self.conv(input)
        input = self.prelu1(input)
        input = self.norm1(input)
        if self.params.causal:
            input = F.pad(input, [self.padding])
        else:
            input = F.pad(input, [self.padding, self.padding])
        input = self.depth_conv(input)
        input = self.prelu2(input)
        input = self.norm2(input)
        output = self.point_conv(input)
        if self.params.skip:
            output_skip = self.skip_conv(input)
            return output + input_bp, output_skip
        return output


@dataclass
class TcnParams:
    kernel_size: int
    in_channel: int
    out_channel: int
    bn_channel: int
    hidden_channel: int
    block: int = 3
    layers: int = 8
    causal: bool = False
    skip: bool = True


class Tcn(nn.Module):
    def __init__(self, params):
        super(Tcn, self).__init__()
        self.params = params
        if params.causal:
            self.norm = gLN(params.in_channel)
        else:
            self.norm = cLN(params.in_channel)

        self.bn = nn.Conv1d(params.in_channel, params.bn_channel, 1)

        self.sconvs = nn.ModuleList()
        for b in range(params.block):
            for i in range(params.layers):
                ds_conv_params = SeperableConvParams(
                    **{
                        'in_channel': params.bn_channel,
                        'hidden_channel': params.hidden_channel,
                        'kernel_size': params.kernel_size,
                        'dilation': 2**i,  # [1,2,4,8...128]
                        'causal': params.causal,
                        'skip': params.skip,
                    })
                sconv = SeperableConv(ds_conv_params)
                self.sconvs.append(sconv)
        self.prelu = nn.PReLU()
        self.mask_conv = nn.Conv1d(params.bn_channel, params.out_channel, 1)

    def forward(self, input):  # input: [B, N, T]
        output = self.bn(self.norm(input))  # [B, D, T]
        if self.params.skip:
            skip_connection = torch.zeros_like(output)
            for i in range(len(self.sconvs)):
                output, skip = self.sconvs[i](output)
                skip_connection = skip_connection + skip
            output = skip_connection
        else:
            for i in range(len(self.modules)):
                output = self.modules[i](output)
        output = self.prelu(output)
        output = self.mask_conv(output)
        output = F.sigmoid(output)
        return output


class Decoder(nn.Module):
    def __init__(self, win_size, win_shift, in_channels):
        super(Decoder, self).__init__()
        self.dconv1d = nn.ConvTranspose1d(in_channels=in_channels,
                                          out_channels=1,
                                          kernel_size=win_size,
                                          bias=False,
                                          stride=win_shift)

    def forward(self, input, masks):  # input:[B, N, L] masks:[B, C, N, L]
        input = torch.unsqueeze(input, 1)  # [B, 1, N, L]
        output = input * masks  # [B, C, N, L]
        shape = output.shape
        output = output.view(shape[0] * shape[1], shape[2],
                             shape[3])  # [B*C, N, L]
        output = self.dconv1d(output)  # [B*C, 1, T]
        output = output.squeeze(1)  # [B*C, T]
        return output
