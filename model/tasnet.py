from dataclasses import dataclass

import torch
import torch.nn as nn

from utils import Encoder, Tcn, TcnParams, Decoder


@dataclass
class TasNetParams:
    window_size: int
    window_shift: int
    kernel_size: int
    in_channel: int
    bn_channel: int
    hidden_channel: int
    tcn_block: int = 3
    tcn_layer: int = 8
    num_spk: int = 2
    causal: bool = False
    skip: bool = True


class TasNet(nn.Module):
    def __init__(self, params):
        super(TasNet, self).__init__()
        self.params = params
        self.encoder = Encoder(params.window_size, params.window_shift,
                               params.in_channel)
        tcn_params = TcnParams(
            **{
                'kernel_size': params.kernel_size,
                'in_channel': params.in_channel,
                'out_channel': params.in_channel * params.num_spk,
                'bn_channel': params.bn_channel,
                'hidden_channel': params.hidden_channel,
                'block': params.tcn_block,
                'layers': params.tcn_layer,
                'causal': params.causal,
                'skip': params.skip
            })
        self.tcn = Tcn(tcn_params)
        self.decoder = Decoder(params.window_size, params.window_shift,
                               params.in_channel)

    def forward(self, input):
        en_output = self.encoder(input)
        mask_output = self.tcn(en_output)
        mask_output = mask_output.view(mask_output.shape[0],
                                       self.params.num_spk,
                                       self.params.in_channel,
                                       mask_output.shape[2])
        output = self.decoder(en_output, mask_output)
        output = output.view(mask_output.shape[0], self.params.num_spk, -1)
        return output

    def save(self, model_path):
        torch.save(self.state_dict(), model_path)

    def load(self, model_path):
        self.load_state_dict(torch.load(model_path))
