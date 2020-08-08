import torch
from tasnet import TasNet, TasNetParams

tasnet_params = TasNetParams(
    **{
        'window_size': 32,
        'window_shift': 16,
        'kernel_size': 3,
        'in_channel': 512,
        'bn_channel': 128,
        'hidden_channel': 128 * 4,
        'tcn_block': 3,
        'tcn_layer': 8,
        'num_spk': 2,
        'causal': False,
        'skip': True
    })

tasnet_params_causal = TasNetParams(
    **{
        'window_size': 32,
        'window_shift': 16,
        'kernel_size': 3,
        'in_channel': 512,
        'bn_channel': 128,
        'hidden_channel': 128 * 4,
        'tcn_block': 3,
        'tcn_layer': 8,
        'num_spk': 2,
        'causal': False,
        'skip': True
    })


def test_tasnet():
    x = torch.randn([16000])
    x = x.view([1, 16000])
    tasnet = TasNet(tasnet_params)
    output = tasnet(x, )
    print(output.shape)


def test_tasnet_causal():
    x = torch.randn([16000])
    x = x.view([1, 16000])
    tasnet = TasNet(tasnet_params_causal)
    output = tasnet(x)
    print(output.shape)


if __name__ == '__main__':
    test_tasnet()
    test_tasnet_causal()
