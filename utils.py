import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn


def set_global_seeds(seed):
    if seed is None:
        return
    elif np.isscalar(seed):
        # set seeds
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    else:
        raise ValueError(f"Invalid seed: {seed} (type {type(seed)})")


def get_device(cpu_only=False, gpu=0, echo=False):
    if (not cpu_only) and torch.cuda.is_available():
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
        stdout = f'Using CUDA ({torch.cuda.get_device_name(torch.cuda.current_device())})'
        dev = 'cuda'
    else:
        stdout = 'Using CPU'
        dev = 'cpu'

    if echo:
        stdout = "=" * 10 + stdout + "=" * 10
        print(stdout)
    return dev


def init_params(model, gain=1.0):
    for params in model.parameters():
        if len(params.shape) > 1:
            torch.nn.init.xavier_uniform_(params.data, gain=gain)


def get_configs(echo=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=0, type=int)

    # Maze settings
    parser.add_argument('--maze_size', default=20, type=int)
    parser.add_argument('--maze_density', default=0.3, type=float)
    parser.add_argument('--core_depth', default=16, type=int)

    # Algo settings
    parser.add_argument('--max_steps', default=int(1e5), type=int)
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--max_grad_norm', default=10., type=float)
    parser.add_argument('--cpu_only', default=False, action='store_true')

    args, _ = parser.parse_known_args()
    args.device = get_device(cpu_only=args.cpu_only, echo=echo)

    return args


class MLP(nn.Module):
    def __init__(
            self,
            sizes,
            activ_fn=nn.ReLU,
            activ_out=nn.Identity,
            batch_norm=False,
            bn_kwargs={},
    ):
        super(MLP, self).__init__()
        nl = len(sizes) - 2
        assert nl >= 0, "at least two layers should be specified"

        layers = []
        for n, (n_in, n_out) in enumerate(zip(sizes[:-1], sizes[1:])):
            layers += [nn.Linear(n_in, n_out)]
            if n < nl:
                # intermediate layers
                activ = activ_fn
                if batch_norm:
                    layers += [nn.BatchNorm1d(n_out, **bn_kwargs)]
            else:
                # output layers
                activ = activ_out

            layers += [activ()]

        self.module = nn.Sequential(*layers)
        self._output_size = sizes[-1]

    def forward(self, inputs):
        return self.module(inputs)

    @property
    def output_size(self):
        return self._output_size
