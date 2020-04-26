import torch
import torch.nn as nn
from torch.optim import Adam
import time
from copy import deepcopy
from utils import init_params, MLP


def clones(module, N):
    "Produce N identical layers."

    return nn.ModuleList([deepcopy(module) for _ in range(N)])


class Core(nn.Module):
    def __init__(self, input_dims, hid_dim=32, kernel_size=(3, 3), bn_kwargs={}):
        super(Core, self).__init__()

        # preparation
        C, H, W = input_dims
        assert H == W
        fc_dim = C * H * W  # flatten dimensions

        # padding to retain the layer size
        padding = [int((ks - 1) / 2) for ks in kernel_size]

        self.flatten = nn.Flatten()
        # value network
        self.value_net = MLP([fc_dim, hid_dim, H],
                             batch_norm=True,
                             bn_kwargs=bn_kwargs)

        # internal abstraction
        self.conv_net = nn.Sequential(
            nn.Conv2d(hid_dim, hid_dim, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(hid_dim), nn.ReLU())

        # MRP model
        self.reward_net = MLP([fc_dim, hid_dim, H],
                              batch_norm=True,
                              bn_kwargs=bn_kwargs)

        # sigmoid to ensure the gammas and lambdas are in [-1， 1]
        self.gamma_net = MLP([fc_dim, hid_dim, H],
                             batch_norm=True,
                             activ_out=nn.Sigmoid,
                             bn_kwargs=bn_kwargs)
        self.lambda_net = MLP([fc_dim, hid_dim, H],
                              batch_norm=True,
                              activ_out=nn.Sigmoid,
                              bn_kwargs=bn_kwargs)

        # internal transition network
        self.state_net = nn.Sequential(
            nn.Conv2d(hid_dim,
                      hid_dim,
                      kernel_size=kernel_size,
                      padding=padding), nn.BatchNorm2d(hid_dim, **bn_kwargs),
            nn.ReLU(),
            nn.Conv2d(hid_dim,
                      hid_dim,
                      kernel_size=kernel_size,
                      padding=padding), nn.BatchNorm2d(hid_dim, **bn_kwargs),
            nn.ReLU())

    def forward(self, obs):
        # calculate value for the current embeded state
        val = self.value_net(self.flatten(obs))

        # generate up-level state representation by one more convolutions
        obs = self.conv_net(obs)
        obs_flatten = self.flatten(obs)

        # MRP model
        rwd = self.reward_net(obs_flatten)
        gam = self.gamma_net(obs_flatten)
        lam = self.lambda_net(obs_flatten)

        # internal transition
        obs = self.state_net(obs)
        return obs, val, rwd, gam, lam


class Predictron(nn.Module):
    def __init__(self,
                 input_dims,
                 hid_dim=32,
                 n_conv_blocks=2,
                 kernel_size=(3, 3),
                 core_depth=16,
                 lr=1e-3,
                 weight_decay=1e-4,
                 bn_momentum=3e-4,
                 max_grad_norm=10.,
                 device='cpu'
                 ):
        super().__init__()
        C, H, W = input_dims
        assert H == W
        self.core_depth = core_depth
        self.max_grad_norm = max_grad_norm
        self.device = torch.device(device=device)
        self.buffer = Buffer(depth=core_depth, rdim=H)

        # convolutional padding to retain the layer size
        padding = [int((ks - 1) / 2) for ks in kernel_size]

        # a two-layer convolutional network as the state representation
        self.embed = nn.Sequential(
            nn.Conv2d(C, hid_dim, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(hid_dim, momentum=bn_momentum), nn.ReLU(),
            nn.Conv2d(hid_dim, hid_dim, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(hid_dim, momentum=bn_momentum), nn.ReLU()).to(self.device)

        self.cores = clones(
            Core(input_dims=(hid_dim, H, W),
                 hid_dim=hid_dim,
                 kernel_size=kernel_size), core_depth).to(self.device)

        self.flatten = nn.Flatten()
        self.value_net_f = deepcopy(self.cores[0].value_net).to(self.device)
        self.loss_fn = nn.MSELoss(reduction='mean').to(self.device)

        self.optim = Adam(self.parameters(), lr=lr, weight_decay=weight_decay)

        self.apply(init_params)

    def forward(self, x_in, y_in):
        x = torch.as_tensor(x_in, dtype=torch.float32, device=self.device)
        y = torch.as_tensor(y_in, dtype=torch.float32, device=self.device)
        x = self.embed(x)

        self.buffer.reset()
        for core in self.cores:
            x, val, rwd, gam, lam = core(x)
            self.buffer.store(rwd, gam, lam, val)

        val = self.value_net_f(self.flatten(x))
        self.buffer.finish_path(last_val=val)
        pret, g_lam_ret = self.buffer.get()

        y_tile = y.unsqueeze(1).expand_as(pret)
        ploss = self.loss_fn(pret, y_tile)
        lloss = self.loss_fn(g_lam_ret, y)
        loss = ploss + lloss

        self.optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.max_grad_norm)
        self.optim.step()

        return (x.detach().cpu().item() for x in [loss, ploss, lloss])


class Buffer(object):
    def __init__(self, depth, rdim):
        # buffers for each epoch
        self.depth = depth
        self.rdim = rdim
        self.reset()

    def store(self, rew, gam, lam, val):
        """
        Record the internal model steps
        """
        self.rew_buf.append(rew)
        self.lam_buf.append(lam)
        self.gam_buf.append(gam)
        self.val_buf.append(val)

    def reset(self):
        self.rew_buf, self.gam_buf, self.lam_buf, self.val_buf = [], [], [], []

    def finish_path(self, last_val):
        self.val_buf.append(last_val)
        self.rew_buf = [torch.zeros_like(self.rew_buf[0], dtype=torch.float32)] + self.rew_buf
        self.gam_buf = [torch.zeros_like(self.gam_buf[0], dtype=torch.float32)] + self.gam_buf

    def get(self):
        rew = torch.cat(self.rew_buf, dim=1).reshape(-1, self.depth + 1, self.rdim)
        val = torch.cat(self.val_buf, dim=1).reshape(-1, self.depth + 1, self.rdim)
        lam = torch.cat(self.lam_buf, dim=1).reshape(-1, self.depth, self.rdim)
        gam = torch.cat(self.gam_buf, dim=1).reshape(-1, self.depth + 1, self.rdim)

        # calculate preturn
        pret = []
        for i in range(self.depth, -1, -1):
            g_i = val[:, i, :]
            for j in range(i, 0, -1):
                g_i = rew[:, j, :] + gam[:, j, :] * g_i
            pret.append(g_i)
        pret = torch.cat(pret[::-1], dim=1).reshape(-1, self.depth + 1, self.rdim)

        # calculate lambda-preturns
        g_lam_ret = val[:, -1, :]
        for i in range(self.depth - 1, -1, -1):
            g_lam_ret = (1 - lam[:, i, :]) * val[:, i, :] + lam[:, i, :] * (
                rew[:, i + 1, :] + gam[:, i + 1, :] * g_lam_ret)

        return pret, g_lam_ret


def main(env, args):
    model = Predictron((1, args.maze_size, args.maze_size), core_depth=args.core_depth)
    t_start = time.time()
    for i in range(args.max_steps):
        mx, my = env.generate_labelled_mazes(args.batch_size)
        loss, lossp, lossl = model(mx, my)
        if i % 100 == 0:
            print(f'Ep: {i}\t | T: {time.time() - t_start:6.0f} |' +
                  f'L: {loss:.4f} | Lp: {lossp:.4f} | Ll: {lossl:.4f}|')


if __name__ == '__main__':
    from maze import MazeEnv
    from utils import get_configs
    args = get_configs()
    env = MazeEnv(args.maze_size, args.maze_size, args.maze_density)
    main(env, args)
