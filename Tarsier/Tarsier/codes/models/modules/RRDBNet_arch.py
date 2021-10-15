import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import models.modules.module_util as mutil
from math import sqrt
from collections import OrderedDict
from itertools import islice
import operator
from sys import platform

class GaussianNoise(nn.Module):
    """Gaussian noise regularizer.

    Args:
        sigma (float, optional): relative standard deviation used to generate the
            noise. Relative means that it will be multiplied by the magnitude of
            the value your are adding the noise to. This means that sigma can be
            the same regardless of the scale of the vector.
        is_relative_detach (bool, optional): whether to detach the variable before
            computing the scale of the noise. If `False` then the scale of the noise
            won't be seen as a constant but something to optimize: this will bias the
            network to generate vectors with smaller values.
    """

    def __init__(self, sigma=0.1, zeseed='toto', is_relative_detach=False, toto_noise_weights=None):  #### TOUTOU
        super().__init__()
        self.sigma = sigma
        self.pouet = 0
        if zeseed == "toto":
            def read_integers(filename):
                with open(filename) as f:
                    res = []
                    for l in f:
                        for e in l.split(" "):
                            res += [float(e)]
                    return res

            # if toto_noise_weights is not None:
            #     print("#toto: getting the toto weights given as input")
            #     self.toto = toto_noise_weights
            # else:q
            #     print("#toto: Reading toto weights from file toto")
            #     self.toto = read_integers("toto")
        self.zeseed = zeseed
        self.is_relative_detach = is_relative_detach
        device = 'cpu' if platform == 'darwin' else 'cuda'
        self.noise = torch.tensor(0, dtype=torch.float).to(torch.device(device))

    def forward(self, x, toto):
        self.pouet += 1
        if (self.training or self.zeseed) and self.sigma != 0:
            scale = self.sigma * x.detach() if self.is_relative_detach else self.sigma * x
            if self.zeseed:
                if self.zeseed == "toto":
                    sampled_noise = self.noise.repeat(*x.size())
                    sampled_noise_size = list(sampled_noise.size())
                    #print(f"sampled noise size: {sampled_noise_size}")
                    num_sampled_noise = len(toto)
                    # num_sampled_noise = 1
                    # for e in sampled_noise_size:
                    #    num_sampled_noise *= e
                    # print("num_sampled_noise", num_sampled_noise)
                    rsquare_dim = int(sqrt(num_sampled_noise))
                    data = toto.reshape((1, 1, rsquare_dim, rsquare_dim))
                    sampled_noise = torch.nn.Upsample(size=sampled_noise.size()[2:], mode='bilinear')(data)
                    # sampled_noise = torch.tensor(data, dtype=torch.float).to(torch.device('cuda')).reshape(sampled_noise.size())
                    # print("#toto: {} weights ".format(len(toto)))
                    sampled_noise = sampled_noise * scale
                else:
                    sampled_noise = self.noise.repeat(*x.size()).normal_(generator=torch.cuda.manual_seed(self.zeseed +
                                                                                                          self.pouet)) * scale
            else:
                sampled_noise = self.noise.repeat(*x.size()).normal_() * scale
            x = x + sampled_noise
        return x


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, nf=64, gc=32, bias=True, toto_noise_weights=None):
        super(ResidualDenseBlock_5C, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1x1 = conv1x1(nf, gc)
        self.noise = GaussianNoise(toto_noise_weights=toto_noise_weights)
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # initialization
        mutil.initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x, toto):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x2 = x2 + self.conv1x1(x)
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x4 = x4 + x2
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return self.noise(x5 * 0.2 + x, toto)


class RRDB(nn.Module):
    '''Residual in Residual Dense Block'''

    def __init__(self, nf, gc=32, toto_noise_weights=None):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(nf, gc)
        self.RDB2 = ResidualDenseBlock_5C(nf, gc)
        self.RDB3 = ResidualDenseBlock_5C(nf, gc)

    def forward(self, x, toto):
        weights_per_block = len(toto) // 3

        out = self.RDB1(x, toto[:weights_per_block].float().cuda())
        out = self.RDB2(out, toto[weights_per_block:weights_per_block * 2].float().cuda())
        out = self.RDB3(out, toto[-weights_per_block:].float().cuda())
        return out * 0.2 + x


class RRDBNet(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, gc=32, toto_noise_weights=None):
        print("#RDDBNet: initializing")
        super(RRDBNet, self).__init__()
        self.nb = nb
        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)

        self.trunk_layers = []
        for i in range(nb):
            self.trunk_layers.append(RRDB(nf=nf, gc=gc))
        self.RRDB_trunk = SequentialWithNoiseInput(*self.trunk_layers)
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        #### upsampling
        self.upconv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x, toto):
        fea = self.conv_first(x)

        # for i, layer in enumerate(self.trunk_layers):
        #     fea = layer(fea)
        # trunk = self.trunk_conv(fea)
        trunk = self.trunk_conv(self.RRDB_trunk(fea, toto))

        fea = fea + trunk

        fea = self.lrelu(self.upconv1(F.interpolate(fea, scale_factor=2, mode='nearest')))
        fea = self.lrelu(self.upconv2(F.interpolate(fea, scale_factor=2, mode='nearest')))
        out = self.conv_last(self.lrelu(self.HRconv(fea)))

        return out


class SequentialWithNoiseInput(nn.Module):
    r"""A sequential container.
    Modules will be added to it in the order they are passed in the constructor.
    Alternatively, an ordered dict of modules can also be passed in.

    To make it easier to understand, here is a small example::

        # Example of using Sequential
        model = nn.Sequential(
                  nn.Conv2d(1,20,5),
                  nn.ReLU(),
                  nn.Conv2d(20,64,5),
                  nn.ReLU()
                )

        # Example of using Sequential with OrderedDict
        model = nn.Sequential(OrderedDict([
                  ('conv1', nn.Conv2d(1,20,5)),
                  ('relu1', nn.ReLU()),
                  ('conv2', nn.Conv2d(20,64,5)),
                  ('relu2', nn.ReLU())
                ]))
    """

    def __init__(self, *args):
        super(SequentialWithNoiseInput, self).__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)

    def _get_item_by_idx(self, iterator, idx):
        """Get the idx-th item of the iterator"""
        size = len(self)
        idx = operator.index(idx)
        if not -size <= idx < size:
            raise IndexError('index {} is out of range'.format(idx))
        idx %= size
        return next(islice(iterator, idx, None))

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return self.__class__(OrderedDict(list(self._modules.items())[idx]))
        else:
            return self._get_item_by_idx(self._modules.values(), idx)

    def __setitem__(self, idx, module):
        key = self._get_item_by_idx(self._modules.keys(), idx)
        return setattr(self, key, module)

    def __delitem__(self, idx):
        if isinstance(idx, slice):
            for key in list(self._modules.keys())[idx]:
                delattr(self, key)
        else:
            key = self._get_item_by_idx(self._modules.keys(), idx)
            delattr(self, key)

    def __len__(self):
        return len(self._modules)

    def __dir__(self):
        keys = super(SequentialWithNoiseInput, self).__dir__()
        keys = [key for key in keys if not key.isdigit()]
        return keys

    def forward(self, input, toto):
        weights_per_block = len(toto) // len(self._modules.values())
        for i, module in enumerate(self._modules.values()):
            input = module(input, toto[i * weights_per_block: (i + 1) * weights_per_block])
        return input
