import torch
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from torch import nn


def get_nonlinear(config_str, channels):
    nonlinear = nn.Sequential()
    for name in config_str.split('-'):
        if name == 'relu':
            nonlinear.add_module('relu', nn.ReLU(inplace=True))
        elif name == 'prelu':
            nonlinear.add_module('prelu', nn.PReLU(channels))
        elif name == 'batchnorm':
            nonlinear.add_module('batchnorm', nn.BatchNorm1d(channels))
        elif name == 'batchnorm_':
            nonlinear.add_module('batchnorm',
                                 nn.BatchNorm1d(channels, affine=False))
        else:
            raise ValueError('Unexpected module ({}).'.format(name))
    return nonlinear


def statistics_pooling(x, dim=-1, keepdim=False, unbiased=True, eps=1e-2):
    mean = x.mean(dim=dim)
    std = x.std(dim=dim, unbiased=unbiased)
    stats = torch.cat([mean, std], dim=-1)
    if keepdim:
        stats = stats.unsqueeze(dim=dim)
    return stats


def high_order_statistics_pooling(x,
                                  dim=-1,
                                  keepdim=False,
                                  unbiased=True,
                                  eps=1e-2):
    mean = x.mean(dim=dim)
    std = x.std(dim=dim, unbiased=unbiased)
    norm = (x - mean.unsqueeze(dim=dim)) \
        / std.clamp(min=eps).unsqueeze(dim=dim)
    skewness = norm.pow(3).mean(dim=dim)
    kurtosis = norm.pow(4).mean(dim=dim)
    stats = torch.cat([mean, std, skewness, kurtosis], dim=-1)
    if keepdim:
        stats = stats.unsqueeze(dim=dim)
    return stats


class StatsPool(nn.Module):
    def forward(self, x):
        return statistics_pooling(x)


class HighOrderStatsPool(nn.Module):
    def forward(self, x):
        return high_order_statistics_pooling(x)


class TDNNLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 bias=False,
                 config_str='batchnorm-relu'):
        super(TDNNLayer, self).__init__()
        if padding < 0:
            assert kernel_size % 2 == 1, 'Expect equal paddings, but got even kernel size ({})'.format(
                kernel_size)
            padding = (kernel_size - 1) // 2 * dilation
        self.linear = nn.Conv1d(in_channels,
                                out_channels,
                                kernel_size,
                                stride=stride,
                                padding=padding,
                                dilation=dilation,
                                bias=bias)
        self.nonlinear = get_nonlinear(config_str, out_channels)

    def forward(self, x):
        x = self.linear(x)
        x = self.nonlinear(x)
        return x


class DenseTDNNLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 bn_channels,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 bias=False,
                 config_str='batchnorm-relu',
                 memory_efficient=False):
        super(DenseTDNNLayer, self).__init__()
        assert kernel_size % 2 == 1, 'Expect equal paddings, but got even kernel size ({})'.format(
            kernel_size)
        padding = (kernel_size - 1) // 2 * dilation
        self.memory_efficient = memory_efficient
        self.nonlinear1 = get_nonlinear(config_str, in_channels)
        self.linear1 = nn.Conv1d(in_channels, bn_channels, 1, bias=False)
        self.nonlinear2 = get_nonlinear(config_str, bn_channels)
        self.linear2 = nn.Conv1d(bn_channels,
                                 out_channels,
                                 kernel_size,
                                 stride=stride,
                                 padding=padding,
                                 dilation=dilation,
                                 bias=bias)

    def bn_function(self, x):
        return self.linear1(self.nonlinear1(x))

    def forward(self, x):
        if self.training and self.memory_efficient:
            x = cp.checkpoint(self.bn_function, x)
        else:
            x = self.bn_function(x)
        x = self.linear2(self.nonlinear2(x))
        return x


class DenseTDNNBlock(nn.ModuleList):
    def __init__(self,
                 num_layers,
                 in_channels,
                 out_channels,
                 bn_channels,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 bias=False,
                 config_str='batchnorm-relu',
                 memory_efficient=False):
        super(DenseTDNNBlock, self).__init__()
        for i in range(num_layers):
            layer = DenseTDNNLayer(in_channels=in_channels + i * out_channels,
                                   out_channels=out_channels,
                                   bn_channels=bn_channels,
                                   kernel_size=kernel_size,
                                   stride=stride,
                                   dilation=dilation,
                                   bias=bias,
                                   config_str=config_str,
                                   memory_efficient=memory_efficient)
            self.add_module('tdnnd%d' % (i + 1), layer)

    def forward(self, x):
        for layer in self:
            x = torch.cat([x, layer(x)], dim=1)
        return x


class StatsSelect(nn.Module):
    def __init__(self, channels, branches, null=False, reduction=1):
        super(StatsSelect, self).__init__()
        self.gather = HighOrderStatsPool()
        self.linear1 = nn.Conv1d(channels * 4, channels // reduction, 1)
        self.linear2 = nn.ModuleList()
        if null:
            branches += 1
        for _ in range(branches):
            self.linear2.append(nn.Conv1d(channels // reduction, channels, 1))
        self.channels = channels
        self.branches = branches
        self.null = null
        self.reduction = reduction

    def forward(self, x):
        f = torch.cat([_x.unsqueeze(dim=1) for _x in x], dim=1)
        x = torch.sum(f, dim=1)
        x = self.linear1(self.gather(x).unsqueeze(dim=-1))
        s = []
        for linear in self.linear2:
            s.append(linear(x).view(-1, 1, self.channels))
        s = torch.cat(s, dim=1)
        s = F.softmax(s, dim=1).unsqueeze(dim=-1)
        if self.null:
            s = s[:, :-1, :, :]
        return torch.sum(f * s, dim=1)

    def extra_repr(self):
        return 'channels={}, branches={}, reduction={}'.format(
            self.channels, self.branches, self.reduction)


class SqueezeExcitation(nn.Module):
    def __init__(self, channels, reduction=1):
        super(SqueezeExcitation, self).__init__()
        self.linear1 = nn.Conv1d(channels, channels // reduction, 1)
        self.relu = nn.ReLU(inplace=True)
        self.linear2 = nn.Conv1d(channels // reduction, channels, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        s = self.linear1(x.mean(-1, keepdim=True)+self.seg_pooling(x))
        s = self.relu(s)
        s = self.sigmoid(self.linear2(s))
        return x*s

    def seg_pooling(self, x, seg_len=100):
        s_x = F.max_pool1d(x, kernel_size=seg_len, stride=seg_len, ceil_mode=True)
        out = s_x.unsqueeze(-1).expand(-1, -1, -1, seg_len).reshape(*x.shape[:-1], -1)
        out = out[:, :, :x.shape[-1]]
        return out

class PoolingBlock(nn.Module):
    def __init__(self, bn_channels, out_channels, kernel_size, stride, padding, dilation, bias, reduction=2):
        super(PoolingBlock, self).__init__()
        self.linear_stem = nn.Conv1d(bn_channels,
                                 out_channels,
                                 kernel_size,
                                 stride=stride,
                                 padding=padding,
                                 dilation=dilation,
                                 bias=bias)
        self.linear1 = nn.Conv1d(bn_channels, bn_channels // reduction, 1)
        self.relu = nn.ReLU(inplace=True)
        # self.bn = nn.BatchNorm1d(out_channels)
        self.linear2 = nn.Conv1d(bn_channels // reduction, out_channels, 1)
        self.sigmoid = nn.Sigmoid()
        # self.linear3 = nn.Conv1d(out_channels, out_channels, 1)

    def forward(self, x):
        y = self.linear_stem(x)
        s = self.linear1(x.mean(-1, keepdim=True)+self.seg_pooling(x))
        s = self.relu(s)
        s = self.sigmoid(self.linear2(s))
        return y*s
    
    def seg_pooling(self, x, seg_len=100):
        s_x = F.max_pool1d(x, kernel_size=seg_len, stride=seg_len, ceil_mode=True)
        out = s_x.unsqueeze(-1).expand(-1, -1, -1, seg_len).reshape(*x.shape[:-1], -1)
        out = out[:, :, :x.shape[-1]]
        return out


class MultiBranchDenseTDNNLayer(DenseTDNNLayer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 bn_channels,
                 kernel_size,
                 stride=1,
                 dilation=(1, ),
                 bias=False,
                 null=False,
                 reduction=1,
                 config_str='batchnorm-relu',
                 memory_efficient=False):
        super(DenseTDNNLayer, self).__init__()
        assert kernel_size % 2 == 1, 'Expect equal paddings, but got even kernel size ({})'.format(
            kernel_size)
        padding = (kernel_size - 1) // 2
        if not isinstance(dilation, (tuple, list)):
            dilation = (dilation, )
        self.memory_efficient = memory_efficient
        self.nonlinear1 = get_nonlinear(config_str, in_channels)
        self.linear1 = nn.Conv1d(in_channels, bn_channels, 1, bias=False)
        self.nonlinear2 = get_nonlinear(config_str, bn_channels)
        self.linear2 = nn.ModuleList()
        for _dilation in dilation:
            self.linear2.append(
                nn.Conv1d(bn_channels,
                          out_channels,
                          kernel_size,
                          stride=stride,
                          padding=padding * _dilation,
                          dilation=_dilation,
                          bias=bias))
        self.select = StatsSelect(out_channels,
                                  len(dilation),
                                  null=null,
                                  reduction=reduction)

    def forward(self, x):
        if self.training and self.memory_efficient:
            x = cp.checkpoint(self.bn_function, x)
        else:
            x = self.bn_function(x)
        x = self.nonlinear2(x)
        x = self.select([linear(x) for linear in self.linear2])
        return x

class SEDenseTDNNLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 bn_channels,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 bias=False,
                 config_str='batchnorm-relu',
                 memory_efficient=False):
        super(SEDenseTDNNLayer, self).__init__()
        assert kernel_size % 2 == 1, 'Expect equal paddings, but got even kernel size ({})'.format(
            kernel_size)
        padding = (kernel_size - 1) // 2 * dilation
        self.memory_efficient = memory_efficient
        self.nonlinear1 = get_nonlinear(config_str, in_channels)
        self.linear1 = nn.Conv1d(in_channels, bn_channels, 1, bias=False)
        self.nonlinear2 = get_nonlinear(config_str, bn_channels)
        # self.linear2 = nn.Conv1d(bn_channels,
        #                          out_channels,
        #                          kernel_size,
        #                          stride=stride,
        #                          padding=padding,
        #                          dilation=dilation,
        #                          bias=bias)
        # self.se = SqueezeExcitation(out_channels)
        self.se = PoolingBlock(bn_channels,
                                out_channels,
                                kernel_size,
                                stride=stride,
                                padding=padding,
                                dilation=dilation,
                                bias=bias)

    def bn_function(self, x):
        return self.linear1(self.nonlinear1(x))

    def forward(self, x):
        if self.training and self.memory_efficient:
            x = cp.checkpoint(self.bn_function, x)
        else:
            x = self.bn_function(x)
        # x = self.linear2(self.nonlinear2(x))
        x = self.se(self.nonlinear2(x))
        return x

class SEDenseTDNNBlock(nn.ModuleList):
    def __init__(self,
                 num_layers,
                 in_channels,
                 out_channels,
                 bn_channels,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 bias=False,
                 config_str='batchnorm-relu',
                 memory_efficient=False):
        super(SEDenseTDNNBlock, self).__init__()
        for i in range(num_layers):
            layer = SEDenseTDNNLayer(in_channels=in_channels + i * out_channels,
                                   out_channels=out_channels,
                                   bn_channels=bn_channels,
                                   kernel_size=kernel_size,
                                   stride=stride,
                                   dilation=dilation,
                                   bias=bias,
                                   config_str=config_str,
                                   memory_efficient=memory_efficient)
            self.add_module('tdnnd%d' % (i + 1), layer)

    def forward(self, x):
        for layer in self:
            x = torch.cat([x, layer(x)], dim=1)
        return x

class MultiBranchDenseTDNNBlock(DenseTDNNBlock):
    def __init__(self,
                 num_layers,
                 in_channels,
                 out_channels,
                 bn_channels,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 bias=False,
                 null=False,
                 reduction=1,
                 config_str='batchnorm-relu',
                 memory_efficient=False):
        super(DenseTDNNBlock, self).__init__()
        for i in range(num_layers):
            layer = MultiBranchDenseTDNNLayer(
                in_channels=in_channels + i * out_channels,
                out_channels=out_channels,
                bn_channels=bn_channels,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                bias=bias,
                null=null,
                reduction=reduction,
                config_str=config_str,
                memory_efficient=memory_efficient)
            self.add_module('tdnnd%d' % (i + 1), layer)


class TransitLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 bias=True,
                 config_str='batchnorm-relu'):
        super(TransitLayer, self).__init__()
        self.nonlinear = get_nonlinear(config_str, in_channels)
        self.linear = nn.Conv1d(in_channels, out_channels, 1, bias=bias)

    def forward(self, x):
        x = self.nonlinear(x)
        x = self.linear(x)
        return x


class DenseLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 bias=False,
                 config_str='batchnorm-relu'):
        super(DenseLayer, self).__init__()
        self.linear = nn.Conv1d(in_channels, out_channels, 1, bias=bias)
        self.nonlinear = get_nonlinear(config_str, out_channels)

    def forward(self, x):
        if len(x.shape) == 2:
            x = self.linear(x.unsqueeze(dim=-1)).squeeze(dim=-1)
        else:
            x = self.linear(x)
        x = self.nonlinear(x)
        return x


if __name__ == '__main__':
    model = SqueezeExcitation(channels=32)
    model.eval() 

    x = torch.randn(1, 32, 298)
    y = model(x)
    print(y.size())
    from thop import profile
    macs, num_params = profile(model, inputs=(x, ))
    # num_params = sum(p.numel() for p in model.parameters())
    print("MACs: {} G".format(macs / 1e9))
    print("Params: {} M".format(num_params / 1e6))