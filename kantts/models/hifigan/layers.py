import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv1d, ConvTranspose1d
from torch.nn.utils import weight_norm, remove_weight_norm
from kantts.models.utils import init_weights


def get_padding_casual(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


class CausalConv1d(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
    ):
        super(CausalConv1d, self).__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv1d = weight_norm(
            Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding=0,
                dilation=dilation,
                groups=groups,
                bias=bias,
                padding_mode=padding_mode,
            )
        )
        self.conv1d.apply(init_weights)

    def forward(self, x):  # bdt
        x = F.pad(
            x, (self.pad, 0, 0, 0, 0, 0), "constant"
        )  # described starting from the last dimension and moving forward.
        #  x = F.pad(x, (self.pad, self.pad, 0, 0, 0, 0), "constant")
        x = self.conv1d(x)[:, :, : x.size(2)]
        return x

    def remove_weight_norm(self):
        remove_weight_norm(self.conv1d)


#  FIXME: HACK to get shape right
class CausalConvTranspose1d(torch.nn.Module):
    """CausalConvTranspose1d module with customized initialization."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding=0,
        output_padding=0,
    ):
        """Initialize CausalConvTranspose1d module."""
        super(CausalConvTranspose1d, self).__init__()
        self.deconv = weight_norm(
            ConvTranspose1d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding=0,
                output_padding=0,
            )
        )
        self.stride = stride
        self.deconv.apply(init_weights)
        self.pad = kernel_size - stride

    def forward(self, x):
        """Calculate forward propagation.
        Args:
            x (Tensor): Input tensor (B, in_channels, T_in).
        Returns:
            Tensor: Output tensor (B, out_channels, T_out).
        """
        #  x = F.pad(x, (self.pad, 0, 0, 0, 0, 0), "constant")
        return self.deconv(x)[:, :, : -self.pad]
        #  return self.deconv(x)

    def remove_weight_norm(self):
        remove_weight_norm(self.deconv)


class ResidualBlock(torch.nn.Module):
    def __init__(
        self,
        channels,
        kernel_size=3,
        dilation=(1, 3, 5),
        nonlinear_activation="LeakyReLU",
        nonlinear_activation_params={"negative_slope": 0.1},
    ):
        super(ResidualBlock, self).__init__()
        assert kernel_size % 2 == 1, "Kernal size must be odd number."
        self.convs1 = nn.ModuleList(
            [
                CausalConv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=dilation[i],
                    padding=get_padding_casual(kernel_size, dilation[i]),
                )
                for i in range(len(dilation))
            ]
        )

        self.convs2 = nn.ModuleList(
            [
                CausalConv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=1,
                    padding=get_padding_casual(kernel_size, 1),
                )
                for i in range(len(dilation))
            ]
        )

        self.activation = getattr(torch.nn, nonlinear_activation)(
            **nonlinear_activation_params
        )

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = self.activation(x)
            xt = c1(xt)
            xt = self.activation(xt)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for layer in self.convs1:
            layer.remove_weight_norm()
        for layer in self.convs2:
            layer.remove_weight_norm()
