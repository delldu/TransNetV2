import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

import random
import pdb


class TransNetV2(nn.Module):
    def __init__(
        self,
        F=16,
        L=3,
        S=2,
        D=1024,
        use_mean_pooling=False,
        dropout_rate=0.5,
    ):
        super(TransNetV2, self).__init__()

        self.SDDCNN = nn.ModuleList(
            [StackedDDCNNV2(in_filters=3, n_blocks=S, filters=F)]
            + [StackedDDCNNV2(in_filters=(F * 2 ** (i - 1)) * 4, n_blocks=S, filters=F * 2 ** i) for i in range(1, L)]
        )

        self.frame_sim_layer = FrameSimilarity(
            sum([(F * 2 ** i) * 4 for i in range(L)]),
            lookup_window=101,
            output_dim=128,
            similarity_dim=128,
            use_bias=True,
        )
        self.color_hist_layer = ColorHistograms(lookup_window=101, output_dim=128)

        self.dropout = nn.Dropout(dropout_rate) if dropout_rate is not None else None

        output_dim = ((F * 2 ** (L - 1)) * 4) * 3 * 6  # 3x6 for spatial dimensions
        output_dim += 128  # use_frame_similarity
        output_dim += 128  # use_color_histograms

        self.fc1 = nn.Linear(output_dim, D)
        self.cls_layer1 = nn.Linear(D, 1)
        self.cls_layer2 = nn.Linear(D, 1)

        self.use_mean_pooling = use_mean_pooling
        self.eval()

    def forward(self, inputs):
        inputs = inputs.unsqueeze(0)
        assert (
            isinstance(inputs, torch.Tensor) and list(inputs.shape[2:]) == [27, 48, 3] and inputs.dtype == torch.uint8
        ), "incorrect input type and/or shape"
        # uint8 of shape [B, T, H, W, 3] to float of shape [B, 3, T, H, W]
        x = inputs.permute([0, 4, 1, 2, 3]).float()
        x = x.div_(255.0)

        block_features = []
        for block in self.SDDCNN:
            x = block(x)
            block_features.append(x)

        if self.use_mean_pooling:
            x = torch.mean(x, dim=[3, 4])
            x = x.permute(0, 2, 1)
        else:
            x = x.permute(0, 2, 3, 4, 1)
            x = x.reshape(x.shape[0], x.shape[1], -1)

        x = torch.cat([self.frame_sim_layer(block_features), x], 2)
        x = torch.cat([self.color_hist_layer(inputs), x], 2)

        x = self.fc1(x)
        x = F.relu(x)

        if self.dropout is not None:
            x = self.dropout(x)

        one_hot = self.cls_layer1(x)

        return one_hot, {"many_hot": self.cls_layer2(x)}


class StackedDDCNNV2(nn.Module):
    def __init__(
        self,
        in_filters,
        n_blocks,
        filters,
        shortcut=True,
        pool_type="avg",
    ):
        super(StackedDDCNNV2, self).__init__()
        assert pool_type == "max" or pool_type == "avg"

        self.shortcut = shortcut
        self.DDCNN = nn.ModuleList(
            [
                DilatedDCNNV2(
                    in_filters if i == 1 else filters * 4,
                    filters,
                    activation=F.relu if i != n_blocks else nn.Identity(),
                )
                for i in range(1, n_blocks + 1)
            ]
        )
        self.pool = nn.MaxPool3d(kernel_size=(1, 2, 2)) if pool_type == "max" else nn.AvgPool3d(kernel_size=(1, 2, 2))

    def forward(self, inputs):
        x = inputs

        shortcut_init = False
        shortcut = x  # Set fake value for torch.jit.script

        for block in self.DDCNN:
            x = block(x)
            if not shortcut_init:
                shortcut = x
                shortcut_init = True

        x = F.relu(x)
        x += shortcut

        x = self.pool(x)
        return x


class DilatedDCNNV2(nn.Module):
    def __init__(self, in_filters, filters, activation=None):  # not supported
        super(DilatedDCNNV2, self).__init__()

        self.Conv3D_1 = Conv3DConfigurable(in_filters, filters, 1, use_bias=False)
        self.Conv3D_2 = Conv3DConfigurable(in_filters, filters, 2, use_bias=False)
        self.Conv3D_4 = Conv3DConfigurable(in_filters, filters, 4, use_bias=False)
        self.Conv3D_8 = Conv3DConfigurable(in_filters, filters, 8, use_bias=False)

        self.bn = nn.BatchNorm3d(filters * 4, eps=1e-3)
        self.activation = activation

    def forward(self, inputs):
        conv1 = self.Conv3D_1(inputs)
        conv2 = self.Conv3D_2(inputs)
        conv3 = self.Conv3D_4(inputs)
        conv4 = self.Conv3D_8(inputs)

        x = torch.cat([conv1, conv2, conv3, conv4], dim=1)
        x = self.bn(x)
        x = self.activation(x)

        return x


class Conv3DConfigurable(nn.Module):
    def __init__(
        self,
        in_filters,
        filters,
        dilation_rate,
        separable=True,
        use_bias=True,
    ):
        super(Conv3DConfigurable, self).__init__()

        if separable:
            # (2+1)D convolution https://arxiv.org/pdf/1711.11248.pdf
            conv1 = nn.Conv3d(
                in_filters, 2 * filters, kernel_size=(1, 3, 3), dilation=(1, 1, 1), padding=(0, 1, 1), bias=False
            )
            conv2 = nn.Conv3d(
                2 * filters,
                filters,
                kernel_size=(3, 1, 1),
                dilation=(dilation_rate, 1, 1),
                padding=(dilation_rate, 0, 0),
                bias=use_bias,
            )
            self.layers = nn.ModuleList([conv1, conv2])
        else:
            conv = nn.Conv3d(
                in_filters,
                filters,
                kernel_size=3,
                dilation=(dilation_rate, 1, 1),
                padding=(dilation_rate, 1, 1),
                bias=use_bias,
            )
            self.layers = nn.ModuleList([conv])

    def forward(self, inputs):
        x = inputs
        for layer in self.layers:
            x = layer(x)
        return x


class FrameSimilarity(nn.Module):
    def __init__(
        self,
        in_filters,
        similarity_dim=128,
        lookup_window=101,
        output_dim=128,
        use_bias=False,
    ):
        super(FrameSimilarity, self).__init__()

        self.projection = nn.Linear(in_filters, similarity_dim, bias=use_bias)
        self.fc = nn.Linear(lookup_window, output_dim)

        self.lookup_window = lookup_window
        assert lookup_window % 2 == 1, "`lookup_window` must be odd integer"

    def forward(self, inputs: List[torch.Tensor]):
        x = torch.cat([torch.mean(x, dim=[3, 4]) for x in inputs], dim=1)
        x = torch.transpose(x, 1, 2)

        x = self.projection(x)
        x = F.normalize(x, p=2.0, dim=2)

        batch_size, time_window = x.shape[0], x.shape[1]
        similarities = torch.bmm(x, x.transpose(1, 2))  # [batch_size, time_window, time_window]
        similarities_padded = F.pad(similarities, [(self.lookup_window - 1) // 2, (self.lookup_window - 1) // 2])

        batch_indices = (
            torch.arange(0, batch_size, device=x.device)
            .view([batch_size, 1, 1])
            .repeat([1, time_window, self.lookup_window])
        )
        time_indices = (
            torch.arange(0, time_window, device=x.device)
            .view([1, time_window, 1])
            .repeat([batch_size, 1, self.lookup_window])
        )
        lookup_indices = (
            torch.arange(0, self.lookup_window, device=x.device)
            .view([1, 1, self.lookup_window])
            .repeat([batch_size, time_window, 1])
            + time_indices
        )

        similarities = similarities_padded[batch_indices, time_indices, lookup_indices]
        return F.relu(self.fc(similarities))


# To support torch.jit.script, we move get_bin() from compute_color_histograms()
def get_bin(frames):
    # returns 0 .. 511
    R, G, B = frames[:, :, 0], frames[:, :, 1], frames[:, :, 2]
    R, G, B = R >> 5, G >> 5, B >> 5
    return (R << 6) + (G << 3) + B


class ColorHistograms(nn.Module):
    def __init__(self, lookup_window=101, output_dim=128):
        super(ColorHistograms, self).__init__()

        self.fc = nn.Linear(lookup_window, output_dim)
        self.lookup_window = lookup_window
        assert lookup_window % 2 == 1, "`lookup_window` must be odd integer"

    @staticmethod
    def compute_color_histograms(frames):
        frames = frames.int()

        batch_size, time_window, height, width, no_channels = frames.shape
        assert no_channels == 3
        frames_flatten = frames.view(batch_size * time_window, height * width, 3)

        binned_values = get_bin(frames_flatten)
        frame_bin_prefix = (torch.arange(0, batch_size * time_window, device=frames.device) << 9).view(-1, 1)
        binned_values = (binned_values + frame_bin_prefix).view(-1)

        histograms = torch.zeros(batch_size * time_window * 512, dtype=torch.int32, device=frames.device)
        histograms.scatter_add_(
            0, binned_values, torch.ones(len(binned_values), dtype=torch.int32, device=frames.device)
        )

        histograms = histograms.view(batch_size, time_window, 512).float()
        histograms_normalized = F.normalize(histograms, p=2.0, dim=2)
        return histograms_normalized

    def forward(self, inputs):
        x = self.compute_color_histograms(inputs)

        batch_size, time_window = x.shape[0], x.shape[1]
        similarities = torch.bmm(x, x.transpose(1, 2))  # [batch_size, time_window, time_window]
        similarities_padded = F.pad(similarities, [(self.lookup_window - 1) // 2, (self.lookup_window - 1) // 2])

        batch_indices = (
            torch.arange(0, batch_size, device=x.device)
            .view([batch_size, 1, 1])
            .repeat([1, time_window, self.lookup_window])
        )
        time_indices = (
            torch.arange(0, time_window, device=x.device)
            .view([1, time_window, 1])
            .repeat([batch_size, 1, self.lookup_window])
        )
        lookup_indices = (
            torch.arange(0, self.lookup_window, device=x.device)
            .view([1, 1, self.lookup_window])
            .repeat([batch_size, time_window, 1])
            + time_indices
        )

        similarities = similarities_padded[batch_indices, time_indices, lookup_indices]

        return F.relu(self.fc(similarities))
