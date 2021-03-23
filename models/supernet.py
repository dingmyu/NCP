import logging
import functools
import torch
from torch import nn
from torch.nn import functional as F


def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None):
    return F.interpolate(input, size, scale_factor, mode, align_corners)


class Identity(nn.Module):
    """Module proxy for null op."""

    def forward(self, x):
        return x


def get_active_fn(name):
    """Select activation function."""
    active_fn = {
        'nn.ReLU6': functools.partial(nn.ReLU6, inplace=True),
        'nn.ReLU': functools.partial(nn.ReLU, inplace=True),
    }[name]
    return active_fn


def _make_divisible(v, divisor, min_value=None):
    """Make channels divisible to divisor.
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def conv3x3(in_planes, out_planes, stride=1, dilation=1):
    """3x3 convolution with padding."""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        dilation=dilation,
        bias=False)


class ConvBNReLU(nn.Sequential):
    """Convolution-BatchNormalization-ActivateFn."""

    def __init__(self,
                 in_planes,
                 out_planes,
                 kernel_size=3,
                 stride=1,
                 groups=1,
                 active_fn=None,
                 batch_norm_kwargs=None,
                 dilation=1,
                 padding=None,
                 **kwargs):
        if batch_norm_kwargs is None:
            batch_norm_kwargs = {}
        if not padding:
            padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes,
                      out_planes,
                      kernel_size,
                      stride,
                      padding,
                      dilation=dilation,
                      groups=groups,
                      bias=False),
            nn.BatchNorm2d(out_planes, **batch_norm_kwargs), active_fn() if active_fn is not None else Identity())


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 active_fn=None,
                 **kwargs
                 ):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = active_fn()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = None
        self.stride = stride
        if self.stride != 1 or inplanes != planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes,
                          planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


def get_block_wrapper(block_str):
    """Wrapper for netowork block."""

    if block_str == 'ConvBNReLU':
        return ConvBNReLU
    elif block_str == 'BasicBlock':
        return BasicBlock
    else:
        raise ValueError('Unknown type of blocks.')



class ParallelModule(nn.Module):
    def __init__(self,
                 num_branches=2,
                 block=get_block_wrapper('BasicBlock'),
                 num_blocks=[2, 2],
                 num_channels=[32, 32],
                 expand_ratio=6,
                 kernel_sizes=[3],
                 batch_norm_kwargs=None,
                 active_fn=get_active_fn('nn.ReLU6')):
        super(ParallelModule, self).__init__()

        self.num_branches = num_branches
        self.active_fn = active_fn
        self.batch_norm_kwargs = batch_norm_kwargs
        self.expand_ratio = expand_ratio
        self.kernel_sizes = kernel_sizes

        self._check_branches(
            num_branches, num_blocks, num_channels)
        self.branches = self._make_branches(
            num_branches, block, num_blocks, num_channels)

    def _check_branches(self, num_branches, num_blocks, num_channels):
        if num_branches != len(num_blocks):
            error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(
                num_branches, len(num_blocks))
            logging.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(
                num_branches, len(num_channels))
            logging.error(error_msg)
            raise ValueError(error_msg)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels):
        layers = []
        for i in range(0, num_blocks[branch_index]):
            layers.append(
                block(
                    num_channels[branch_index],
                    num_channels[branch_index],
                    expand_ratio=self.expand_ratio,
                    kernel_sizes=self.kernel_sizes,
                    stride=1,
                    batch_norm_kwargs=self.batch_norm_kwargs,
                    active_fn=self.active_fn))
        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []
        for i in range(num_branches):
            branches.append(
                self._make_one_branch(i, block, num_blocks, num_channels))
        return nn.ModuleList(branches)

    def forward(self, x):
        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])
        return x


class FuseModule(nn.Module):

    def __init__(self,
                 in_branches=1,
                 out_branches=2,
                 block=get_block_wrapper('BasicBlock'),
                 in_channels=[16],
                 out_channels=[16, 32],
                 expand_ratio=6,
                 kernel_sizes=[3],
                 batch_norm_kwargs=None,
                 active_fn=get_active_fn('nn.ReLU6'),
                 use_hr_format=False,
                 only_fuse_neighbor=True,
                 directly_downsample=True):
        super(FuseModule, self).__init__()

        self.out_branches = out_branches
        self.in_branches = in_branches
        self.active_fn = active_fn
        self.batch_norm_kwargs = batch_norm_kwargs
        self.expand_ratio = expand_ratio
        self.kernel_sizes = kernel_sizes
        self.only_fuse_neighbor = only_fuse_neighbor
        self.in_channels_large_stride = True  # see 3.
        if only_fuse_neighbor:
            self.use_hr_format = out_branches > in_branches
        else:
            self.use_hr_format = out_branches > in_branches and \
                                 not (out_branches == 2 and in_branches == 1)

        self.relu = functools.partial(nn.ReLU, inplace=False)
        if use_hr_format:
            block = ConvBNReLU
        block = ConvBNReLU

        fuse_layers = []
        for i in range(out_branches if not self.use_hr_format else in_branches):
            fuse_layer = []
            for j in range(in_branches):
                if only_fuse_neighbor:
                    if j < i - 1 or j > i + 1:
                        fuse_layer.append(None)
                        continue
                if j > i:
                    fuse_layer.append(nn.Sequential(
                        block(
                            in_channels[j],
                            out_channels[i],
                            expand_ratio=self.expand_ratio,
                            kernel_sizes=self.kernel_sizes,
                            stride=1,
                            batch_norm_kwargs=self.batch_norm_kwargs,
                            active_fn=self.relu if not use_hr_format else None,
                            kernel_size=1
                        ),
                        nn.Upsample(scale_factor=2 ** (j - i), mode='nearest')))
                elif j == i:
                    if use_hr_format and in_channels[j] == out_channels[i]:
                        fuse_layer.append(None)
                    else:
                        fuse_layer.append(
                            block(
                                in_channels[j],
                                out_channels[i],
                                expand_ratio=self.expand_ratio,
                                kernel_sizes=self.kernel_sizes,
                                stride=1,
                                batch_norm_kwargs=self.batch_norm_kwargs,
                                active_fn=self.relu if not use_hr_format else None,
                                kernel_size=3
                            ))
                else:
                    downsamples = []
                    if directly_downsample:
                        downsamples.append(
                            block(
                                in_channels[j],
                                out_channels[i],
                                expand_ratio=self.expand_ratio,
                                kernel_sizes=self.kernel_sizes,
                                stride=2 ** (i - j),
                                batch_norm_kwargs=self.batch_norm_kwargs,
                                active_fn=self.relu if not use_hr_format else None,
                                kernel_size=3
                            ))
                    else:
                        for k in range(i - j):
                            if self.in_channels_large_stride:
                                if k == i - j - 1:
                                    downsamples.append(
                                        block(
                                            in_channels[j],
                                            out_channels[i],
                                            expand_ratio=self.expand_ratio,
                                            kernel_sizes=self.kernel_sizes,
                                            stride=2,
                                            batch_norm_kwargs=self.batch_norm_kwargs,
                                            active_fn=self.relu if not use_hr_format else None,
                                            kernel_size=3
                                        ))
                                else:
                                    downsamples.append(
                                        block(
                                            in_channels[j],
                                            in_channels[j],
                                            expand_ratio=self.expand_ratio,
                                            kernel_sizes=self.kernel_sizes,
                                            stride=2,
                                            batch_norm_kwargs=self.batch_norm_kwargs,
                                            active_fn=self.relu,
                                            kernel_size=3
                                        ))
                            else:
                                if k == 0:
                                    downsamples.append(
                                        block(
                                            in_channels[j],
                                            out_channels[j + 1],
                                            expand_ratio=self.expand_ratio,
                                            kernel_sizes=self.kernel_sizes,
                                            stride=2,
                                            batch_norm_kwargs=self.batch_norm_kwargs,
                                            active_fn=self.relu if not (use_hr_format and i == j + 1) else None,
                                            kernel_size=3
                                        ))
                                elif k == i - j - 1:
                                    downsamples.append(
                                        block(
                                            out_channels[j + k],
                                            out_channels[i],
                                            expand_ratio=self.expand_ratio,
                                            kernel_sizes=self.kernel_sizes,
                                            stride=2,
                                            batch_norm_kwargs=self.batch_norm_kwargs,
                                            active_fn=self.relu if not use_hr_format else None,
                                            kernel_size=3
                                        ))
                                else:
                                    downsamples.append(
                                        block(
                                            out_channels[j + k],
                                            out_channels[j + k + 1],
                                            expand_ratio=self.expand_ratio,
                                            kernel_sizes=self.kernel_sizes,
                                            stride=2,
                                            batch_norm_kwargs=self.batch_norm_kwargs,
                                            active_fn=self.relu,
                                            kernel_size=3
                                        ))
                    fuse_layer.append(nn.Sequential(*downsamples))
            fuse_layers.append(nn.ModuleList(fuse_layer))
        if self.use_hr_format:
            for branch in range(in_branches, out_branches):
                fuse_layers.append(nn.ModuleList([block(
                    out_channels[branch - 1],
                    out_channels[branch],
                    expand_ratio=self.expand_ratio,
                    kernel_sizes=self.kernel_sizes,
                    stride=2,
                    batch_norm_kwargs=self.batch_norm_kwargs,
                    active_fn=self.relu,
                    kernel_size=3
                )]))
        self.fuse_layers = nn.ModuleList(fuse_layers)

    def forward(self, x):
        x_fuse = []
        if not self.only_fuse_neighbor:
            for i in range(len(self.fuse_layers) if not self.use_hr_format else self.in_branches):
                y = self.fuse_layers[i][0](x[0]) if self.fuse_layers[i][0] else x[0]
                for j in range(1, self.in_branches):
                    if self.fuse_layers[i][j]:
                        y = y + self.fuse_layers[i][j](x[j])
                    else:
                        y = y + x[j]
                x_fuse.append(self.relu(y))
            if self.use_hr_format:
                for branch in range(self.in_branches, self.out_branches):
                    x_fuse.append(self.fuse_layers[branch][0](x_fuse[branch - 1]))
        else:
            for i in range(len(self.fuse_layers) if not self.use_hr_format else self.in_branches):
                flag = 1
                for j in range(i-1, i+2):
                    if 0 <= j < self.in_branches:
                        if flag:
                            y = self.fuse_layers[i][j](x[j]) if self.fuse_layers[i][j] else x[j]
                            flag = 0
                        else:
                            if self.fuse_layers[i][j]:
                                y = y + resize(
                                    self.fuse_layers[i][j](x[j]),
                                    size=y.shape[2:],
                                    mode='bilinear',
                                    align_corners=False)
                            else:
                                y = y + x[j]
                x_fuse.append(self.relu()(y))
            if self.use_hr_format:
                for branch in range(self.in_branches, self.out_branches):
                    x_fuse.append(self.fuse_layers[branch][0](x_fuse[branch - 1]))
        return x_fuse


class MultiResolutionNet(nn.Module):

    def __init__(self,
                 num_classes=19,
                 input_size=224,
                 input_stride=4,
                 input_channel=[16, 16],
                 last_channel=90,
                 bn_momentum=0.1,
                 bn_epsilon=1e-5,
                 active_fn='nn.ReLU',
                 block='BasicBlock',
                 width_mult=1.0,
                 round_nearest=2,
                 expand_ratio=4,
                 kernel_sizes=[3],
                 network_setting=[
                     [1, [1], [24]],
                     [2, [2, 2], [18, 36]],
                     [3, [2, 2, 3], [18, 36, 72]],
                     [4, [2, 2, 3, 4], [18, 36, 72, 144]],
                     [4, [2, 2, 3, 4], [18, 36, 72, 144]]
                 ],
                 task='segmentation',
                 align_corners=False,
                 fcn_head_for_seg=True,
                 **kwargs):
        super(MultiResolutionNet, self).__init__()

        batch_norm_kwargs = {
            'momentum': bn_momentum,
            'eps': bn_epsilon
        }

        self.avg_pool_size = input_size // 32
        self.input_stride = input_stride
        self.input_channel = [_make_divisible(item * width_mult, round_nearest) for item in input_channel]
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        self.batch_norm_kwargs = batch_norm_kwargs
        self.active_fn = get_active_fn(active_fn)
        self.kernel_sizes = kernel_sizes
        self.expand_ratio = expand_ratio
        self.task = task
        self.align_corners = align_corners

        self.block = get_block_wrapper(block)
        self.network_setting = network_setting

        downsamples = []
        if self.input_stride > 1:
            downsamples.append(ConvBNReLU(
                3,
                input_channel[0],
                kernel_size=3,
                stride=2,
                batch_norm_kwargs=self.batch_norm_kwargs,
                active_fn=self.active_fn))
        if self.input_stride > 2:
            downsamples.append(ConvBNReLU(
                input_channel[0],
                input_channel[1],
                kernel_size=3,
                stride=2,
                batch_norm_kwargs=self.batch_norm_kwargs,
                active_fn=self.active_fn))
        self.downsamples = nn.Sequential(*downsamples)

        features = []
        for index in range(len(network_setting)):
            in_branches = 1 if index == 0 else network_setting[index - 1][0]
            in_channels = [input_channel[1]] if index == 0 else network_setting[index - 1][-1]
            features.append(
                FuseModule(
                    in_branches=in_branches,
                    out_branches=network_setting[index][0],
                    in_channels=in_channels,
                    out_channels=network_setting[index][-1],
                    block=self.block,
                    expand_ratio=self.expand_ratio,
                    kernel_sizes=self.kernel_sizes,
                    batch_norm_kwargs=self.batch_norm_kwargs,
                    active_fn=self.active_fn)
            )
            features.append(
                ParallelModule(
                    num_branches=network_setting[index][0],
                    num_blocks=network_setting[index][1],
                    num_channels=network_setting[index][2],
                    block=self.block,
                    expand_ratio=self.expand_ratio,
                    kernel_sizes=self.kernel_sizes,
                    batch_norm_kwargs=self.batch_norm_kwargs,
                    active_fn=self.active_fn)
            )

        if fcn_head_for_seg:
            self.transform = ConvBNReLU(
                sum(network_setting[-1][-1]),
                last_channel,
                kernel_size=1,
                batch_norm_kwargs=self.batch_norm_kwargs,
                active_fn=self.active_fn
            )
        else:
            self.transform = self.block(
                    sum(network_setting[-1][-1]),
                    last_channel,
                    expand_ratio=self.expand_ratio,
                    kernel_sizes=self.kernel_sizes,
                    stride=1,
                    batch_norm_kwargs=self.batch_norm_kwargs,
                    active_fn=self.active_fn,
                )

        self.classifier = nn.Conv2d(last_channel,
                                    num_classes,
                                    kernel_size=1)

        self.features = nn.Sequential(*features)

        self.init_weights()

    def _transform_inputs(self, inputs):
        """Transform inputs for decoder.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
        Returns:
            Tensor: The transformed inputs
        """

        upsampled_inputs = [
            resize(
                input=x,
                size=inputs[0].shape[2:] if self.task == 'segmentation' else inputs[-1].shape[2:],
                mode='bilinear',
                align_corners=self.align_corners) for x in inputs
        ]
        inputs = torch.cat(upsampled_inputs, dim=1)
        inputs = self.transform(inputs)
        return inputs

    def init_weights(self, pretrained=None):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.downsamples(x)
        x = self.features([x])
        x = self._transform_inputs(x)

        if self.task != 'segmentation':
            x = F.avg_pool2d(x, kernel_size=x.size()[2:])

        x = self.classifier(x)
        return x
