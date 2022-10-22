#  #!/usr/bin/env python3
#  -*- coding: utf-8 -*-
#  Copyright (c) 2021.  Luca Clissa
#  #Licensed under the Apache License, Version 2.0 (the "License");
#  #you may not use this file except in compliance with the License.
#  #You may obtain a copy of the License at
#  #http://www.apache.org/licenses/LICENSE-2.0
#  #Unless required by applicable law or agreed to in writing, software
#  #distributed under the License is distributed on an "AS IS" BASIS,
#  #WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  #See the License for the specific language governing permissions and
#  #limitations under the License.
__all__ = ['ResUnet', 'c_resunet', 'ResUnetEnc', 'c_resunet_enc','load_model', 'load_ae_inference', 'load_ae_to_bin',
           'ResUnetVAE', 'c_resunetVAE', '_resunetVAE']

from fastai.vision.all import *
from ._blocks import *
from ._utils import *
from model.utils import InverseSquareRootLinearUnit, Dec1, ConstrainedConv2d, LinConstr
#from fluocells.config import MODELS_PATH


class ResUnet(nn.Module):
    def __init__(self, n_features_start=16, n_out=1, c0=True):
        super(ResUnet, self).__init__()
        pool_ks, pool_stride, pool_pad = 2, 2, 0

        if c0:
            print('including c0 in the network')
            self.encoder = nn.ModuleDict({

                'colorspace': nn.Conv2d(3, 1, kernel_size=1, padding=0),

                # block 1
                'conv_block': ConvBlock(1, n_features_start),
                'pool1': nn.MaxPool2d(pool_ks, pool_stride, pool_pad),

                # block 2
                'residual_block1': ResidualBlock(n_features_start, 2 * n_features_start, is_conv=True),
                'pool2': nn.MaxPool2d(pool_ks, pool_stride, pool_pad),

                # block 3
                'residual_block2': ResidualBlock(2 * n_features_start, 4 * n_features_start, is_conv=True),
                'pool3': nn.MaxPool2d(pool_ks, pool_stride, pool_pad),

                # bottleneck
                'bottleneck': Bottleneck(4 * n_features_start, 8 * n_features_start, kernel_size=5, padding=2),
            })
        else:
            print('not including c0 in the network')
            self.encoder = nn.ModuleDict({

                # block 1
                'conv_block': ConvBlock(3, n_features_start),
                'pool1': nn.MaxPool2d(pool_ks, pool_stride, pool_pad),

                # block 2
                'residual_block1': ResidualBlock(n_features_start, 2 * n_features_start, is_conv=True),
                'pool2': nn.MaxPool2d(pool_ks, pool_stride, pool_pad),

                # block 3
                'residual_block2': ResidualBlock(2 * n_features_start, 4 * n_features_start, is_conv=True),
                'pool3': nn.MaxPool2d(pool_ks, pool_stride, pool_pad),

                # bottleneck
                'bottleneck': Bottleneck(4 * n_features_start, 8 * n_features_start, kernel_size=5, padding=2),
            })

        self.decoder = nn.ModuleDict({
            # block 6
            'upconv_block1': UpResidualBlock(n_in=8 * n_features_start, n_out=4 * n_features_start),

            # block 7
            'upconv_block2': UpResidualBlock(4 * n_features_start, 2 * n_features_start),

            # block 8
            'upconv_block3': UpResidualBlock(2 * n_features_start, n_features_start),
        })

        # output
        #self.head = Heatmap2d(
        #    n_features_start, n_out, kernel_size=1, stride=1, padding=0)
        if n_out > 1 and ae_bin:
            self.head = HeatmapAE(
            n_features_start, n_out, kernel_size=1, stride=1, padding=0)
        else:
            self.head = Heatmap(
            n_features_start, n_out, kernel_size=1, stride=1, padding=0)

    def _forward_impl(self, x: Tensor) -> Tensor:
        downblocks = []
        for lbl, layer in self.encoder.items():
            x = layer(x)
            if 'block' in lbl: downblocks.append(x)
            # NEXT loop is hon the values and so we don't hane the name as in the items of the previous loop
        for layer, long_connect in zip(self.decoder.values(), reversed(downblocks)):
            x = layer(x, long_connect)
        return self.head(x)

    def init_kaiming_normal(self, mode='fan_in'):
        print('Initializing conv2d weights with Kaiming He normal')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode=mode)
            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def _resunet(
        arch: str,
        n_features_start: int,
        n_out: int,
        c0: bool,
        #     block: Type[Union[BasicBlock, Bottleneck]],
        #     layers: List[int],
        pretrained: bool,
        progress: bool,
        **kwargs,
) -> ResUnet:
    model = ResUnet(n_features_start, n_out, c0)  # , **kwargs)
    model.__name__ = arch
    # TODO: implement weights fetching if not present
    if pretrained:
        print('error')
    else:
        model.init_kaiming_normal()
    return model

def c_resunet(arch='c-ResUnet', n_features_start: int = 16, n_out: int = 1, c0=True, pretrained: bool = False,
              progress: bool = True,
              **kwargs) -> ResUnet:
    r"""cResUnet model from `"Automating Cell Counting in Fluorescent Microscopy through Deep Learning with c-ResUnet"
    <https://www.nature.com/articles/s41598-021-01929-5>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resunet(arch=arch, n_features_start=n_features_start, n_out=n_out, c0=c0, pretrained=pretrained,
                    progress=progress, **kwargs)

def load_model(resume_path, device, n_features_start=16, n_out=1, ae_bin=False, fine_tuning=False
               ,unfreezed_layers=1):

    model = nn.DataParallel(c_resunet(arch='c-ResUnet', n_features_start=n_features_start, n_out=n_out, ae_bin=ae_bin,
                                      device=device).to(device))

    checkpoint_file = torch.load(resume_path)
    model.load_state_dict(checkpoint_file, strict=False)
    if fine_tuning:
        print('fine_tuning')
        if unfreezed_layers.isdecimal():
            unfreezed_layers = int(unfreezed_layers)
        for block in list(list(model.children())[0].named_children())[::-1]:  # encoder, decoder, head
            #print('unfreezing {} of {}'.format(unfreezed_layers, block))
            if block[0] == 'head':
                for nc, cc in list(block[1].named_children())[::-1]:  # [1] because 0 is the name
                    if unfreezed_layers > 0: #and isinstance(c, nn.Conv2d):
                        for n, p in cc.named_parameters():
                            p.requires_grad_(True)
                            #print(block, n, p.requires_grad)
                        print('unfreezed {}'.format(nc))
                    else:
                        for n, p in cc.named_parameters():
                            p.requires_grad_(False)
                            #print(block, n, p.requires_grad)
                unfreezed_layers = int(unfreezed_layers) - 1

            else:
                for nc, cc in list(block[1].named_children())[::-1]:
                    if unfreezed_layers > 0:
                        for n, p in cc.named_parameters():
                            p.requires_grad_(True)
                        print('unfreezed {}'.format(nc))
                    else:
                        for n, p in cc.named_parameters():
                            p.requires_grad_(False)
                        print('keep freezed {}'.format(nc))
                    unfreezed_layers = int(unfreezed_layers) - 1

        print('requires grad for each layer:')
        for block in list(list(model.children())[0].named_children())[::-1]:
            for n, p in list(block[1].named_parameters())[::-1]:
                print(n, p.requires_grad)

    return model

def load_ae_inference(resume_path, device, n_features_start=16, n_out=3, fine_tuning=False, unfreezed_layers=1):
    model = nn.DataParallel(c_resunet(arch='c-ResUnet', n_features_start=n_features_start, n_out=n_out,
                  device=device).to(device))

    layers_to_remove = ['module.head.conv2d.weight', 'module.head.conv2d.bias']
    layers_to_rename = ['module.head.conv2d_binary.weight', 'module.head.conv2d_binary.bias'] #take automatically the name (they are the last lauyers)
    checkpoint_file = torch.load(resume_path)
    for k in list(checkpoint_file.keys()):
        if k in layers_to_remove:
            checkpoint_file.pop(k)
    for k in list(checkpoint_file.keys()):
        if k in layers_to_rename:
            checkpoint_file[k.replace("_binary", "")] = checkpoint_file.pop(k)

    model.load_state_dict(checkpoint_file, strict=False)
    if fine_tuning:
        print('fine_tuning')
        if unfreezed_layers.isdecimal():
            unfreezed_layers = int(unfreezed_layers)
        for block in list(list(model.children())[0].named_children())[::-1]:  # encoder, decoder, head
            #print('unfreezing {} of {}'.format(unfreezed_layers, block))
            if block[0] == 'head':
                for nc, cc in list(block[1].named_children())[::-1]:  # [1] because 0 is the name
                    if unfreezed_layers > 0: #and isinstance(c, nn.Conv2d):
                        for n, p in cc.named_parameters():
                            p.requires_grad_(True)
                            #print(block, n, p.requires_grad)
                        print('unfreezed {}'.format(nc))
                    else:
                        for n, p in cc.named_parameters():
                            p.requires_grad_(False)
                            #print(block, n, p.requires_grad)
                unfreezed_layers = int(unfreezed_layers) - 1

            else:
                for nc, cc in list(block[1].named_children())[::-1]:
                    if unfreezed_layers > 0:
                        for n, p in cc.named_parameters():
                            p.requires_grad_(True)
                        print('unfreezed {}'.format(nc))
                    else:
                        for n, p in cc.named_parameters():
                            p.requires_grad_(False)
                        print('keep freezed {}'.format(nc))
                    unfreezed_layers = int(unfreezed_layers) - 1

        print('requires grad for each layer:')
        for block in list(list(model.children())[0].named_children())[::-1]:
            for n, p in list(block[1].named_parameters())[::-1]:
                print(n, p.requires_grad)

    return model


def load_ae_to_bin(resume_path, device, n_features_start=16, n_out=3, fine_tuning=False, unfreezed_layers=1):
    model = nn.DataParallel(c_resunet(arch='c-ResUnet', n_features_start=n_features_start, n_out=1,
                  device=device).to(device))

    layers_to_remove = ['module.head.conv2d.weight', 'module.head.conv2d.bias']
    checkpoint_file = torch.load(resume_path)
    for k in list(checkpoint_file.keys()):
        if k in layers_to_remove:
            checkpoint_file.pop(k)

    model.load_state_dict(checkpoint_file, strict=False)
    if fine_tuning:
        print('fine_tuning')
        if unfreezed_layers.isdecimal():
            unfreezed_layers = int(unfreezed_layers)
        for block in list(list(model.children())[0].named_children())[::-1]:  # encoder, decoder, head
            #print('unfreezing {} of {}'.format(unfreezed_layers, block))
            if block[0] == 'head':
                for nc, cc in list(block[1].named_children())[::-1]:  # [1] because 0 is the name
                    if unfreezed_layers > 0: #and isinstance(c, nn.Conv2d):
                        for n, p in cc.named_parameters():
                            p.requires_grad_(True)
                            #print(block, n, p.requires_grad)
                        print('unfreezed {}'.format(nc))
                    else:
                        for n, p in cc.named_parameters():
                            p.requires_grad_(False)
                            #print(block, n, p.requires_grad)
                unfreezed_layers = int(unfreezed_layers) - 1

            else:
                for nc, cc in list(block[1].named_children())[::-1]:
                    if unfreezed_layers > 0:
                        for n, p in cc.named_parameters():
                            p.requires_grad_(True)
                        print('unfreezed {}'.format(nc))
                    else:
                        for n, p in cc.named_parameters():
                            p.requires_grad_(False)
                        print('keep freezed {}'.format(nc))
                    unfreezed_layers = int(unfreezed_layers) - 1

        print('requires grad for each layer:')
        for block in list(list(model.children())[0].named_children())[::-1]:
            for n, p in list(block[1].named_parameters())[::-1]:
                print(n, p.requires_grad)

    return model


class ResUnetEnc(nn.Module):
    def __init__(self, n_features_start=16, code_dim=16):
        super(ResUnetEnc, self).__init__()
        pool_ks, pool_stride, pool_pad = 2, 2, 0
        self.code_dim = code_dim

        self.encoder = nn.ModuleDict({
            'colorspace': nn.Conv2d(3, 1, kernel_size=1, padding=0),

            # block 1
            'conv_block': ConvBlock(1, n_features_start),
            'pool1': nn.MaxPool2d(pool_ks, pool_stride, pool_pad),

            # block 2
            'residual_block1': ResidualBlock(n_features_start, 2 * n_features_start, is_conv=True),
            'pool2': nn.MaxPool2d(pool_ks, pool_stride, pool_pad),

            # block 3
            'residual_block2': ResidualBlock(2 * n_features_start, 4 * n_features_start, is_conv=True),
            'pool3': nn.MaxPool2d(pool_ks, pool_stride, pool_pad),

            # bottleneck
            'bottleneck': Bottleneck(4 * n_features_start, 8 * n_features_start, kernel_size=5, padding=2),
        })

        # output
        # self.head = Heatmap2d(
        #    n_features_start, n_out, kernel_size=1, stride=1, padding=0)

        self.features_size = self._get_final_flattened_size()
        # The architecture ends with a fully connected layer where the number
        # of neurons is equal to the number of input classes or to the code dim
        self.fc1 = nn.Linear(self.features_size, self.code_dim)
        # self.head = Embedding(self.features_size, self.code_dim)

    def _forward_impl(self, x: Tensor) -> Tensor:
        for lbl, layer in self.encoder.items():
            x = layer(x)
        x = x.view(-1, self.features_size)
        return self.fc1(x)

    def init_kaiming_normal(self, mode='fan_in'):
        print('Initializing conv2d weights with Kaiming He normal')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode=mode)
            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _get_final_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros(
                (1, 3, 128, 128)
            )
            for lbl, layer in self.encoder.items():
                x = layer(x)
                t, c, w, h = x.size()
        return t * c * w * h

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def _resunet_enc(
        arch: str,
        n_features_start: int,
        code_dim: int,
        #     block: Type[Union[BasicBlock, Bottleneck]],
        #     layers: List[int],
        pretrained: bool,
        progress: bool,
        **kwargs,
) -> ResUnetEnc:
    model = ResUnetEnc(n_features_start, code_dim)  # , **kwargs)
    model.__name__ = arch
    # TODO: implement weights fetching if not present
    if pretrained:
        print('error')
    else:
        model.init_kaiming_normal()
    return model


def c_resunet_enc(arch='c-ResUnetEnc', n_features_start: int = 16, code_dim=16,
                  pretrained: bool = False,
                  progress: bool = True,
                  **kwargs) -> ResUnetEnc:
    r"""cResUnet model from `"Automating Cell Counting in Fluorescent Microscopy through Deep Learning with c-ResUnet"
    <https://www.nature.com/articles/s41598-021-01929-5>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resunet_enc(arch=arch, n_features_start=n_features_start, code_dim=16, pretrained=pretrained,
                        progress=progress, **kwargs)



class ResUnetVAE(nn.Module):
    def __init__(self, n_features_start=16, zDim=64, n_out=1, n_outRec=3, fully_conv=False):
        super(ResUnetVAE, self).__init__()
        pool_ks, pool_stride, pool_pad = 2, 2, 0

        self.act2 = InverseSquareRootLinearUnit()
        self.n_out = n_out
        self.n_outRec = n_outRec
        self.zDim = zDim
        self.n_features_start = n_features_start
        self.fully_conv = fully_conv

        self.encoder = nn.ModuleDict({
            'colorspace': nn.Conv2d(3, 1, kernel_size=1, padding=0),

            # block 1
            'conv_block': ConvBlock(1, n_features_start),
            'pool1': nn.MaxPool2d(pool_ks, pool_stride, pool_pad),

            # block 2
            'residual_block1': ResidualBlock(n_features_start, 2 * n_features_start, is_conv=True),
            'pool2': nn.MaxPool2d(pool_ks, pool_stride, pool_pad),

            # block 3
            'residual_block2': ResidualBlock(2 * n_features_start, 4 * n_features_start, is_conv=True),
            'pool3': nn.MaxPool2d(pool_ks, pool_stride, pool_pad),

            # bottleneck
            #'bottleneck': BottleneckVAE(4 * n_features_start, 8 * n_features_start, kernel_size=5, padding=2,
            #                            featureDim=8 * self.n_features_start*64*64, zDim=64),
            'bottleneck': BottleneckVAE(4 * n_features_start, 8 * n_features_start, kernel_size=5, padding=2,
                                        featureDim=8 * self.n_features_start * 64 * 64, zDim=zDim),
        })

        #self.pre_up = Dec1(64, 128 * 64 * 64) # switch to (64, 128*64*64)>>>upconv_block:UpResidualBlock(n_in=8 * n_features_start, n_out=4 * n_features_start)
        #self.pre_up = Dec1(zDim, 64 * 64 * 64)
        self.pre_up = nn.Conv2d(1, 8 * n_features_start, 1, 1, padding=0)
        self.rebase_up = nn.ConvTranspose2d(n_features_start, 1, kernel_size=2, stride=2, padding=0)


        self.decoder_segm = nn.ModuleDict({
            # block 6
            'upconv_block1': UpResidualBlock(n_in=8 * n_features_start, n_out=4 * n_features_start),
            #'upconv_block1': UpResidualBlockVAE(n_in=4 * n_features_start, n_out=4 * n_features_start),

            # block 7
            'upconv_block2': UpResidualBlock(4 * n_features_start, 2 * n_features_start),

            # block 8
            'upconv_block3': UpResidualBlock(2*n_features_start, n_features_start),
        })

        self.decoder_rec = nn.ModuleDict({
            # block 6
            'upconv_block1NoConcat': UpResidualBlockNoConcat(n_in=8 * n_features_start, n_out=4 * n_features_start),
            #'upconv_block1': UpResidualBlockVAE(n_in=4 * n_features_start, n_out=4 * n_features_start),

            # block 7
            'upconv_block2NoConcat': UpResidualBlockNoConcat(4 * n_features_start, 2 * n_features_start),

            # block 8
            'upconv_block3NoConcat': UpResidualBlockNoConcat(2*n_features_start, n_features_start),
        })

        self.decoder_conc = nn.ModuleDict({
            # block 6
            'upconv_block1NoConv': UpResidualBlockNoConv(n_in=8 * n_features_start, n_out=4 * n_features_start),
            #'upconv_block1': UpResidualBlockVAE(n_in=4 * n_features_start, n_out=4 * n_features_start),

            # block 7
            'upconv_block2NoConvt': UpResidualBlockNoConv(4 * n_features_start, 2 * n_features_start),

            # block 8
            'upconv_block3NoConv': UpResidualBlockNoConv(2*n_features_start, n_features_start),
        })

        self.headSeg = HeatmapVAE(self.n_features_start, self.n_out, kernel_size=1, stride=1, padding=0)
        self.headRec = HeatmapVAERecon(self.n_features_start, self.n_outRec, kernel_size=1, stride=1, padding=0)

    def reparameterize_logvar(self, mu, log_var):
        """
        :param mu: mean from the encoder's latent space
        :param log_var: log variance from the encoder's latent space
        """
        std = torch.exp(0.5*log_var) # standard deviation
        eps = torch.randn_like(std) # `randn_like` as we need the same size
        sample = mu + (eps * std) # sampling
        return sample

    def reparameterize(self, mu, sigma):
        """
        :param mu: mean from the encoder's latent space
        :param log_var: log variance from the encoder's latent space
        """
        std = sigma # standard deviation
        eps = torch.randn_like(std) # `randn_like` as we need the same size
        sample = mu + (eps * std) # sampling
        return sample

    def _forward_impl(self, x: Tensor) -> Tensor:
        downblocks = []
        for lbl, layer in self.encoder.items():
            if "bottle" in lbl:
                mu, sigma, x_seg = layer(x)
                sigma = self.act2(sigma)
            else:
                x = layer(x)
                if "color" in lbl:
                    gray_rgb = x
                if "pool1" in lbl:
                    recon_base = x
                if 'block' in lbl: downblocks.append(x)

        # after bottleneck x came back to (bs,64,64,64) size becauese the view is used inside the bottnk block
        z = self.reparameterize(mu, sigma)
        z = self.pre_up(z)
        z = nn.ELU()(z)
        #x = nn.ReLU()(x) # TO REMOVE
        #x = x.view(-1, self.n_features_start * 4, 64, 64) #x = x.view(-1, self.n_features_start*8, 64, 64)

        # PATH FOR SEGMENTATION
        for layer, long_connect in zip(self.decoder_segm.values(), reversed(downblocks)):
            x_seg = layer(x_seg, long_connect)
        segm_out = self.headSeg(x_seg)

        # PATH FOR RECONSTRUCTION
        for lbl, layer in self.decoder_rec.items(): #downblock store the long connection
            z = layer(z)
        recon_out = self.headRec(z)

        #return mu, sigma, segm_out, recon_out
        return mu, sigma, segm_out, recon_out

    def init_kaiming_normal(self, mode='fan_in'):
        print('Initializing conv2d weights with Kaiming He normal')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode=mode)
            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

def _resunetVAE(
        arch: str,
        n_features_start: int,
        zDim: int,
        n_out: int,
        n_outRec: int,
        fully_conv: bool,
        #     block: Type[Union[BasicBlock, Bottleneck]],
        #     layers: List[int],
        pretrained: bool,
        progress: bool,
        **kwargs,
) -> ResUnet:
    model = ResUnetVAE(n_features_start, zDim , n_out,  n_outRec, fully_conv)  # , **kwargs)
    model.__name__ = arch
    # TODO: implement weights fetching if not present
    model.init_kaiming_normal()
    return model

def c_resunetVAE(arch='c-ResUnetVAE', n_features_start: int = 16, zDim: int = 64, n_out: int = 3,  n_outRec=3, fully_conv = False,
                 pretrained: bool = False, progress: bool = True,
              **kwargs) -> ResUnet:
    r"""cResUnet model from `"Automating Cell Counting in Fluorescent Microscopy through Deep Learning with c-ResUnet"
    <https://www.nature.com/articles/s41598-021-01929-5>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resunetVAE(arch=arch, n_features_start=n_features_start, zDim = zDim, n_out=n_out,  n_outRec=n_outRec, fully_conv=fully_conv,
                       pretrained=pretrained, progress=progress, **kwargs)