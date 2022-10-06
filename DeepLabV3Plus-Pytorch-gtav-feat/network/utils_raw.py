import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from collections import OrderedDict
# from .adain import AdaptiveInstanceNormalization
import random

class _SimpleSegmentationModel(nn.Module):
    def __init__(self, backbone, classifier):
        super(_SimpleSegmentationModel, self).__init__()
        self.backbone = backbone
        self.classifier = classifier
        
    def forward(self, x, feat_align=False):
        input_shape = x.shape[-2:]

        features = self.backbone(x, feat_align)

        # x,f_out, f_low = self.classifier(features)
        x = self.classifier(features)

        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        # f_out = F.interpolate(f_out, size=input_shape, mode='bilinear', align_corners=False)

        return x

class IntermediateLayerGetter(nn.ModuleDict):
    """
    Module wrapper that returns intermediate layers from a model

    It has a strong assumption that the modules have been registered
    into the model in the same order as they are used.
    This means that one should **not** reuse the same nn.Module
    twice in the forward if you want this to work.

    Additionally, it is only able to query submodules that are directly
    assigned to the model. So if `model` is passed, `model.feature1` can
    be returned, but not `model.feature1.layer2`.

    Arguments:
        model (nn.Module): model on which we will extract the features
        return_layers (Dict[name, new_name]): a dict containing the names
            of the modules for which the activations will be returned as
            the key of the dict, and the value of the dict is the name
            of the returned activation (which the user can specify).

    Examples::

        >>> m = torchvision.models.resnet18(pretrained=True)
        >>> # extract layer1 and layer3, giving as names `feat1` and feat2`
        >>> new_m = torchvision.models._utils.IntermediateLayerGetter(m,
        >>>     {'layer1': 'feat1', 'layer3': 'feat2'})
        >>> out = new_m(torch.rand(1, 3, 224, 224))
        >>> print([(k, v.shape) for k, v in out.items()])
        >>>     [('feat1', torch.Size([1, 64, 56, 56])),
        >>>      ('feat2', torch.Size([1, 256, 14, 14]))]
    """
    def __init__(self, model, return_layers, hrnet_flag=False):
        if not set(return_layers).issubset([name for name, _ in model.named_children()]):
            raise ValueError("return_layers are not present in model")

        self.hrnet_flag = hrnet_flag

        orig_return_layers = return_layers
        return_layers = {k: v for k, v in return_layers.items()}
        layers = OrderedDict()
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break

        super(IntermediateLayerGetter, self).__init__(layers)
        self.return_layers = orig_return_layers

    def forward(self, x, feat_align):
        
        out = OrderedDict()
        for name, module in self.named_children():
            if self.hrnet_flag and name.startswith('transition'): # if using hrnet, you need to take care of transition
                if name == 'transition1': # in transition1, you need to split the module to two streams first
                    x = [trans(x) for trans in module]
                else: # all other transition is just an extra one stream split
                    x.append(module(x[-1]))
            else: # other models (ex:resnet,mobilenet) are convolutions in series.
                
                x = module(x)
                if feat_align and (name == 'conv1') and random.random() < 0.5:
                    x = x_adain(x)
                # if feat_align and name == 'bn1':   
                #     pass
                # else:                 
                #     x = module(x)
                #     if feat_align and name == 'conv1':
                #         x = x_adain(x)
                

            if name in self.return_layers:
                out_name = self.return_layers[name]
                if name == 'stage4' and self.hrnet_flag: # In HRNetV2, we upsample and concat all outputs streams together
                    output_h, output_w = x[0].size(2), x[0].size(3)  # Upsample to size of highest resolution stream
                    x1 = F.interpolate(x[1], size=(output_h, output_w), mode='bilinear', align_corners=False)
                    x2 = F.interpolate(x[2], size=(output_h, output_w), mode='bilinear', align_corners=False)
                    x3 = F.interpolate(x[3], size=(output_h, output_w), mode='bilinear', align_corners=False)
                    x = torch.cat([x[0], x1, x2, x3], dim=1)
                    out[out_name] = x
                else:
                    out[out_name] = x
        return out


def x_adain(x_cont):

    # assert (x_cont.size()[:2] == x_style.size()[:2])
    size = x_cont.size()
    # style_mean, style_std = calc_mean_std(x_style)
    content_mean, content_std = calc_mean_std(x_cont)
    
    size_mean = content_mean.size()
    
    index = torch.randint(0, size[0], (1, size[0]))
    index = list(index)
    # style_mean = content_mean.clone().detach()[index] * (torch.rand(size_mean).to(torch.device('cuda')) - 0.5) * 0.1
    style_mean = content_mean.clone().detach()[index] * ((torch.rand(size_mean).to(torch.device('cuda')) - 0.5)  + 1)
    # style_mean = content_mean.clone().detach()[index] * (torch.rand(size[0], size[1]).to(torch.device('cuda'))) * 2 
    # style_mean = content_mean.clone().detach()[index] 
    
    # style_std = content_std.clone().detach()[index] * (torch.rand(size_mean).to(torch.device('cuda'))) * 0.1
    style_std = content_std.clone().detach()[index] * ((torch.rand(size_mean).to(torch.device('cuda'))- 0.5)  + 1)
    # style_std = content_std.clone().detach()[index] * (torch.rand(size[0], size[1]).to(torch.device('cuda'))) * 2
    # style_std = content_std.clone().detach()[index]
    
        

    
     
    

    # style_mean = (content_mean.max()-content_mean.min()) * torch.randn(size_mean).to(torch.device('cuda')) + content_mean.min()
    # style_std = (content_std.max()-content_std.min()) * torch.randn(size_mean).to(torch.device('cuda')) + content_std.min()

    # style_mean = (torch.randn(size_mean).to(torch.device('cuda')) - 1) * content_mean
    # style_std = (torch.randn(size_mean).to(torch.device('cuda')) - 1) * content_std

    
    normalized_x_cont = (x_cont - content_mean.expand(size))/content_std.expand(size)
    denormalized_x_cont = normalized_x_cont * style_std.expand(size) + style_mean.expand(size)

    return denormalized_x_cont
    
def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std