import torch
import torch.nn as nn
import torchvision.datasets as datasets
import scipy.io as io
import utils.transconv as transconv

from .utils import get_weights,get_all_modules,get_weights_2d,get_modules_3d,get_modules_2d,get_modules_2d_head, _is_2d_prunable_module, _is_prunable_module
import numpy as np

from spconv.pytorch.modules import SparseModule, SparseBatchNorm, SparseReLU
import spconv.pytorch as spconv
from pcdet.models.backbones_2d.base_bev_backbone import BaseBEVBackbone
from pcdet.models.dense_heads.voxelnext_head import VoxelNeXtHead
from pcdet.models.backbones_2d.base_bev_backbone import BaseBEVBackbone
from pcdet.models.backbones_2d.map_to_bev.height_compression import HeightCompression
from pcdet.models.backbones_3d.spconv_backbone import VoxelBackBone8x
from pcdet.models.dense_heads import VoxelNeXtHead,AnchorHeadSingle,CenterHead


# def _is_2d_prunable_module(m):
#     # if (isinstance(m,BaseBEVBackbone)):
#     return (isinstance(m,nn.Linear) or isinstance(m,nn.Conv2d)) or isinstance(m,nn.ConvTranspose2d)

# def _is_prunable_module(m):
#     return  isinstance(m,spconv.SubMConv3d) or isinstance(m,spconv.SparseConv3d)


spconv.modules.is_spconv_module = lambda m: isinstance(m, (SparseModule, SparseBatchNorm, SparseReLU, transconv.QAWrapper))


device = torch.device("cuda:0")

PARAMETRIZED_MODULE_TYPES = (
    nn.Linear,
    nn.Conv2d,
    nn.ConvTranspose2d,
    spconv.SubMConv3d,
    spconv.SparseConv3d
)
NORM_MODULE_TYPES = (torch.nn.BatchNorm2d, torch.nn.LayerNorm)


def findConvRELU(container, flags):
    if isinstance(container,torch.nn.Sequential):
        for attr in range(0,len(container)):
            findConvRELU(container[attr],flags)
        return
    elif isinstance(container, torch.nn.Conv2d):
        flags.append(0)
        return
    elif isinstance(container, torch.nn.Linear):
        flags.append(0)
        return
    return


def quantize(weights, delta, b):
    if b > 0:
        minpoint = -(2**(b-1))*delta
        maxpoint = (2**(b-1) - 1)*delta
    else:
        minpoint = 0
        maxpoint = 0
    # return (delta*(weights/delta).round()).clamp(minpoint,maxpoint)
    return weights.div(delta).round_().mul_(delta).clamp_(minpoint, maxpoint)


def getdevice():
	global device
	return device



def findconv(container):
    modules = get_modules_3d(container)
    modules.extend(get_modules_2d(container))
    return modules



class TimeWrapper(nn.Module):
    def __init__(self, wrapped_module, tag):
        super().__init__()
        self.wrapped_module = wrapped_module
        self.tag = tag

    def forward(self, *args):
        ret = self.wrapped_module(*args)
        if isinstance(ret, torch.Tensor):
            print(f"ModuleType: {self.tag}, shape:", ret.shape, "count:", ret.numel())
        else:
            print(f"ModuleType: {self.tag}, shape:", [r.shape for r in ret], "count:", sum(r.numel() for r in ret))
        return ret
    
    @property
    def weight(self):
        return getattr(self.wrapped_module, "weight", None)


def convert_qconv(network, stats=True):
    layers = findconv(network)

    with torch.no_grad():
        for l in range(0, len(layers)):
            layers[l] = transconv.QAWrapper(layers[l])
            layers[l].register_buffer('coded', torch.tensor([0.]))
            layers[l].register_buffer('delta', torch.tensor([0.]))
            layers[l].stats = stats

        network = replacelayer(network, [layers], PARAMETRIZED_MODULE_TYPES)

    return network.cuda(), layers


def convert_qconv_forfinetune(network):
    layers = findconv(network)

    with torch.no_grad():
        for l in range(0, len(layers)):
            layers[l] = transconv.QAWrapper(layers[l], True, True)
            layers[l].register_buffer("delta", torch.tensor([0.]))
            layers[l].register_buffer("coded", torch.tensor([0.]))

        network = replacelayer(network, [layers], PARAMETRIZED_MODULE_TYPES)
        for n, p in network.named_parameters():
            if "weight" in n:
                p.requires_grad = False

    return network.cuda()


# def convert_qconv(model, stats=True):
#     print(findconv(model))
#     for name, module in reversed(model._modules.items()):
#         if module is None:
#             continue

#         if len(list(module.children())) > 0:
#             model._modules[name] = convert_qconv(model=module)

#         if _is_2d_prunable_module(module) or _is_prunable_module(module):
#             layer_new = transconv.QAWrapper(module)
#             layer_new.stats = stats
#             model._modules[name] = layer_new

#     return model


def replacelayer(module, layers, classes):
    module_output = module
    # base case
    if isinstance(module, classes):
        module_output, layers[0] = layers[0][0], layers[0][1:]
    if len(layers[0]):
        # recursive
        for name, child in module.named_children():
            module_output.add_module(name, replacelayer(child, layers, classes))
    del module
    return module_output


def hooklayers(layers, backward=False, store_tensor=False):
    return [Hook(layer, backward, store_tensor=store_tensor) for layer in layers]


def hooklayers_with_fp_act(layers, fp_acts):
    return [Hook(layer, fp_act=fp_act) for layer, fp_act in zip(layers, fp_acts)]


class Hook:
    def __init__(self, module, backward=False, fp_act=None, store_tensor=False):
        self.backward = backward
        if not backward:
            self.hook = module.register_forward_hook(self.hook_fn)
        else:
            self.hook = module.register_backward_hook(self.backward_hook_fn)
        self.fp_act = fp_act
        self.store_tensor = store_tensor

    def hook_fn(self, module, input, output):
        if self.store_tensor:
            if hasattr(input[0], "features"):
                self.input_tensor = input[0].features
                self.output_tensor = output.feaPtures
            else:
                self.input_tensor = input[0]
                self.output_tensor = output
        
            self.input = torch.tensor(self.input_tensor.shape[1:])
            self.output = torch.tensor(self.output_tensor.shape[1:])
        else:
            if hasattr(input[0], "features"):
                self.input = torch.tensor(input[0].features.shape[1:])
                self.output = torch.tensor(output.features.shape[1:])
            else:
                self.input = torch.tensor(input[0].shape[1:])
                self.output = torch.tensor(output.shape[1:])
        if module.stats:
            self.mean_err_a = module.sum_err_a / module.count
        if self.fp_act is not None:
            # self.accum_err_act = (self.fp_act - self.input[0]).div_(self.fp_act.max()).pow_(2).mean()
            # print(self.fp_act.unique())
            self.accum_err_act = (self.fp_act - self.input_tensor).div_(self.fp_act.abs().max()).pow_(2).mean()

    def backward_hook_fn(self, module, grad_input, grad_output):
        if self.store_tensor:
            self.input_tensor = grad_input
            self.output_tensor = grad_output
        
            self.input = torch.tensor(self.input_tensor[0].shape[1:])
            self.output = torch.tensor(self.output_tensor[0].shape[1:])
        else:
            if hasattr(input[0], "features"):
                self.input = torch.tensor(grad_input[0].shape[1:])
                self.output = torch.tensor(grad_output[0].shape[1:])
            else:
                self.input = torch.tensor(grad_input[0].shape[1:])
                self.output = torch.tensor(grad_output[0].shape[1:])

    def close(self):
        self.hook.remove()

def gettop1(logits):
    return logits.max(1)[1]


def gettopk(logp,k=1):
    logp = logp.exp()
    logp = logp/logp.sum(1).reshape(-1,1)
    vals, inds = logp.topk(k,dim=1)

    return inds


def loadvarstats(archname,testsize):
    mat = io.loadmat(('%s_stats_%d.mat' % (archname, testsize)))
    return np.array(mat['cov'])


def detection_distortion(Y, Y_hat, keys_to_calc=["pred_boxes", "pred_scores"]):
    dist = 0.
    for key in keys_to_calc:
        all_tensor_y = torch.cat([y[key] for y in Y if key in y], dim=1)
        all_tensor_y_hat = torch.cat([y_hat[key] for y_hat in Y_hat if key in y_hat], dim=1)
        assert all_tensor_y.shape == all_tensor_y_hat.shape

        dist += (all_tensor_y - all_tensor_y_hat).pow_(2).mean().item()
    return dist


def detection_norm(Y, keys_to_calc=["pred_boxes", "pred_scores"]):
    norm = 0.
    for key in keys_to_calc:
        all_tensor_y = torch.cat([y[key] for y in Y if key in y], dim=1)
        norm += all_tensor_y.pow(2).mean()
    return norm
            

def loadrdcurves(archname,l,g,part,nchannelbatch=-1, Amse=False):
    if nchannelbatch>0:
        # print(f"loading act curves from: {archname}_nr_0011_ns_0064_nf_{nchannelbatch:04d}_rdcurves_channelwise_opt_dist_act{'_Amse' if Amse else ''}")
        mat = io.loadmat(f'{archname}_nr_0011_ns_0001_nf_{nchannelbatch:04d}_rdcurves_channelwise_opt_dist_act{"_Amse" if Amse else ""}/{archname}_val_{l:03d}_{g:04d}_output_{part}')
    else:
        # print(f"loading act curves from: {archname}_nr_0011_ns_0064_rdcurves_channelwise_opt_dist_act/{archname}_val_{l:03d}_0064_output_{part}")
        mat = io.loadmat(f'{archname}_nr_0011_ns_0001_rdcurves_channelwise_opt_dist_act/{archname}_val_{l:03d}_0064_output_{part}')
    return mat['%s_Y_sse'%part], mat['%s_delta'%part], mat['%s_coded'%part]
    #mat = io.loadmat('%s_%s_val_1000_%d_%d_output_%s_%s' % (archname,tranname,l+1,l+1,trantype,part))
    #return mat['%s_Y_sse'%part][l,0], mat['%s_delta'%part][l,0], mat['%s_coded'%part][l,0]


def findrdpoints(y_sse,delta,coded,lam_or_bit, is_bit=False):
    # find the optimal quant step-size
    y_sse[np.isnan(y_sse)] = float('inf')
    ind1 = np.nanargmin(y_sse,1)
    ind0 = np.arange(ind1.shape[0]).reshape(-1,1).repeat(ind1.shape[1],1)
    ind2 = np.arange(ind1.shape[1]).reshape(1,-1).repeat(ind1.shape[0],0)
    inds = np.ravel_multi_index((ind0,ind1,ind2),y_sse.shape) # bit_depth x blocks
    y_sse = y_sse.reshape(-1)[inds]
    delta = delta.reshape(-1)[inds]
    coded = coded.reshape(-1)[inds]
    # mean = mean.reshape(-1)[inds]
    # find the minimum Lagrangian cost
    if is_bit:
        point = coded == lam_or_bit
    else:
        point = y_sse + lam_or_bit*coded == (y_sse + lam_or_bit*coded).min(0)
    return np.select(point, y_sse), np.select(point, delta), np.select(point, coded)#, np.select(point, mean)


# import pickle as pkl
# def loadmeanstd(archname, l, part):
#     with open(f'{archname}_nr_0011_ns_0064_rdcurves_channelwise_opt_dist_act/{archname}_val_{l:03d}_0064_output_{part}_meanstd.pkl', 'rb') as f:
#         d = pkl.load(f)
#     return d

class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.v = 0
        self.sum = 0
        self.cnt = 0

    def update(self, v):
        self.v = v
        self.sum += v
        self.cnt += 1

    @property
    def avg(self):
        return self.sum / self.cnt
