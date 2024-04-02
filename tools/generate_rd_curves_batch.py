import argparse

import torch
import datetime
import os
import utils.common as common
import utils.algo as algo
import utils.header as header

import scipy.io as io
from utils.common import *
from utils.algo import *
from utils.header import *
from pathlib import Path
from test import repeat_eval_ckpt,eval_single_ckpt
from eval_utils import eval_utils
import re
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from time import time

from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, model_fn_decorator
from pcdet.utils import common_utils
import math


parser = argparse.ArgumentParser(description='generate rate-distortion curves')
parser.add_argument("--cfg_file", type=str, default='cfgs/nuscenes_models/cbgs_voxel0075_voxelnext.yaml', help="specify the config for training")
parser.add_argument(
        "--set", dest="set_cfgs", default=None, nargs=argparse.REMAINDER, help="set extra config keys if needed"
    )
parser.add_argument("--pretrained_model", type=str, default="../output/nuscene_models/voxelnext_nuscenes_kernel1.pth", help="specify a ckpt to start from")
parser.add_argument("--launcher", choices=["none", "pytorch", "slurm"], default="none") 
parser.add_argument("--fix_random_seed", action="store_true", default=False, help="")
parser.add_argument('--maxsteps', default=16, type=int,
                    help='number of Delta to enumerate')
parser.add_argument('--maxdeadzones', default=10, type=int,
                    help='number of sizes of dead zones')
parser.add_argument('--maxrates', default=11, type=int,
                    help='number of bit rates')
parser.add_argument('--gpuid', default=0, type=int,
                    help='gpu id')
parser.add_argument('--calib_num_iters', default=1, type=int)
parser.add_argument('--batch_size', default=2, type=int,
                    help='batch size')
parser.add_argument('--nchannelbatch', default=128, type=int,
                    help='number of filters for each quantization batch')
parser.add_argument('--part_id', default=0, type=int, help="break total layers into parts and process each part in each process.")
parser.add_argument('--num_parts', default=6, type=int)
parser.add_argument('--workers', default=1, type=int,)
parser.add_argument('--bias_corr_weight', '-bcw', action="store_true")
parser.add_argument('--disable-deadzone', action="store_true")
parser.add_argument('--keys_dist', nargs="+", default=['center', 'center_z', 'dim', 'rot', 'hm', 'vel'], help="keys to calculate distortion")
parser.add_argument('--sort_channels', action="store_true")

args = parser.parse_args()
args.save_to_file = False

cfg_from_yaml_file(args.cfg_file, cfg)
cfg.TAG = Path(args.cfg_file).stem
cfg.EXP_GROUP_PATH = "/".join(args.cfg_file.split("/")[1:-1])  # remove 'cfgs' and 'xxxx.yaml'

args.archname = cfg.MODEL.NAME

if args.set_cfgs is not None:
    cfg_from_list(args.set_cfgs, cfg)

path_output = ('%s_ndz_%04d_nr_%04d_ns_%04d_nf_%04d_%srdcurves_%schannelwise_opt_dist' % (args.archname, args.maxdeadzones, args.maxrates, \
                                                                   args.calib_num_iters, args.nchannelbatch, "bcw_" if args.bias_corr_weight else "", "sorted_" if args.sort_channels else ""))
isExists=os.path.exists(path_output)
if not isExists:
    os.makedirs(path_output)


output_dir = cfg.ROOT_DIR / 'output' / cfg.EXP_GROUP_PATH / cfg.TAG
output_dir.mkdir(parents=True, exist_ok=True)

eval_output_dir = output_dir / 'eval' / "pretrained" / cfg.DATA_CONFIG.DATA_SPLIT['test']
eval_output_dir.mkdir(parents=True, exist_ok=True)
log_file = eval_output_dir / ('log_eval_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)

# log to file
logger.info('**********************Start logging**********************')
gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)

for key, val in vars(args).items():
    logger.info('{:16} {}'.format(key, val))


test_set, test_loader, sampler = build_dataloader(
    dataset_cfg=cfg.DATA_CONFIG,
    class_names=cfg.CLASS_NAMES,
    batch_size=args.batch_size,
    dist=False, workers=args.workers, logger=logger, training=False
)

net = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=test_set)
# images, labels = loadvaldata(args.datapath, args.gpuid, testsize=args.testsize)

net.load_params_from_file(filename=args.pretrained_model, logger=logger, to_cpu=False, pre_trained_path=args.pretrained_model)

layers = findconv(net)
print('total number of layers: %d' % (len(layers)))


for l in range(0, len(layers)):
    layer_weights = layers[l].weight.clone()
    nchannels = get_num_output_channels(layer_weights)
    n_channel_elements = get_ele_per_output_channel(layer_weights)

net = net.cuda()



# if args.archname == 'VoxelNeXt':
#     feat_fn = lambda batch_dict: batch_dict["encoded_spconv_tensor"].dense()
#     dist_fn = lambda Y, Y_hat: ((Y - Y_hat) ** 2).mean()
# elif args.archname == 'SECONDNet':
#     feat_fn = lambda batch_dict: batch_dict["spatial_features_2d"]
#     dist_fn = lambda Y, Y_hat: (((Y - Y_hat) * ((Y != 0) | (Y_hat != 0)).float()) ** 2).mean()
# elif args.archname == "CenterPoint":
#     feat_fn = lambda batch_dict: batch_dict["spatial_features_2d"]
#     dist_fn = lambda Y, Y_hat: (((Y - Y_hat) * ((Y != 0) | (Y_hat != 0)).float()) ** 2).mean()


if not hasattr(cfg.MODEL, "BACKBONE_2D"):
    feat_fn = lambda batch_dict: batch_dict["encoded_spconv_tensor"].dense()
    dist_fn = lambda Y, Y_hat: ((Y - Y_hat) ** 2).mean()
else:
    feat_fn = lambda batch_dict: batch_dict["spatial_features_2d"]
    dist_fn = lambda Y, Y_hat: (((Y - Y_hat) * ((Y != 0) | (Y_hat != 0)).float()) ** 2).mean()

# net.train()
# Y = eval_single_ckpt(net, test_loader, args, eval_output_dir, logger, 0, dist_test=False, return_pred_dicts=True)
Y = eval_utils.eval_one_iter(net, test_loader, num_iters=args.calib_num_iters, feat_fn=feat_fn)
# Y = eval_utils.inference_one_iter_with_grad(net, test_loader, num_iters=args.calib_num_iters)

# loss = detection_norm(Y, keys_to_calc=args.keys_dist)
# loss.backward()

# grad_list = []
# for l in range(0, len(layers)):
#     grad_list.append(layers[l].weight.grad.clone())

# net.eval()


len_part = math.ceil(len(layers) / args.num_parts)

with torch.no_grad():
    for layerid in range(args.part_id * len_part, (args.part_id + 1) * len_part):
        layer_weights = layers[layerid].weight.clone()
        nchannels = get_num_output_channels(layer_weights)
        n_channel_elements = get_ele_per_output_channel(layer_weights)
        print('filter size %d %d ' % (nchannels, n_channel_elements))
        if args.sort_channels:
            inds_channel_range = torch.argsort(layer_weights.view(nchannels, -1).max(1)[0])
            # rev_range_per_channel = torch.argsort(range_per_channel)

        for f in range(0, nchannels, args.nchannelbatch):
            
            st_id = f
            if f + args.nchannelbatch > nchannels:
                ed_id = nchannels
            else:
                ed_id = f + args.nchannelbatch
            
            if args.sort_channels:
                inds = inds_channel_range[st_id: ed_id]
            else:
                inds = list(range(st_id, ed_id))

            rst_phi = torch.ones(args.maxdeadzones, args.maxrates, device=getdevice())
            rst_delta = torch.ones(args.maxdeadzones, args.maxrates, device=getdevice())
            rst_entropy = torch.ones(args.maxdeadzones, args.maxrates, device=getdevice())
            rst_dist = torch.ones(args.maxdeadzones, args.maxrates, device=getdevice())
            rst_rate = torch.ones(args.maxdeadzones, args.maxrates, device=getdevice())
            rst_rate_entropy = torch.ones(args.maxdeadzones, args.maxrates, device=getdevice())
            rst_delta_mse = torch.ones(args.maxdeadzones, args.maxrates, device=getdevice())
            rst_dist_mse = torch.ones(args.maxdeadzones, args.maxrates, device=getdevice())
            # rst_approx_dist = torch.ones(args.maxdeadzones, args.maxrates, device=getdevice())

            #scale = (layer_weights[st_id:ed_id, :].reshape(-1) ** 2).mean().sqrt().log2().floor()
            output_channels = get_output_channels_inds(layer_weights, inds)
            scale = (output_channels.reshape(-1) ** 2).mean().sqrt().log2().floor()
            
            end = time()

            for d in range(0, args.maxdeadzones):
                phi = deadzone_ratio(output_channels, d / args.maxdeadzones)
                for b in range(0, args.maxrates):

                    if b == 0:
                        start = scale - 10
                    else:
                        start = rst_delta[d, b-1] - 2

                    min_dist = 1e8
                    min_mse = 1e8
                    pre_mse = 1e8
                    pre_dist = 1e8
                    for s in range(0, args.maxsteps):
                        delta = start + s
                        quant_weights = output_channels.clone()
                        quant_weights = deadzone_quantize(output_channels, phi, 2**delta, b)
                        if args.bias_corr_weight:
                            quant_weights = bcorr_weight(output_channels, quant_weights)

                        cur_mse = ((quant_weights - output_channels)**2).mean()
                        # pdb.set_trace()
                        assign_output_channels_inds(layers[layerid].weight, inds, quant_weights)
                        
                        Y_hat = eval_utils.eval_one_iter(net, test_loader, num_iters=args.calib_num_iters, feat_fn=feat_fn)
                        # cur_dist = ((quant_weights - output_channels) * get_output_channels_inds(grad_list[layerid], inds)).pow(2).mean()
                        # cur_dist = detection_distortion(Y, Y_hat, keys_to_calc=args.keys_dist)
                        cur_dist = dist_fn(Y, Y_hat)
                        
                        print('%s | layer %d: filter %d deadzone ratio %6.6f bit rates %6.6f s %d: phi %2.12f delta %6.6f mse %6.6f entropy %6.6f rate %6.6f distortion %6.6f | time %f' \
                            % (args.archname, layerid, f, d / args.maxdeadzones, b, s, phi, delta, \
                               cur_mse, b, b * n_channel_elements, cur_dist, time() - end))
                        end = time()

                        if (cur_dist < min_dist):
                            rst_phi[d, b] = phi
                            rst_delta[d, b] = delta
                            rst_entropy[d, b] = calc_entropy(quant_weights)
                            rst_rate_entropy[d, b] = (ed_id - st_id) * rst_entropy[d, b] * n_channel_elements
                            rst_rate[d, b] = (ed_id - st_id) * b * n_channel_elements
                            rst_dist[d, b] = cur_dist
                            min_dist = cur_dist

                        # if (cur_mse < min_mse):
                        rst_delta_mse[d, b] = delta
                        rst_dist_mse[d, b] = cur_mse
                            # min_mse = cur_mse

                        layers[layerid].weight[:] = layer_weights[:]

                        # if (cur_dist > pre_dist) and (cur_mse > pre_mse):
                        #     break

                        pre_mse = cur_mse
                        pre_dist = cur_dist

                        del quant_weights, Y_hat, cur_dist, cur_mse

                        if b == 0:
                            break
                    print("min dist", min_dist)
                        

            io.savemat(('%s/%s_%03d_%04d.mat' % (path_output, args.archname, layerid, f)),
               {'rd_phi': rst_phi.cpu().numpy(), 'rd_delta': rst_delta.cpu().numpy(), 
               'rd_entropy': rst_entropy.cpu().numpy(), 'rd_rate': rst_rate.cpu().numpy(), 
                'rd_dist': rst_dist.cpu().numpy(), 'rd_rate_entropy': rst_rate_entropy.cpu().numpy(), 
                # 'rst_approx_dist': rst_approx_dist.cpu().numpy(),
                'rd_delta_mse': rst_delta_mse.cpu().numpy(), 'rst_dist_mse': rst_dist_mse.cpu().numpy()})
    
        if args.sort_channels:
            io.savemat(f'{path_output}/{args.archname}_{layerid:03d}_channel_inds.mat', {'channel_inds': inds_channel_range.cpu().numpy()})
