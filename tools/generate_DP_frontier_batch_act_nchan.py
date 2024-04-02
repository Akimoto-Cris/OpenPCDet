import argparse

import torch
import datetime
import os
import utils.common as common
import utils.algo as algo
import utils.header as header

import math
import scipy.io as io
from utils.common import *
from utils.algo import *
from utils.header import *
from pathlib import Path
from eval_utils import eval_utils
import re
import torch.nn as nn
import time

from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network
from pcdet.utils import common_utils


parser = argparse.ArgumentParser(description='generate rate-distortion curves')
parser.add_argument("--cfg_file", type=str, default='cfgs/nuscenes_models/cbgs_voxel0075_voxelnext_vtc.yaml', help="specify the config for training")
parser.add_argument(
        "--set", dest="set_cfgs", default=None, nargs=argparse.REMAINDER, help="set extra config keys if needed"
    )
parser.add_argument("--pretrained_model", type=str, default="../output/nuscene_models/voxelnext_nuscenes_kernel1.pth", help="specify a ckpt to start from")
parser.add_argument("--launcher", choices=["none", "pytorch", "slurm"], default="none") 
parser.add_argument("--fix_random_seed", action="store_true", default=False, help="")
parser.add_argument('--pathrdcurve', default='./rd_curves', \
                    type=str,
                    help='path of rate distortion curves')
parser.add_argument('--maxdeadzones', default=10, type=int,
                    help='number of sizes of dead zones')
parser.add_argument('--maxrates', default=11, type=int,
                    help='number of bit rates')
parser.add_argument('--gpuid', default=2, type=int,
                    help='gpu id')
parser.add_argument('--target_rates', nargs="+", type=float)
parser.add_argument('--batch_size', default=4, type=int,
                    help='batch size')
parser.add_argument('--workers', default=16, type=int,
                    help='number of works to read images')
parser.add_argument('--nchannelbatch', default=128, type=int,
                    help='number of channels for each quantization batch')
parser.add_argument('--closedeadzone', default=0, type=int,
                    help='swith to open or close dead zone')
parser.add_argument('--bitrangemin', default=0, type=int,
                    help='0 <= bitrangemin <= 10')
parser.add_argument('--bitrangemax', default=10, type=int,
                    help='0 <= bitrangemax <= 10')
parser.add_argument('--msqe', default=0, type=int,
                    help='use msqe to allocate bits')
parser.add_argument('--bit_rate', default=0, type=int,
                    help='use fixed-length code')
parser.add_argument('--relu_bitwidth', default=-1, type=int,
                    help='bit width of activations')
parser.add_argument('--bias_corr_weight', '-bcw', action="store_true")
parser.add_argument('--bit_list', nargs="+", type=int)
parser.add_argument('--save_checkpoint_dir', default="quantized_models", type=str)
args = parser.parse_args()
args.save_to_file = False

cfg_from_yaml_file(args.cfg_file, cfg)
cfg.TAG = Path(args.cfg_file).stem
cfg.EXP_GROUP_PATH = "/".join(args.cfg_file.split("/")[1:-1])  # remove 'cfgs' and 'xxxx.yaml'

args.archname = cfg.MODEL.NAME

if args.set_cfgs is not None:
    cfg_from_list(args.set_cfgs, cfg)


# maxrates = 17
if args.target_rates is None:
    args.target_rates = [4., 6., 8.]
    maxsteps = 3
else:
    maxsteps = len(args.target_rates)

output_dir = cfg.ROOT_DIR / 'output' / cfg.EXP_GROUP_PATH / cfg.TAG
output_dir.mkdir(parents=True, exist_ok=True)

eval_output_dir = output_dir / 'eval' / "dp_frontier" / cfg.DATA_CONFIG.DATA_SPLIT['test']
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

tarnet = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=test_set).cuda()
tarnet.load_params_from_file(filename=args.pretrained_model, logger=logger, to_cpu=False, pre_trained_path=args.pretrained_model)

tarlayers = findconv(tarnet)

hookedlayers = hooklayers(tarlayers)
tarnet.eval()

nlayers = len(tarlayers)
nweights = cal_total_num_weights(tarlayers)
print('total num layers %d weights on %s is %d' % (nlayers, args.archname, nweights))

_ = eval_utils.eval_one_epoch(cfg, args, tarnet, test_loader, 0, logger, dist_test=False, result_dir=eval_output_dir)

dimens = [hookedlayers[i].input if isinstance(tarlayers[i].layer, (nn.Conv2d, nn.ConvTranspose2d)) else hookedlayers[i].input.flip(0) for i in range(0,len(hookedlayers))]
for l in hookedlayers:
    l.close()

# print("\n".join((str(d) for d in dimens)))

rd_rate, rd_rate_entropy, rd_dist, rd_phi, rd_delta, rd_delta_mse, rd_dist_mse = \
        load_rd_curve_batch(args.archname, tarlayers, args.maxdeadzones, args.maxrates, args.pathrdcurve, args.nchannelbatch, \
                        closedeadzone=args.closedeadzone)
if args.bit_list is not None:
    bit_list = list(map(lambda x:x-1, args.bit_list))
    # import pdb; pdb.set_trace()
    rd_rate_entropy = [d[..., bit_list] for d in rd_rate_entropy]
    rd_dist = [d[..., bit_list] for d in rd_dist]
    rd_phi = [d[..., bit_list] for d in rd_phi]
    rd_delta = [d[..., bit_list] for d in rd_delta]


hist_sum_W_sse = torch.ones(maxsteps,device=getdevice()) * Inf
pred_sum_Y_sse = torch.ones(maxsteps,device=getdevice()) * Inf
hist_sum_coded = torch.ones(maxsteps,device=getdevice()) * Inf
hist_sum_coded_w = torch.ones(maxsteps,device=getdevice()) * Inf
hist_sum_denom = torch.ones(maxsteps,device=getdevice()) * Inf
hist_sum_denom_w = torch.ones(maxsteps,device=getdevice()) * Inf
hist_sum_non0s = torch.ones(maxsteps,len(tarlayers),device=getdevice()) * Inf


solve_times = []
for j in range(len(args.target_rates)):
    target_rate = args.target_rates[j]
    pc_phi, pc_delta, pc_bits, pc_size = dp_quantize(tarlayers, rd_rate, rd_dist, rd_phi, rd_delta, 
                                                    args.nchannelbatch, target_rate, device="cuda", piece_length=2**20)

    hist_sum_W_sse[j] = pred_sum_Y_sse[j] = 0.0
    hist_sum_coded[j] = hist_sum_coded_w[j] = hist_sum_denom[j] = hist_sum_denom_w[j] = 0.0

    with torch.no_grad():
        if args.output_bit_allocation:
            to_write = "" 

        sec = time.time()

        solve_times.append(time.time() - sec)
        layer_weights_ = [0] * len(tarlayers)
        total_rates_bits = cal_total_rates(tarlayers, pc_size, args.nchannelbatch)
        # print("total_rates_bits", total_rates_bits)
        
        for l in range(0,len(tarlayers)):
            ##load files here
            layer_weights = tarlayers[l].weight.clone()
            # layer_weights_idx = tarlayers[l].weight.clone()
            nchannels = get_num_output_channels(layer_weights)
            ngroups = math.ceil(nchannels / args.nchannelbatch)

            hist_sum_non0s[j,l] = (layer_weights != 0).any(1).sum()
            hist_sum_denom[j] = hist_sum_denom[j] + layer_weights.numel()
            hist_sum_denom_w[j] = hist_sum_denom_w[j] + layer_weights.numel()

            n_channel_elements = get_ele_per_output_channel(layer_weights)
            quant_weights = tarlayers[l].weight.clone()
            if "sort" in args.pathrdcurve:
                channel_inds = io.loadmat(f'{args.pathrdcurve}/{args.archname}_{l:03d}_channel_inds.mat')['channel_inds'][0]
                # channel_inds = torch.argsort(layer_weights.view(nchannels, -1).max(1)[0])


            for cnt, f in enumerate(range(0, nchannels, args.nchannelbatch)):
                st_layer = f
                ed_layer = f + args.nchannelbatch
                if f + args.nchannelbatch > nchannels:
                    ed_layer = nchannels
                
                if "sort" in args.pathrdcurve:
                    inds = channel_inds[st_layer: ed_layer]
                else:
                    inds = list(range(st_layer, ed_layer))
                # import pdb; pdb.set_trace()
                output_channels = get_output_channels_inds(layer_weights, inds)
                quant_output_channels = deadzone_quantize(output_channels, pc_phi[l][cnt], 2**(pc_delta[l][cnt]), pc_bits[l][cnt])
                quant_index_output_channels = deadzone_quantize_idx(output_channels, pc_phi[l][cnt], 2**(pc_delta[l][cnt]), pc_bits[l][cnt])
                if args.bias_corr_weight:
                    quant_output_channels = bcorr_weight(output_channels, quant_output_channels)

                assign_output_channels_inds(tarlayers[l].weight, inds, quant_output_channels)
                assign_output_channels_inds(quant_weights, inds, quant_index_output_channels)
                hist_sum_coded_w[j] = hist_sum_coded_w[j] + pc_bits[l][cnt] * output_channels.numel()
            
            hist_sum_W_sse[j] = hist_sum_W_sse[j] + ((tarlayers[l].weight - tarlayers[l].weight)**2).sum()
        print(f'W rate: {float(hist_sum_coded_w[j]) / hist_sum_denom_w[j]}')   
        hist_sum_coded[j] += total_rates_bits

        _ = eval_utils.eval_one_epoch(cfg, args, tarnet, test_loader, 0, logger, dist_test=False, result_dir=eval_output_dir)
        hist_sum_W_sse[j] = hist_sum_W_sse[j]/hist_sum_denom[j]
        hist_sum_coded[j] = hist_sum_coded[j]/hist_sum_denom[j]
        sec = time.time() - sec

        print('%s | target rate: %+5.1f, wmse: %5.2e, rate: %5.2f' %
              (args.archname, target_rate, hist_sum_W_sse[j], hist_sum_coded[j]))
        print(f'Avg Optimization Time: {sum(solve_times) / len(solve_times):.3f} s')
        
        with open(f'{args.archname}_acc_dist_curve_dp.txt', "a+") as f:
            f.write(f"{hist_sum_coded[j]}\n")
        
        # if hist_sum_coded[j] == 0.0 or \
        #    hist_sum_Y_tp1[j] <= 0.002:
        #     break

        if args.save_checkpoint_dir is not None:
            if not os.path.exists(args.save_checkpoint_dir):
                os.makedirs(args.save_checkpoint_dir)
            dataset_name = cfg.DATA_CONFIG.DATASET

            torch.save({"model_state": tarnet.state_dict()}, f'{args.save_checkpoint_dir}/{args.archname}_{dataset_name}_rate={hist_sum_coded[j]:.2f}.pth')

io.savemat(('%s_%s_sum_output.mat' % (args.archname, "" if not args.bias_corr_weight else "_bcw")),
           {'pred_sum_Y_sse':pred_sum_Y_sse.cpu().numpy(),'hist_sum_coded':hist_sum_coded.cpu().numpy(),
            'hist_sum_W_sse':hist_sum_W_sse.cpu().numpy(),'hist_sum_denom':hist_sum_denom.cpu().numpy(),
            'hist_sum_non0s':hist_sum_non0s.cpu().numpy()})
