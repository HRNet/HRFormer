import argparse
import torch
import time
import torchvision
import importlib

import numpy as np
import tqdm


from tools.flop_count import flop_count
import models
from data.build import build_dataset
from config import get_config
from models import build_model

import os

print(os.getcwd())


def get_args_parser():
    parser = argparse.ArgumentParser(
        "Set transformer detector FLOPs computation", add_help=False
    )
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--fig_num", default=5, type=int)
    parser.add_argument("--mode", type=str)

    parser.add_argument(
        "--cfg",
        type=str,
        required=True,
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs="+",
    )

    # easy config modification
    parser.add_argument("--batch-size", type=int, help="batch size for single GPU")
    parser.add_argument("--data-path", type=str, help="path to dataset")
    parser.add_argument(
        "--zip",
        action="store_true",
        help="use zipped dataset instead of folder dataset",
    )
    parser.add_argument(
        "--cache-mode",
        type=str,
        default="part",
        choices=["no", "full", "part"],
        help="no: no cache, "
        "full: cache all data, "
        "part: sharding the dataset into nonoverlapping pieces and only cache one piece",
    )
    parser.add_argument("--resume", help="resume from checkpoint")
    parser.add_argument(
        "--accumulation-steps", type=int, help="gradient accumulation steps"
    )
    parser.add_argument(
        "--use-checkpoint",
        action="store_true",
        help="whether to use gradient checkpointing to save memory",
    )
    parser.add_argument(
        "--amp-opt-level",
        type=str,
        default="O1",
        choices=["O0", "O1", "O2"],
        help="mixed precision opt level, if O0, no amp is used",
    )
    parser.add_argument(
        "--output",
        default="output",
        type=str,
        metavar="PATH",
        help="root of output folder, the full path is <output>/<model_name>/<tag> (default: output)",
    )
    parser.add_argument("--tag", help="tag of experiment")
    parser.add_argument("--eval", action="store_true", help="Perform evaluation only")
    parser.add_argument(
        "--throughput", action="store_true", help="Test throughput only"
    )

    # distributed training
    parser.add_argument(
        "--local_rank",
        type=int,
        required=True,
        help="local rank for DistributedDataParallel",
    )
    parser.add_argument(
        "--data-set", default="IMNET", choices=["CIFAR", "IMNET", "IMNET_TSV"], type=str
    )

    args, unparsed = parser.parse_known_args()

    config = get_config(args)

    return args, config


def warmup(model, inputs, N=10):
    for i in range(N):
        out = model(inputs)
    torch.cuda.synchronize()


def measure_time(model, inputs, N=10):
    warmup(model, inputs)
    s = time.time()
    for i in range(N):
        out = model(inputs)
    torch.cuda.synchronize()
    t = (time.time() - s) / N
    return t


def fmt_res(data):
    return data.mean(), data.std(), data.min(), data.max()


import torch.distributed as dist

dist.init_process_group(
    "gloo", init_method="file:///tmp/somefile", rank=0, world_size=1
)
args, config = get_args_parser()


dataset, args.nb_classes = build_dataset(is_train=False, config=config)
images = []
for idx in range(args.fig_num):
    img, t = dataset[idx]
    images.append(img)

device = torch.device("cuda")
results = {}

for model_name in [args.cfg]:
    model = build_model(config)
    print(str(model))
    model.to(device)
    model.eval()
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of params: {}".format(n_parameters))
    with torch.no_grad():
        tmp = []
        tmp2 = []
        for img in tqdm.tqdm(images):
            inputs = img.to(device).unsqueeze(0)
            if args.mode == "flops":
                res = flop_count(model, (inputs,))
                tmp.append(sum(res.values()))
            else:
                tmp.append(0)
            t = measure_time(model, inputs)
            tmp2.append(t)

    results[model_name] = {
        "flops": fmt_res(np.array(tmp)),
        "time": fmt_res(np.array(tmp2)),
        "params": (n_parameters),
    }


print("=============================")
print("")
for r in results:
    print(r)
    for k, v in results[r].items():
        print(" ", k, ":", v)
