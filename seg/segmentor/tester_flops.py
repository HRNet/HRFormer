import time
import numpy as np
import torch

from lib.models.model_manager import ModelManager
from segmentor.tools.module_runner import ModuleRunner
from lib.utils.tools.logger import Logger as Log
from lib.utils.flops.flop_count import flop_count
from .tester import Tester


class TesterFLOPS(Tester):
    def __init__(self, configer, fig_num=5):
        super(TesterFLOPS, self).__init__(configer)
        self.fig_num = fig_num
        Log.info(f"Evaluate FLOPS on first {self.fig_num} images.")

    def test(self, data_loader=None):
        """
        Validation function during the train phase.
        """
        self.seg_net.eval()
        model = self.seg_net

        start_time = time.time()
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("number of params: {}".format(n_parameters))

        tmp = []
        tmp2 = []
        for j, data_dict in enumerate(self.test_loader):
            if j >= self.fig_num:
                Log.info(f"Evaluated FLOPS on first {self.fig_num} images.")
                break

            if isinstance(data_dict["img"], (tuple, list)):
                assert len(data_dict["img"]) == 1
                data_dict["img"] = data_dict["img"][0]
            if data_dict["img"].dim() == 3:
                data_dict["img"] = data_dict["img"].unsqueeze(0)
            inputs = data_dict["img"].cuda(non_blocking=True)

            with torch.no_grad():
                res = flop_count(model, (inputs,))
                tmp.append(sum(res.values()))
                t = measure_time(model, inputs)
                tmp2.append(t)

        Log.info(
            "flops: [{}], time: [{}], params: [{}]".format(
                fmt_res(np.array(tmp)), fmt_res(np.array(tmp2)), n_parameters
            )
        )


class TesterFLOPSFixedSize(Tester):
    def __init__(self, configer, shape=(1024, 512)):
        self.configer = configer
        self.module_runner = ModuleRunner(configer)
        self.model_manager = ModelManager(configer)
        self.seg_net = None
        self._init_model()

        self.shape = shape if isinstance(shape, (tuple, list)) else (shape, shape)
        assert len(self.shape) == 2
        Log.info(f"Evaluate FLOPS with image of shape {self.shape}.")

    def _init_model(self):
        self.seg_net = self.model_manager.semantic_segmentor()
        self.seg_net = self.module_runner.load_net(self.seg_net)
        self.seg_net.eval()

    def test(self, data_loader=None):
        """
        Validation function during the train phase.
        """
        self.seg_net.eval()
        model = self.seg_net

        start_time = time.time()
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("number of params: {}".format(n_parameters))

        inputs = torch.randn(1, 3, self.shape[0], self.shape[1]).cuda(non_blocking=True)
        with torch.no_grad():
            res = flop_count(model, (inputs,))
            flops = sum(res.values())
            ftime = measure_time(model, inputs)

        Log.info(f"flops: [{flops}], time: [{ftime}], params: [{n_parameters}]")


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
    return f"mean={data.mean():.6f}, std={data.std():.6f}, min={data.min():.6f}, max={data.max():.6f}"
