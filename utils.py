import random
import torch
import numpy as np
import argparse
from default_config import getDefaultConfig
import os
import json
from model import CPCAR, buildTransformerAR, CPCModel, ConcatenatedModel
from copy import deepcopy
from bisect import bisect_left


def setSeed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def loadArgs(args, locArgs, forbiddenAttr=None):
    for k, v in vars(locArgs).items():
        if forbiddenAttr is not None:
            if k not in forbiddenAttr:
                setattr(args, k, v)
        else:
            setattr(args, k, v)


def getCheckpointData(pathDir):
    if not os.path.isdir(pathDir):
        return None
    checkpoints = [x for x in os.listdir(pathDir)
                   if os.path.splitext(x)[1] == '.pt'
                   and os.path.splitext(x[11:])[0].isdigit()]
    if len(checkpoints) == 0:
        print("No checkpoints found at " + pathDir)
        return None
    checkpoints.sort(key=lambda x: int(os.path.splitext(x[11:])[0]))
    data = os.path.join(pathDir, checkpoints[-1])
    with open(os.path.join(pathDir, 'checkpoint_logs.json'), 'rb') as file:
        logs = json.load(file)

    with open(os.path.join(pathDir, 'checkpoint_args.json'), 'rb') as file:
        args = json.load(file)

    args = argparse.Namespace(**args)
    defaultArgs = getDefaultConfig()
    loadArgs(defaultArgs, args)

    return os.path.abspath(data), logs, defaultArgs


def getEncoder(args):

    if args.encoder_type == 'mfcc':
        from .model import MFCCEncoder
        return MFCCEncoder(args.hiddenEncoder)
    elif args.encoder_type == 'lfb':
        from .model import LFBEnconder
        return LFBEnconder(args.hiddenEncoder)
    else:
        from .model import CPCEncoder
        return CPCEncoder(args.hiddenEncoder, args.normMode)


def getAR(args):
    if args.arMode == 'transformer':
        arNet = buildTransformerAR(args.hiddenEncoder, 1,
                                   args.sizeWindow // 160, args.abspos)
        args.hiddenGar = args.hiddenEncoder
    else:
        arNet = CPCAR(args.hiddenEncoder, args.hiddenGar,
                      args.samplingType == "sequential",
                      args.nLevelsGRU,
                      mode=args.arMode,
                      reverse=args.cpcMode == "reverse")
    return arNet


def loadModel(pathCheckpoints, loadStateDict=True):
    models = []
    hiddenGar, hiddenEncoder = 0, 0
    for path in pathCheckpoints:
        print(f"Loading checkpoint {path}")
        _, _, locArgs = getCheckpointData(os.path.dirname(path))

        doLoad = locArgs.load is not None and \
            (len(locArgs.load) > 1 or
             os.path.dirname(locArgs.load[0]) != os.path.dirname(path))

        if doLoad:
            m_, hg, he = loadModel(locArgs.load, loadStateDict=False)
            hiddenGar += hg
            hiddenEncoder += he
        else:
            encoderNet = getEncoder(locArgs)

            arNet = getAR(locArgs)
            m_ = CPCModel(encoderNet, arNet)

        if loadStateDict:
            print(f"Loading the state dict at {path}")
            state_dict = torch.load(path, 'cpu')
            m_.load_state_dict(state_dict["gEncoder"], strict=False)
        if not doLoad:
            hiddenGar += locArgs.hiddenGar
            hiddenEncoder += locArgs.hiddenEncoder

        models.append(m_)

    if len(models) == 1:
        return models[0], hiddenGar, hiddenEncoder

    return ConcatenatedModel(models), hiddenGar, hiddenEncoder


class SchedulerCombiner:
    r"""
    An object which applies a list of learning rate schedulers sequentially.
    """

    def __init__(self, scheduler_list, activation_step, curr_step=0):
        r"""
        Args:
            - scheduler_list (list): a list of learning rate schedulers
            - activation_step (list): a list of int. activation_step[i]
            indicates at which step scheduler_list[i] should be activated
            - curr_step (int): the starting step. Must be lower than
            activation_step[0]
        """

        if len(scheduler_list) != len(activation_step):
            raise ValueError("The number of scheduler must be the same as "
                             "the number of activation step")
        if activation_step[0] > curr_step:
            raise ValueError("The first activation step cannot be higher than "
                             "the current step.")
        self.scheduler_list = scheduler_list
        self.activation_step = deepcopy(activation_step)
        self.curr_step = curr_step

    def step(self):
        self.curr_step += 1
        index = bisect_left(self.activation_step, self.curr_step) - 1
        for i in reversed(range(index, len(self.scheduler_list))):
            self.scheduler_list[i].step()

    def __str__(self):
        out = "SchedulerCombiner \n"
        out += "(\n"
        for index, scheduler in enumerate(self.scheduler_list):
            out += f"({index}) {scheduler.__str__()} \n"
        out += ")\n"
        return out


def rampSchedulingFunction(n_epoch_ramp, epoch):
    if epoch >= n_epoch_ramp:
        return 1
    else:
        return (epoch + 1) / n_epoch_ramp
