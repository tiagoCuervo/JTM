import torch
from dataloader import AudioBatchData
from model import CPCEncoder, CPCAR, CPCModel, CPCUnsupersivedCriterion
from trainer import run
from datetime import datetime
import os
import argparse
from default_config import setDefaultConfig
import sys
import random
from utils import setSeed, getCheckpointData, loadArgs, loadModel, getAR, SchedulerCombiner, rampSchedulingFunction
import json

rawAudioPath = 'data/musicnet_lousy/train_data'
metadataPathTrain = 'data/musicnet_lousy/metadata_train.csv'
metadataPathVal = 'data/musicnet_lousy/metadata_train.csv'


def getCriterion(config):
    if not config.supervised:
        cpcCriterion = CPCUnsupersivedCriterion(nPredicts=config.nPredicts,
                                                dimOutputAR=config.hiddenGAr,
                                                dimOutputEncoder=config.hiddenEncoder,
                                                negativeSamplingExt=config.negativeSamplingExt,
                                                mode=config.cpcMode,
                                                dropout=config.dropout)
    else:
        raise NotImplementedError
    return cpcCriterion


def loadCriterion(pathCheckpoint):
    _, _, locArgs = getCheckpointData(os.path.dirname(pathCheckpoint))
    criterion = getCriterion(locArgs)

    state_dict = torch.load(pathCheckpoint, 'cpu')

    criterion.load_state_dict(state_dict["cpcCriterion"])
    return criterion


def parseArgs(argv):
    # Run parameters
    parser = argparse.ArgumentParser(description='Trainer')

    # Default arguments:
    parser = setDefaultConfig(parser)

    groupDb = parser.add_argument_group('Dataset')
    # groupDb.add_argument('--audioPath', type=str, default=None,
    #                      help='Path to the directory containing the '
    #                           'raw audio data.')
    # groupDb.add_argument('--metadataPathTrain', type=str, default=None,
    #                      help='Path to the directory containing the '
    #                           'meta data for the train set.')
    # groupDb.add_argument('--metadataPathVal', type=str, default=None,
    #                      help='Path to the directory containing the '
    #                           'meta data for the validation set.')
    # groupDb.add_argument('--fileExtension', type=str, default=".wav",
    #                      help="Extension of the audio files in the dataset.")
    groupDb.add_argument('--ignoreCache', action='store_true',
                         help='Activate if the dataset has been modified '
                              'since the last training session.')
    groupDb.add_argument('--chunkSize', type=int, default=1e9,
                         help='Size (in bytes) of a data chunk')
    groupDb.add_argument('--maxChunksInMem', type=int, default=7,
                         help='Maximal amount of data chunks a dataset '
                              'can hold in memory at any given time')
    groupDb.add_argument('--labelsBy', type=str, default='ensemble',
                         help="What attribute of the data set to use as labels. Only important if 'samplingType' "
                              "is 'samecategory'")
    group_supervised = parser.add_argument_group(
        'Supervised mode')
    group_supervised.add_argument('--supervised', action='store_true',
                                  help='Disable the CPC loss and activate '
                                  'the supervised mode. By default, the supervised '
                                  'training method is the ensemble classification.')
    # group_supervised.add_argument('--pathPhone', type=str, default=None,
    #                               help='(Supervised mode only) Path to a .txt '
    #                               'containing the phone labels of the dataset. If given '
    #                               'and --supervised, will train the model using a '
    #                               'phone classification task.')
    # group_supervised.add_argument('--CTC', action='store_true')

    groupSave = parser.add_argument_group('Save')
    # groupSave.add_argument('--pathCheckpoint', type=str, default=None,
    #                        help="Path of the output directory.")
    groupSave.add_argument('--loggingStep', type=int, default=1000)
    groupSave.add_argument('--saveStep', type=int, default=1,
                           help="Frequency (in epochs) at which a checkpoint "
                                "should be saved")
    groupSave.add_argument('--log2Board', type=int, default=2,
                           help="0 : do not log to Comet\n1 : log only losses\n>1 : log histograms of weights"
                                "and gradients.\nFor log2Board > 0 you will need to provide Comet.ml credentials. ")
    groupLoad = parser.add_argument_group('Load')
    groupLoad.add_argument('--load', type=str, default=None, nargs='*',
                           help="Load an exsiting checkpoint. Should give a path "
                                "to a .pt file. The directory containing the file to "
                                "load should also have a 'checkpoint.logs' and a "
                                "'checkpoint.args'")
    groupLoad.add_argument('--loadCriterion', action='store_true',
                           help="If --load is activated, load the state of the "
                                "training criterion as well as the state of the "
                                "feature network (encoder + AR)")
    # groupLoad.add_argument('--restart', action='store_true',
    #                        help="If any checkpoint is found, ignore it and "
    #                             "restart the training from scratch.")

    # group_gpu = parser.add_argument_group('GPUs')
    # group_gpu.add_argument('--nGPU', type=int, default=-1,
    #                        help="Number of GPU to use (default: use all "
    #                        "available GPUs)")
    # group_gpu.add_argument('--batchSizeGPU', type=int, default=8, help='Number of batches per GPU.')
    # parser.add_argument('--debug', action='store_true', help="Load only a very small amount of files for "
    #                                                          "debugging purposes.")
    config = parser.parse_args(argv)

    if not (os.path.exists(rawAudioPath) and os.path.exists(metadataPathTrain) and os.path.exists(metadataPathVal)):
        print("Please make sure the following paths exist:\n"
              f"1. {rawAudioPath}\n2. {metadataPathTrain}\n3. {metadataPathVal}")
        sys.exit()

    if config.pathCheckpoint is not None:
        config.pathCheckpoint = os.path.abspath(config.pathCheckpoint)

    if config.load is not None:
        config.load = [os.path.abspath(x) for x in config.load]

    # Set it up if needed, so that it is dumped along with other args
    if config.randomSeed is None:
        config.randomSeed = random.randint(0, 2 ** 31)

    return config


def main(config):
    config = parseArgs(config)
    setSeed(config.random_seed)
    logs = {"epoch": [], "iter": [], "saveStep": config.saveStep, "loggingStep": config.loggingStep}

    loadOptimizer = False
    if config.pathCheckpoint is not None:
        cdata = getCheckpointData(config.pathCheckpoint)
        if cdata is not None:
            data, logs, locArgs = cdata
            print(f"Checkpoint detected at {data}")
            loadArgs(config, locArgs, forbiddenAttr={"nGPU", "pathCheckpoint", "maxChunksInMem", "chunkSize"})
            config.load, loadOptimizer = [data], True
            config.loadCriterion = True

    print(f'CONFIG: \n{json.dumps(vars(config), indent=4, sort_keys=True)}')
    print('-' * 50)

    useGPU = torch.cuda.is_available()
    print("Loading the training dataset")
    trainDataset = AudioBatchData(rawAudioPath='data/musicnet_lousy/train_data',
                                  metadataPath='data/musicnet_lousy/metadata_train.csv',
                                  sizeWindow=config.sizeWindow,
                                  labelsBy=config.labelsBy,
                                  outputPath='data/musicnet_lousy/train_data/train',
                                  CHUNK_SIZE=config.chunkSize,
                                  NUM_CHUNKS_INMEM=config.maxChunksInMem,
                                  useGPU=useGPU)
    print("Training dataset loaded")
    print("")

    print("Loading the validation dataset")
    valDataset = AudioBatchData(rawAudioPath='data/musicnet_lousy/train_data',
                                metadataPath='data/musicnet_lousy/metadata_val.csv',
                                sizeWindow=config.sizeWindow,
                                labelsBy=config.labelsBy,
                                outputPath='data/musicnet_lousy/train_data/val',
                                CHUNK_SIZE=config.chunkSize,
                                NUM_CHUNKS_INMEM=config.maxChunksInMem,
                                useGPU=False)
    print("Validation dataset loaded")
    print("")

    if config.load is not None:
        cpcModel, config.hiddenGar, config.hiddenEncoder = loadModel(config.load)

    else:
        # Encoder network
        encoderNet = CPCEncoder(512, 'layerNorm', sincNet=config.sincNetEncoder)
        # AR Network
        arNet = getAR(config)

        cpcModel = CPCModel(encoderNet, arNet)

    batchSize = config.batchSize
    cpcModel.supervised = config.supervised

    # Training criterion
    if config.load is not None and config.loadCriterion:
        cpcCriterion = loadCriterion(config.load[0])
    else:
        cpcCriterion = getCriterion(config)

    if loadOptimizer:
        stateDict = torch.load(config.load[0], 'cpu')
        cpcCriterion.load_state_dict(stateDict["cpcCriterion"])

    if useGPU:
        cpcCriterion.cuda()
        cpcModel.cuda()

    # Optimizer
    gParams = list(cpcCriterion.parameters()) + list(cpcModel.parameters())
    lr = config.learningRate
    optimizer = torch.optim.Adam(gParams, lr=lr, betas=(config.beta1, config.beta2), eps=config.epsilon)

    if loadOptimizer:
        print("Loading optimizer " + config.load[0])
        state_dict = torch.load(config.load[0], 'cpu')
        if "optimizer" in state_dict:
            optimizer.load_state_dict(state_dict["optimizer"])

    # Checkpoint
    expDescription = f'{config.samplingType}_'
    if config.samplingType == 'samecategory':
        expDescription += f'{config.labelsBy}_'

    pathCheckpoint = f'logs/{expDescription}{datetime.now().strftime("%d-%m_%H-%M-%S")}'
    os.makedirs(pathCheckpoint, exist_ok=True)
    pathCheckpoint = os.path.join(pathCheckpoint, "checkpoint")

    scheduler = None
    if config.schedulerStep > 0:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                    config.schedulerStep,
                                                    gamma=0.5)
    if config.schedulerRamp is not None:
        n_epoch = config.schedulerRamp
        print(f"Ramp activated. n_e = {n_epoch}")
        scheduler_ramp = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                           lr_lambda=lambda epoch: rampSchedulingFunction(n_epoch,
                                                                                                          epoch),
                                                           last_epoch=-1)
        if scheduler is None:
            scheduler = scheduler_ramp
        else:
            scheduler = SchedulerCombiner([scheduler_ramp, scheduler], [0, config.schedulerRamp])
    if scheduler is not None:
        for i in range(len(logs["epoch"])):
            scheduler.step()

    experiment = None
    if config.log2Board:
        import comet_ml
        comet_ml.init(project_name="jtm", workspace="tiagocuervo")
        experiment = comet_ml.Experiment()

    run(trainDataset, valDataset, batchSize, config.samplingType, cpcModel, cpcCriterion, config.nEpoch, optimizer,
        pathCheckpoint, logs, useGPU, log2Board=config.log2Board, experiment=experiment)


if __name__ == "__main__":
    args = sys.argv[1:]
    main(args)
