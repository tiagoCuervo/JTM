try:
    # noinspection PyUnresolvedReferences
    import comet_ml
except ImportError:
    pass
import torch
from dataloader import AudioBatchData
from model import CPCEncoder, CPCModel, CPCUnsupersivedCriterion, loadModel, getAR, CategoryCriterion, TranscriptionCriterion
from trainer import run
from datetime import datetime
import os
import argparse
from default_config import setDefaultConfig, rawAudioPath, metadataPathTrain, metadataPathTest
import sys
import random
from utils import setSeed, getCheckpointData, loadArgs, SchedulerCombiner, rampSchedulingFunction
import json
import pandas as pd
from sklearn.model_selection import train_test_split


def getCriterion(config, downsampling, nClasses=None):
    if not config.supervised:
        cpcCriterion = CPCUnsupersivedCriterion(nPredicts=config.nPredicts,
                                                dimOutputAR=config.hiddenGar,
                                                dimOutputEncoder=config.hiddenEncoder,
                                                negativeSamplingExt=config.negativeSamplingExt,
                                                mode=config.cpcMode,
                                                dropout=config.dropout)
    else:
        if config.task == 'classification':
            cpcCriterion = CategoryCriterion(config.hiddenGar, config.sizeWindow, downsampling, nClasses, pool=(4, 0, 4))
        elif config.task == 'transcription':
            cpcCriterion = TranscriptionCriterion(config.hiddenGar, config.sizeWindow, downsampling, pool=None)
                                                  #pool=(128/(config.transcriptionWindow/10), 0, 128/(config.transcriptionWindow/10)))
        else:
            raise NotImplementedError

    return cpcCriterion


def loadCriterion(pathCheckpoint, downsampling, nClasses):
    _, _, locArgs = getCheckpointData(os.path.dirname(pathCheckpoint))
    criterion = getCriterion(locArgs, downsampling, nClasses)

    state_dict = torch.load(pathCheckpoint, 'cpu')

    criterion.load_state_dict(state_dict["cpcCriterion"])
    return criterion


def parseArgs(argv):
    # Run parameters
    parser = argparse.ArgumentParser(description='Trainer')

    # Default arguments:
    parser = setDefaultConfig(parser)

    groupDb = parser.add_argument_group('Dataset')
    groupDb.add_argument('--ignoreCache', action='store_true',
                         help='Activate if the dataset has been modified '
                              'since the last training session.')
    groupDb.add_argument('--chunkSize', type=int, default=1e9,
                         help='Size (in bytes) of a data chunk')
    groupDb.add_argument('--maxChunksInMem', type=int, default=2,
                         help='Maximal amount of data chunks a dataset '
                              'can hold in memory at any given time')
    groupDb.add_argument('--labelsBy', type=str, default='id',
                         help="What attribute of the data set to use as labels. Only important if 'samplingType' "
                              "is 'samecategory'")

    group_supervised = parser.add_argument_group('Supervised mode')
    group_supervised.add_argument('--supervised', action='store_true',
                                  help='Disable the CPC loss and activate '
                                  'the supervised mode. By default, the supervised '
                                  'training method is the ensemble classification.')
    group_supervised.add_argument('--task', type=str, default='transcription',
                                   help='Type of the donwstream task if in supeprvised mode. '
                                        'Currently supported tasks are classification and transcription.')
    group_supervised.add_argument('--transcriptionWindow', type=int, default=160,
                                   help='Size of the transcription window (in ms) in the transcription downstream task.')

    groupSave = parser.add_argument_group('Save')
    groupSave.add_argument('--pathCheckpoint', type=str, default=None,
                           help="Path of the output directory.")
    groupSave.add_argument('--loggingStep', type=int, default=1000)
    groupSave.add_argument('--saveStep', type=int, default=1,
                           help="Frequency (in epochs) at which a checkpoint "
                                "should be saved")
    groupSave.add_argument('--log2Board', type=int, default=1,
                           help="Defines what (if any) data to log to Comet.ml:\n"
                                "\t0 : do not log to Comet\n\t1 : log losses and accuracy\n\t>1 : log histograms of "
                                "weights and gradients.\nFor log2Board > 0 you will need to provide Comet.ml "
                                "credentials.")

    groupLoad = parser.add_argument_group('Load')
    groupLoad.add_argument('--load', type=str, default=None, nargs='*',
                           help="Load an existing checkpoint. Should give a path "
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

    if config.load is not None:
        config.load = [os.path.abspath(x) for x in config.load]

    # Set it up if needed, so that it is dumped along with other args
    if config.randomSeed is None:
        config.randomSeed = random.randint(0, 2 ** 31)

    return config


def main(config):
    if not (os.path.exists(rawAudioPath) and os.path.exists(metadataPathTrain) and os.path.exists(metadataPathTest)):
        print("The audio data and csv metadata must be located in the following paths:\n"
              f"1. {rawAudioPath}\n2. {metadataPathTrain}\n3. {metadataPathTest}")
        sys.exit()

    config = parseArgs(config)
    setSeed(config.randomSeed)
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

    if not os.path.exists(f'data/musicnet_metadata_train_{config.labelsBy}_trainsplit.csv'):
        if config.transcriptionWindow is not None:
           musicNetMetadataTrain = pd.read_csv('data/musicnet_metadata_transcript_train.csv')
        else:
           musicNetMetadataTrain = pd.read_csv('data/musicnet_metadata_train.csv', index = 'id')
        try:
            metadataTrain, metadataVal = train_test_split(musicNetMetadataTrain, test_size=0.1,
                                                          stratify=musicNetMetadataTrain[config.labelsBy])
        except ValueError:
            for col, count in zip(musicNetMetadataTrain[config.labelsBy].value_counts().index,
                                  musicNetMetadataTrain[config.labelsBy].value_counts().values):
                if count == 1:
                    subDF = musicNetMetadataTrain.loc[musicNetMetadataTrain[config.labelsBy] == col]
                    musicNetMetadataTrain = musicNetMetadataTrain.append(subDF)
            metadataTrain, metadataVal = train_test_split(musicNetMetadataTrain, test_size=0.1,
                                                          stratify=musicNetMetadataTrain[config.labelsBy])
        metadataTrain.to_csv(f'data/musicnet_metadata_train_{config.labelsBy}_trainsplit.csv')
        metadataVal.to_csv(f'data/musicnet_metadata_train_{config.labelsBy}_valsplit.csv')
    else:
        if config.transcriptionWindow is not None:
           metadataTrain = pd.read_csv(f'data/musicnet_metadata_train_{config.labelsBy}_trainsplit.csv')
           metadataVal = pd.read_csv(f'data/musicnet_metadata_train_{config.labelsBy}_valsplit.csv')
        else:
           metadataTrain = pd.read_csv(f'data/musicnet_metadata_train_{config.labelsBy}_trainsplit.csv', index = 'id')
           metadataVal = pd.read_csv(f'data/musicnet_metadata_train_{config.labelsBy}_valsplit.csv', index = 'id')

    print("Loading the training dataset")
    trainDataset = AudioBatchData(rawAudioPath=rawAudioPath,
                                  metadata=metadataTrain,
                                  sizeWindow=config.sizeWindow,
                                  labelsBy=config.labelsBy,
                                  outputPath='data/musicnet_lousy/train_data/train',
                                  CHUNK_SIZE=config.chunkSize,
                                  NUM_CHUNKS_INMEM=config.maxChunksInMem,
                                  useGPU=useGPU,
                                  transcript_window=config.transcriptionWindow)
    print("Training dataset loaded")
    print("")

    print("Loading the validation dataset")
    valDataset = AudioBatchData(rawAudioPath=rawAudioPath,
                                metadata=metadataVal,
                                sizeWindow=config.sizeWindow,
                                labelsBy=config.labelsBy,
                                outputPath='data/musicnet_lousy/train_data/val',
                                CHUNK_SIZE=config.chunkSize,
                                NUM_CHUNKS_INMEM=config.maxChunksInMem,
                                useGPU=False,
                                transcript_window=config.transcriptionWindow)
    print("Validation dataset loaded")
    print("")

    if config.load is not None:
        cpcModel, config.hiddenGar, config.hiddenEncoder = loadModel(config.load, config)
    else:
        # Encoder network
        encoderNet = CPCEncoder(config.hiddenEncoder, 'layerNorm', sincNet=config.encoderType == 'sinc')
        # AR Network
        arNet = getAR(config)

        cpcModel = CPCModel(encoderNet, arNet)

    batchSize = config.batchSize
    cpcModel.supervised = config.supervised

    # Training criterion
    if config.load is not None and config.loadCriterion:
        cpcCriterion = loadCriterion(config.load[0], cpcModel.gEncoder.DOWNSAMPLING,
                                     len(metadataTrain[config.labelsBy].unique()))
    else:
        cpcCriterion = getCriterion(config, cpcModel.gEncoder.DOWNSAMPLING,
                                    len(metadataTrain[config.labelsBy].unique())) # change for transcription labels

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
        comet_ml.init(project_name="jtm", workspace="tiagocuervo")
        if not os.path.exists('.comet.config'):
            cometKey = input("Please enter your Comet.ml API key: ")
            experiment = comet_ml.Experiment(cometKey)
            cometConfigFile = open(".comet.config", "w")
            cometConfigFile.write(f"[comet]\napi_key={cometKey}")
            cometConfigFile.close()
        else:
            experiment = comet_ml.Experiment()
        experiment.log_parameters(vars(config))

    run(trainDataset, valDataset, batchSize, config.samplingType, cpcModel, cpcCriterion, config.nEpoch, optimizer,
        scheduler, pathCheckpoint, logs, useGPU, log2Board=config.log2Board, experiment=experiment)


if __name__ == "__main__":
    args = sys.argv[1:]
    main(args)
