import torch
from dataloader import AudioBatchData
from model import CPCEncoder, CPCAR, CPCModel, CPCUnsupersivedCriterion
from trainer import run
from datetime import datetime
import os

if __name__ == '__main__':
    labelsBy = 'ensemble'
    print("Loading the training dataset")
    trainDataset = AudioBatchData(rawAudioPath='data/musicnet_lousy/train_data',
                                  metadataPath='data/musicnet_lousy/metadata_train.csv',
                                  sizeWindow=20480,
                                  labelsBy=labelsBy,
                                  outputPath='data/musicnet_lousy/train_data/train',
                                  CHUNK_SIZE=1e9,
                                  NUM_CHUNKS_INMEM=7)
    print("Training dataset loaded")
    print("")

    print("Loading the validation dataset")
    valDataset = AudioBatchData(rawAudioPath='data/musicnet_lousy/train_data',
                                metadataPath='data/musicnet_lousy/metadata_val.csv',
                                sizeWindow=20480,
                                labelsBy=labelsBy,
                                outputPath='data/musicnet_lousy/train_data/val',
                                CHUNK_SIZE=1e9,
                                NUM_CHUNKS_INMEM=1)
    print("Validation dataset loaded")
    print("")

    samplingType = 'samesequence'

    # Encoder network
    encoderNet = CPCEncoder(512, 'layerNorm')
    # AR Network
    arNet = CPCAR(512, 256, samplingType == 'sequential', 1, mode="GRU", reverse=False)

    cpcModel = CPCModel(encoderNet, arNet)
    batchSize = 8
    cpcModel.supervised = False

    cpcCriterion = CPCUnsupersivedCriterion(nPredicts=12,
                                            dimOutputAR=256,
                                            dimOutputEncoder=512,
                                            negativeSamplingExt=128,
                                            mode=None,
                                            dropout=False)
    useGPU = torch.cuda.is_available()

    if useGPU:
        cpcCriterion.cuda()
        cpcModel.cuda()

    gParams = list(cpcCriterion.parameters()) + list(cpcModel.parameters())
    lr = 2e-4
    optimizer = torch.optim.Adam(gParams, lr=lr, betas=(0.9, 0.999), eps=1e-8)

    expDescription = f'{samplingType}_'
    if samplingType == 'samecategory':
        expDescription += f'{labelsBy}_'

    pathCheckpoint = f'logs/{expDescription}{datetime.now().strftime("%d-%m_%H-%M-%S")}'
    if not os.path.isdir(pathCheckpoint):
        os.mkdir(pathCheckpoint)
    pathCheckpoint = os.path.join(pathCheckpoint, "checkpoint")

    logs = {"epoch": [], "iter": [], "saveStep": 1, "logging_step": 1000}
    run(trainDataset, valDataset, batchSize, samplingType, cpcModel, cpcCriterion, 30, optimizer, pathCheckpoint, logs,
        useGPU)
