import torch
from dataloader import AudioBatchData
from model import CPCEncoder, CPCAR, CPCModel, CPCUnsupersivedCriterion
from trainer import run

if __name__ == '__main__':
    print("Loading the training dataset")
    trainDataset = AudioBatchData(rawAudioPath='data/musicnet_lousy/train_data',
                                  metadataPath='data/musicnet_lousy/metadata_train.csv',
                                  sizeWindow=20480,
                                  labelsBy='ensemble',
                                  outputPath='data/musicnet_lousy/train_data/train',
                                  CHUNK_SIZE=1e9,
                                  NUM_CHUNKS_INMEM=7)
    print("Training dataset loaded")
    print("")

    print("Loading the validation dataset")
    valDataset = AudioBatchData(rawAudioPath='data/musicnet_lousy/train_data',
                                metadataPath='data/musicnet_lousy/metadata_val.csv',
                                sizeWindow=20480,
                                labelsBy='ensemble',
                                outputPath='data/musicnet_lousy/train_data/val',
                                CHUNK_SIZE=1e9,
                                NUM_CHUNKS_INMEM=1)
    print("Validation dataset loaded")
    print("")

    # Encoder network
    encoderNet = CPCEncoder(512, 'layerNorm')
    # AR Network
    arNet = CPCAR(512, 256, False, 1, mode="GRU", reverse=False)

    cpcModel = CPCModel(encoderNet, arNet)
    batchSize = 8
    cpcModel.supervised = False

    sizeInputSeq = (20480 // 160)
    cpcCriterion = CPCUnsupersivedCriterion(nPredicts=12,
                                            dimOutputAR=256,
                                            dimOutputEncoder=512,
                                            negativeSamplingExt=128,
                                            mode=None,
                                            dropout=False)
    cpcCriterion.cuda()
    cpcModel.cuda()
    gParams = list(cpcCriterion.parameters()) + list(cpcModel.parameters())
    lr = 2e-4
    optimizer = torch.optim.Adam(gParams, lr=lr, betas=(0.9, 0.999), eps=1e-8)

    logs = {"epoch": [], "iter": [], "saveStep": 10, "logging_step": 1000}
    run(trainDataset, valDataset, batchSize, 'samecategory', cpcModel, cpcCriterion, 5, optimizer, logs)
