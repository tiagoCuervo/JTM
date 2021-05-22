from pathlib import Path
import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler, BatchSampler
import librosa
import tqdm
import random
import time
import pickle
import re


class AudioBatchData(Dataset):

    def __init__(self,
                 rawAudioPath,
                 metadataPath,
                 sizeWindow,
                 labelsBy='composer',
                 outputPath=None,
                 CHUNK_SIZE=1e9,
                 NUM_CHUNKS_INMEM=2):
        """
        Args:
            - rawAudioPath (string): path to the raw audio files
            - metadataPath (string): path to the data set metadata (used to define labels)
            - sizeWindow (int): size of the sliding window
            - labelsBy (string): name of column in metadata according to which create labels
            - outputPath (string): path to the directory where chunks are to be created or are stored
            - CHUNK_SIZE (int): desired size in bytes of a chunk
            - NUM_CHUNKS_INMEM (int): target maximal size chunks of data to load in memory at a time
        """
        self.NUM_CHUNKS_INMEM = NUM_CHUNKS_INMEM
        self.CHUNK_SIZE = CHUNK_SIZE
        self.rawAudioPath = Path(rawAudioPath)
        self.sizeWindow = sizeWindow

        self.sequencesData = pd.read_csv(metadataPath, index_col='id')
        self.sequencesData = self.sequencesData.sort_values(by=labelsBy)
        self.sequencesData[labelsBy] = self.sequencesData[labelsBy].astype('category')
        self.sequencesData[labelsBy] = self.sequencesData[labelsBy].cat.codes

        self.labels = list(range(len(self.sequencesData[labelsBy].values)))
        self.category = labelsBy

        if outputPath is None:
            self.chunksDir = self.rawAudioPath / labelsBy
        else:
            self.chunksDir = Path(outputPath) / labelsBy

        if not os.path.exists(self.chunksDir):
            os.makedirs(self.chunksDir)

        packages2Load = [fileName for fileName in os.listdir(self.chunksDir) if
                         re.match(r'chunk_.*[0-9]+.pickle', fileName)]

        if len(packages2Load) == 0:
            self._createChunks()
            packages2Load = [fileName for fileName in os.listdir(self.chunksDir) if
                             re.match(r'chunk_.*[0-9]+.pickle', fileName)]
        else:
            print("Chunks already exist at", self.chunksDir)

        self.packs = []
        packOfChunks = []
        for i, packagePath in enumerate(packages2Load):
            packOfChunks.append(packagePath)
            if (i + 1) % self.NUM_CHUNKS_INMEM == 0:
                self.packs.append(packOfChunks)
                packOfChunks = []
        if len(packOfChunks) > 0:
            self.packs.append(packOfChunks)

        self.currentPack = -1
        self.nextPack = 0
        self.sequenceIdx = 0

        self.data = None

        self._loadNextPack(first=True)
        self._loadNextPack()

    def _createChunks(self):
        print("Creating chunks at", self.chunksDir)
        pack = []
        packIds = []
        packageSize = 0
        packageIdx = 0
        for trackId in tqdm.tqdm(self.sequencesData.index):
            sequence, samplingRate = librosa.load(self.rawAudioPath / (str(trackId) + '.wav'), sr=16000)
            sequence = torch.tensor(sequence).float()
            packIds.append(trackId)
            pack.append(sequence)
            packageSize += len(sequence) * 4
            if packageSize >= self.CHUNK_SIZE:
                print(f"Saved pack {packageIdx}")
                with open(self.chunksDir / f'chunk_{packageIdx}.pickle', 'wb') as handle:
                    pickle.dump(torch.cat(pack, dim=0), handle, protocol=pickle.HIGHEST_PROTOCOL)
                with open(self.chunksDir / f'ids_{packageIdx}.pickle', 'wb') as handle:
                    pickle.dump(packIds, handle, protocol=pickle.HIGHEST_PROTOCOL)
                pack = []
                packIds = []
                packageSize = 0
                packageIdx += 1
        print(f"Saved pack {packageIdx}")
        with open(self.chunksDir / f'chunk_{packageIdx}.pickle', 'wb') as handle:
            pickle.dump(torch.cat(pack, dim=0), handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(self.chunksDir / f'ids_{packageIdx}.pickle', 'wb') as handle:
            pickle.dump(packIds, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def _loadNextPack(self, first=False):
        self.clear()
        if not first:
            self.currentPack = self.nextPack
            startTime = time.time()
            print('Loading files')
            self.categoryLabel = [0]
            packageIdx = [0]
            self.seqLabel = [0]
            packageSize = 0
            previousCategory = 0
            for packagePath in self.packs[self.currentPack]:
                with open(self.chunksDir / ('ids_' + packagePath.split('_', maxsplit=1)[-1]), 'rb') as handle:
                    chunkIds = pickle.load(handle)
                for seqId in chunkIds:
                    currentCategory = self.sequencesData.loc[seqId][self.category]
                    if currentCategory != previousCategory:
                        self.categoryLabel.append(packageSize)
                    previousCategory = currentCategory
                    packageSize += self.sequencesData.loc[seqId].length
                    self.seqLabel.append(packageSize)
                packageIdx.append(packageSize)

            self.data = torch.empty(size=(packageSize,))
            for i, packagePath in enumerate(self.packs[self.currentPack]):
                with open(self.chunksDir / packagePath, 'rb') as handle:
                    self.data[packageIdx[i]:packageIdx[i + 1]] = pickle.load(handle)
            self.totSize = len(self.data)
            print(f'Loaded {len(self.seqLabel) - 1} sequences, elapsed={time.time() - startTime:.3f} secs')

        self.nextPack = (self.currentPack + 1) % len(self.packs)
        if self.nextPack == 0 and len(self.packs) > 1:
            self.currentPack = -1
            self.nextPack = 0
            self.sequenceIdx = 0

    def clear(self):
        if 'data' in self.__dict__:
            del self.data
        if 'categoryLabel' in self.__dict__:
            del self.categoryLabel
        if 'seqLabel' in self.__dict__:
            del self.seqLabel

    def getCategoryLabel(self, idx):
        idCategory = next(x[0] for x in enumerate(self.categoryLabel) if x[1] > idx) - 1
        return idCategory

    def getSequenceLabel(self, idx):
        return self.categoryLabel[idx]

    def __len__(self):
        return self.totSize // self.sizeWindow

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self.data) - self.sizeWindow - 1:
            print(idx)

        outData = self.data[idx:(self.sizeWindow + idx)].view(1, -1)
        label = torch.tensor(self.getCategoryLabel(idx), dtype=torch.long)
        return outData, label

    def getBaseSampler(self, samplingType, batchSize, offset):
        if samplingType == "samecategory":
            return SameTrackSampler(batchSize, self.categoryLabel, self.sizeWindow, offset)
        if samplingType == "samesequence":
            return SameTrackSampler(batchSize, self.seqLabel, self.sizeWindow, offset)
        if samplingType == "sequential":
            return SequentialSampler(len(self.data), self.sizeWindow, offset, batchSize)

        sampler = UniformAudioSampler(len(self.data), self.sizeWindow, offset)
        return BatchSampler(sampler, batchSize, True)

    def getDataLoader(self, batchSize, samplingType, randomOffset, numWorkers=0,
                      onLoop=-1):
        r"""
        Get a batch sampler for the current dataset.
            - batchSize (int): batch size
            - groupSize (int): in the case of type in ["track", "sequence"]
            number of items sharing a same label in the group
            (see AudioBatchSampler)
            - type (string):
                type == "track": grouped sampler track-wise
                type == "sequence": grouped sampler sequence-wise
                type == "sequential": sequential sampling
                else: uniform random sampling of the full audio
                vector
            - randomOffset (bool): if True add a random offset to the sampler
                                   at the begining of each iteration
        """
        nLoops = len(self.packs)
        totSize = self.totSize // (self.sizeWindow * batchSize)
        if onLoop >= 0:
            self.currentPack = onLoop - 1
            self._loadNextPack()
            nLoops = 1

        def samplerCall():
            offset = random.randint(0, self.sizeWindow // 2) \
                if randomOffset else 0
            return self.getBaseSampler(samplingType, batchSize, offset)

        return AudioLoader(self, samplerCall, nLoops, self._loadNextPack, totSize, numWorkers)


class AudioLoader(object):
    r"""
    A DataLoader meant to handle an AudioBatchData object.
    In order to handle big datasets AudioBatchData works with big chunks of
    audio it loads sequentially in memory: once all batches have been sampled
    on a chunk, the AudioBatchData loads the next one.
    """

    def __init__(self,
                 dataset,
                 samplerCall,
                 nLoop,
                 updateCall,
                 size,
                 numWorkers):
        r"""
        Args:
            - dataset (AudioBatchData): target dataset
            - samplerCall (function): batch-sampler to call
            - nLoop (int): number of chunks to load
            - updateCall (function): function loading the next chunk
            - size (int): total number of batches
            - numWorkers (int): see torch.utils.data.DataLoader
        """
        self.samplerCall = samplerCall
        self.updateCall = updateCall
        self.nLoop = nLoop
        self.size = size
        self.dataset = dataset
        self.numWorkers = numWorkers

    def __len__(self):
        return self.size

    def __iter__(self):

        for i in range(self.nLoop):
            sampler = self.samplerCall()
            dataloader = DataLoader(self.dataset,
                                    batch_sampler=sampler,
                                    num_workers=self.numWorkers)
            for x in dataloader:
                yield x
            if i < self.nLoop - 1:
                self.updateCall()


class UniformAudioSampler(Sampler):

    def __init__(self,
                 dataSize,
                 sizeWindow,
                 offset):
        self.len = dataSize // sizeWindow
        self.sizeWindow = sizeWindow
        self.offset = offset
        if self.offset > 0:
            self.len -= 1

    def __iter__(self):
        return iter((self.offset
                     + self.sizeWindow * torch.randperm(self.len)).tolist())

    def __len__(self):
        return self.len


class SequentialSampler(Sampler):

    def __init__(self, dataSize, sizeWindow, offset, batchSize):

        self.len = (dataSize // sizeWindow) // batchSize
        self.sizeWindow = sizeWindow
        self.offset = offset
        self.startBatches = [x * (dataSize // batchSize)
                             for x in range(batchSize)]
        self.batchSize = batchSize
        if self.offset > 0:
            self.len -= 1

    def __iter__(self):
        for idx in range(self.len):
            yield [self.offset + self.sizeWindow * idx
                   + start for start in self.startBatches]

    def __len__(self):
        return self.len


class SameTrackSampler(Sampler):

    def __init__(self,
                 batchSize,
                 samplingIntervals,
                 sizeWindow,
                 offset):

        self.samplingIntervals = samplingIntervals
        self.sizeWindow = sizeWindow
        self.batchSize = batchSize
        self.offset = offset

        if self.samplingIntervals[0] != 0:
            raise AttributeError("Sampling intervals should start at zero")

        nWindows = len(self.samplingIntervals) - 1
        self.sizeSamplers = [(self.samplingIntervals[i + 1] -
                              self.samplingIntervals[i]) // self.sizeWindow
                             for i in range(nWindows)]

        if self.offset > 0:
            self.sizeSamplers = [max(0, x - 1) for x in self.sizeSamplers]

        order = [(x, torch.randperm(val).tolist())
                 for x, val in enumerate(self.sizeSamplers) if val > 0]

        # Build Batches
        self.batches = []
        for indexSampler, randperm in order:
            indexStart, sizeSampler = 0, self.sizeSamplers[indexSampler]
            while indexStart < sizeSampler:
                indexEnd = min(sizeSampler, indexStart + self.batchSize)
                locBatch = [self.getIndex(x, indexSampler)
                            for x in randperm[indexStart:indexEnd]]
                indexStart = indexEnd
                self.batches.append(locBatch)

    def __len__(self):
        return len(self.batches)

    def getIndex(self, x, iInterval):
        return self.offset + x * self.sizeWindow \
               + self.samplingIntervals[iInterval]

    def __iter__(self):
        random.shuffle(self.batches)
        return iter(self.batches)
