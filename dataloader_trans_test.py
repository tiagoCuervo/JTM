from pathlib import Path
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler, BatchSampler
import librosa
import tqdm
import random
import time
import pickle
import re
import numpy as np
import pandas as pd

# change later
from default_config import rawLabelsPath

class AudioBatchData(Dataset):

    def __init__(self,
                 rawAudioPath,
                 metadata,
                 sizeWindow,
                 rawLabelsPath=rawLabelsPath,
                 labelsBy='composer',
                 ensemble='Solo Piano',
                 outputPath=None,
                 CHUNK_SIZE=1e9,
                 NUM_CHUNKS_INMEM=2,
                 useGPU=False,
                 transcript_window=None):
        """
        Args:
            - rawAudioPath (string): path to the raw audio files
            - metadataPath (string): path to the data set metadata (used to define labels)
            - sizeWindow (int): size of the sliding window
            - labelsBy (string): name of column in metadata according to which create labels
            - outputPath (string): path to the directory where chunks are to be created or are stored
            - CHUNK_SIZE (int): desired size in bytes of a chunk
            - NUM_CHUNKS_INMEM (int): target maximal size chunks of data to load in memory at a time
            - transcript_window (int): size of the encoded window for transcription
        """
        self.NUM_CHUNKS_INMEM = NUM_CHUNKS_INMEM
        self.CHUNK_SIZE = CHUNK_SIZE
        self.rawAudioPath = Path(rawAudioPath)
        self.rawLabelsPath = Path(rawLabelsPath)
        self.sizeWindow = sizeWindow
        self.useGPU = useGPU
        self.transcript_window = (transcript_window) / 1000 * 16000
        # self.sequencesData = pd.read_csv(metadataPath, index_col='id')
        self.sequencesData = metadata.sort_values(by=labelsBy)
        # self.sequencesData[labelsBy] = self.sequencesData[labelsBy].astype('category')
        # self.sequencesData[labelsBy] = self.sequencesData[labelsBy].cat.codes

        self.totSize = self.sequencesData['length'].sum()

        self.filter_transcription = ensemble
        self.category = labelsBy
        self.labels = self.sequencesData[labelsBy + '_cat'].unique()

        if outputPath is None:
            self.chunksDir = self.rawAudioPath / labelsBy
        else:
            self.chunksDir = Path(outputPath) / labelsBy

        if not os.path.exists(self.chunksDir):
            os.makedirs(self.chunksDir)

        packages2Load = [fileName for fileName in os.listdir(self.chunksDir) if
                         re.match(r'chunk_.*[0-9]+.pickle', fileName)]

        transcripts2Load = [fileName for fileName in os.listdir(self.chunksDir) if
                                    re.match(r'transcript_.*[0-9]+.pickle', fileName)]
        if len(packages2Load) == 0:
            self._createChunks()
            packages2Load = [fileName for fileName in os.listdir(self.chunksDir) if
                             re.match(r'chunk_.*[0-9]+.pickle', fileName)]
            if self.transcript_window is not None:
                transcripts2Load = [fileName for fileName in os.listdir(self.chunksDir) if
                                    re.match(r'transcript_.*[0-9]+.pickle', fileName)]

            ##########################################################
            ## NEED TO LOAD THE CREATED TRANSCRIPTIONS HERE AS WELL ##
            # NOT SURE IF STRAIGHT COPYING THE METHOD FOR DATA WORKS #
            ##########################################################

        else:
            print("Chunks already exist at", self.chunksDir)

        self.packs = []
        self.transcript_packs = []
        packOfChunks = []
        packOfTranscripts = []

        for i, packagesPaths in enumerate(zip(packages2Load, transcripts2Load)):
            packagePath, transcriptPath = packagesPaths
            packOfChunks.append(packagePath)
            if self.transcript_window is not None:
                packOfTranscripts.append(transcriptPath)
            if (i + 1) % self.NUM_CHUNKS_INMEM == 0:
                self.packs.append(packOfChunks)
                if self.transcript_window is not None:
                    self.transcript_packs.append(packOfTranscripts)
                packOfChunks = []
                packOfTranscripts = []
        if len(packOfChunks) > 0:
            self.packs.append(packOfChunks)
        if len(packOfTranscripts) > 0:
            if self.transcript_window is not None:
                self.transcript_packs.append(packOfTranscripts)

        self.currentPack = -1
        self.nextPack = 0
        self.sequenceIdx = 0
        self.previousCategory = -1

        self.data = None
        self.transcript = None

        self._loadNextPack(first=True)
        self._loadNextPack()

    def _createChunks(self):
        print("Creating chunks at", self.chunksDir)
        pack = []
        transcriptions = []
        packIds = []
        packageSize = 0
        packageIdx = 0
        for trackId in tqdm.tqdm(self.sequencesData.index):
            if self.sequencesData.loc[trackId, 'ensemble'] == self.filter_transcription:
                sequence, samplingRate = librosa.load(self.rawAudioPath / (str(trackId) + '.wav'), sr=16000)
                sequence = torch.tensor(sequence).float()
                # the transcript is created alongside the data
                # this ensures the same ordering when the data and the corresponding transcripts are loaded!!
                # if I understand this correctly, there is nothing more we need to do regarding this
                # - simply load them as labels in the latter functions
                if self.transcript_window is not None:
                    transcript = self._musicTranscripter(sequence, f'{self.rawLabelsPath}/{str(trackId)}.csv')
                    transcriptions.append(transcript)
                packIds.append(trackId) # the Ids of songs are saved here if needed to be recalled for indexing later?
                pack.append(sequence)
                packageSize += len(sequence) * 4
                if packageSize >= self.CHUNK_SIZE:
                    print(f"Saved pack {packageIdx}")
                    with open(self.chunksDir / f'chunk_{packageIdx}.pickle', 'wb') as handle:
                        pickle.dump(torch.cat(pack, dim=0), handle, protocol=pickle.HIGHEST_PROTOCOL)
                    with open(self.chunksDir / f'ids_{packageIdx}.pickle', 'wb') as handle:
                        pickle.dump(packIds, handle, protocol=pickle.HIGHEST_PROTOCOL)
                    if self.transcript_window is not None:
                        with open(self.chunksDir / f'transcript_{packageIdx}.pickle', 'wb') as handle:
                            pickle.dump(torch.cat(transcriptions, dim=0), handle, protocol=pickle.HIGHEST_PROTOCOL)
                    pack = []
                    transcriptions = []
                    packIds = []
                    packageSize = 0
                    packageIdx += 1
                else:
                    continue
        print(f"Saved pack {packageIdx}")
        with open(self.chunksDir / f'chunk_{packageIdx}.pickle', 'wb') as handle:
            pickle.dump(torch.cat(pack, dim=0), handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(self.chunksDir / f'ids_{packageIdx}.pickle', 'wb') as handle:
            pickle.dump(packIds, handle, protocol=pickle.HIGHEST_PROTOCOL)
        if self.transcript_window is not None:
            with open(self.chunksDir / f'transcript_{packageIdx}.pickle', 'wb') as handle:
                        pickle.dump(torch.cat(transcriptions, dim=0), handle, protocol=pickle.HIGHEST_PROTOCOL)

    def _loadNextPack(self, first=False):
        self.clear()
        if not first:
            self.currentPack = self.nextPack
            startTime = time.time()
            print('Loading files')
            self.categoryLabel = [0]
            packageIdx = [0]
            trackId =[0]
            self.seqLabel = [0]
            packageSize = 0
            if self.currentPack == 0:
                with open(self.chunksDir / (
                        'ids_' + self.packs[0][0].split('_', maxsplit=1)[-1]), 'rb') as handle:
                    chunkIds = pickle.load(handle)

                    self.previousCategory = self.sequencesData.loc[chunkIds[0]][self.category]

            for packagePath in self.packs[self.currentPack]:
                with open(self.chunksDir / ('ids_' + packagePath.split('_', maxsplit=1)[-1]), 'rb') as handle:
                    chunkIds = pickle.load(handle)
                for seqId in chunkIds:
                    currentCategory = seqId
                    self.categoryLabel.append(packageSize)
                    packageSize += np.unique(self.sequencesData.loc[seqId]['length'])[0]
                    self.seqLabel.append(packageSize)
                packageIdx.append(packageSize)
            self.categoryLabel.append(packageSize)

            self.data = torch.empty(size=(packageSize,))

            ##########################################################
            ## NEED TO LOAD THE CREATED TRANSCRIPTIONS HERE AS WELL ##
            ##########################################################

            for i, packagePath in enumerate(self.packs[self.currentPack]):
                with open(self.chunksDir / packagePath, 'rb') as handle:
                    self.data[packageIdx[i]:packageIdx[i + 1]] = pickle.load(handle)
            for i, packagePath in enumerate(self.transcript_packs[self.currentPack]):
                with open(self.chunksDir / packagePath, 'rb') as handle:
                    # better to predefine an empty tensor and fill it up.
                    if i == 0:
                        self.transcript = pickle.load(handle)
                    else:
                        self.transcript = torch.cat((self.transcript, pickle.load(handle)), dim=0)
            if self.useGPU:
                self.data = self.data.cuda(non_blocking=True)
                self.transcript = self.transcript.cuda(non_blocking=True)
                print("Data moved to GPU")
                print("Data in: ", self.data.device)
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
        return self.labels[idCategory]

    def __len__(self):
        return self.totSize // self.sizeWindow

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self.data) - self.sizeWindow - 1:
            print(idx)

        outData = self.data[idx:(self.sizeWindow + idx)].view(1, -1)
    
        if self.transcript_window is not None:
            ##########################################################################
            # THIS SEEMS TO BE ALRIGHT PROVIDING SELF.TRANSCRIPT IS CREATED PROPERLY #
            ##########################################################################
            # label = self.transcript[idx:int(self.sizeWindow / self.transcript_window * idx), :]
        
            window_start = int(idx // self.transcript_window)
            window_end   = int((self.sizeWindow + idx) // self.transcript_window)
            label = self.transcript[window_start:window_end, :]
        else:
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


    def _musicTranscripter(self, 
                           sequence,
                           labels_path,
                           n_notes=128):
    
        '''
        Provides binary target matrices denoting all instruments that play during
        for each window_size_ms window in the latent representations
        '''

        labeling = pd.read_csv(labels_path)
        labeling['start_time'] = labeling['start_time'] / 44100
        labeling['end_time'] = labeling['end_time'] / 44100

        ###################################################################
        ########## POTENTIAL TROUBLES WITH THE NUMBER OF WINDOWS ##########
        ###################################################################

        n_windows = int(np.ceil(sequence.shape[0] / self.transcript_window)) # allow for aggregating windows
        transcript = torch.zeros(n_windows, n_notes + 1)
        
        transcript_window_ms = self.transcript_window * 1000 / 16000

        for i in range(n_windows):
            
            start = (i) * transcript_window_ms
            end = (i+1) * transcript_window_ms
        
            notes = labeling[(labeling['start_time'] <= end) & \
                           (start<= labeling['start_time'])].note.values

            if len(notes) == 0:
                # if silence, then instrument == 0 plays note == 0
                transcript[i, 0] = 1
            else:
                transcript[i, notes] = 1

        return transcript


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
            # print("Data loader nLoop: ", self.nLoop)
            # print("Len data loader: ", len(dataloader))
            # print("Len of sampler: ", len(sampler))
            # assert False
            # print("Dataloader len: \n", len(dataloader))
            for j, x in enumerate(dataloader):
                # print("Data loader yielded batch #: ", j)
                yield x
            # print("Len data loader: ", len(dataloader), "and consummed: ", j + 1)
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

        # print("Sampling intervals:\n", self.samplingIntervals)
        nWindows = len(self.samplingIntervals) - 1
        self.sizeSamplers = [(self.samplingIntervals[i + 1] -
                              self.samplingIntervals[i]) // self.sizeWindow
                             for i in range(nWindows)]  # How many windows a sequence/category lasts

        # assert False
        if self.offset > 0:
            self.sizeSamplers = [max(0, x - 1) for x in self.sizeSamplers]
        # print("Size samplers:\n", self.sizeSamplers)
        # print("Size samplers over batch size:\n", np.array(self.sizeSamplers) // self.batchSize)

        order = [(x, torch.randperm(val).tolist())
                 for x, val in enumerate(self.sizeSamplers) if
                 val > 0]  # (index of seq/cat, randomly permuted numbers from 0 to num windows in seq(cat))

        # Build Batches
        self.batches = []
        for indexSampler, randperm in order:
            indexStart, sizeSampler = 0, self.sizeSamplers[indexSampler]
            while indexStart < (sizeSampler - self.batchSize):
                indexEnd = indexStart + self.batchSize
                locBatch = [self.getIndex(x, indexSampler)
                            for x in randperm[indexStart:indexEnd]]
                indexStart = indexEnd
                self.batches.append(locBatch)
        # print("Number of batches:\n", len(self.batches))
        # print("Batches:\n", self.batches)
        # print("Batches shape: \n", np.array(self.batches).shape)
        # print("Batches vstack shape: \n", np.vstack(self.batches).shape)
        self.batches = np.vstack(self.batches)

    def __len__(self):
        return len(self.batches)

    def getIndex(self, x, iInterval):
        return self.offset + x * self.sizeWindow + self.samplingIntervals[iInterval]

    def __iter__(self):
        random.shuffle(self.batches)
        return iter(self.batches)
