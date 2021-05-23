import torch
import torch.nn as nn


class ChannelNorm(nn.Module):

    def __init__(self,
                 numFeatures,
                 epsilon=1e-05,
                 affine=True):

        super(ChannelNorm, self).__init__()
        if affine:
            self.weight = nn.parameter.Parameter(torch.Tensor(1,
                                                              numFeatures, 1))
            self.bias = nn.parameter.Parameter(torch.Tensor(1, numFeatures, 1))
        else:
            self.weight = None
            self.bias = None
        self.epsilon = epsilon
        self.p = 0
        self.affine = affine
        self.reset_parameters()

    def reset_parameters(self):
        if self.affine:
            torch.nn.init.ones_(self.weight)
            torch.nn.init.zeros_(self.bias)

    def forward(self, x):

        cumMean = x.mean(dim=1, keepdim=True)
        cumVar = x.var(dim=1, keepdim=True)
        x = (x - cumMean) * torch.rsqrt(cumVar + self.epsilon)

        if self.weight is not None:
            x = x * self.weight + self.bias
        return x


class CPCEncoder(nn.Module):

    def __init__(self,
                 sizeHidden=512,
                 normMode="layerNorm"):

        super(CPCEncoder, self).__init__()

        validModes = ["batchNorm", "instanceNorm", "ID", "layerNorm"]
        if normMode not in validModes:
            raise ValueError(f"Norm mode must be in {validModes}")

        if normMode == "instanceNorm":
            def normLayer(x):
                return nn.InstanceNorm1d(x, affine=True)
        elif normMode == "layerNorm":
            normLayer = ChannelNorm
        else:
            normLayer = nn.BatchNorm1d

        self.dimEncoded = sizeHidden
        self.conv0 = nn.Conv1d(1, sizeHidden, 10, stride=5, padding=3)
        self.batchNorm0 = normLayer(sizeHidden)
        self.conv1 = nn.Conv1d(sizeHidden, sizeHidden, 8, stride=4, padding=2)
        self.batchNorm1 = normLayer(sizeHidden)
        self.conv2 = nn.Conv1d(sizeHidden, sizeHidden, 4,
                               stride=2, padding=1)
        self.batchNorm2 = normLayer(sizeHidden)
        self.conv3 = nn.Conv1d(sizeHidden, sizeHidden, 4, stride=2, padding=1)
        self.batchNorm3 = normLayer(sizeHidden)
        self.conv4 = nn.Conv1d(sizeHidden, sizeHidden, 4, stride=2, padding=1)
        self.batchNorm4 = normLayer(sizeHidden)
        self.DOWNSAMPLING = 160

    def getDimOutput(self):
        return self.conv4.out_channels

    def forward(self, x):
        x = torch.relu(self.batchNorm0(self.conv0(x)))
        x = torch.relu(self.batchNorm1(self.conv1(x)))
        x = torch.relu(self.batchNorm2(self.conv2(x)))
        x = torch.relu(self.batchNorm3(self.conv3(x)))
        x = torch.relu(self.batchNorm4(self.conv4(x)))
        return x


class CPCAR(nn.Module):

    def __init__(self,
                 dimEncoded,
                 dimOutput,
                 keepHidden,
                 nLevelsGRU,
                 mode="GRU",
                 reverse=False):

        super(CPCAR, self).__init__()
        self.RESIDUAL_STD = 0.1

        if mode == "LSTM":
            self.baseNet = nn.LSTM(dimEncoded, dimOutput,
                                   num_layers=nLevelsGRU, batch_first=True)
        elif mode == "RNN":
            self.baseNet = nn.RNN(dimEncoded, dimOutput,
                                  num_layers=nLevelsGRU, batch_first=True)
        else:
            self.baseNet = nn.GRU(dimEncoded, dimOutput,
                                  num_layers=nLevelsGRU, batch_first=True)

        self.hidden = None
        self.keepHidden = keepHidden
        self.reverse = reverse

    def getDimOutput(self):
        return self.baseNet.hidden_size

    def forward(self, x):

        if self.reverse:
            x = torch.flip(x, [1])
        try:
            self.baseNet.flatten_parameters()
        except RuntimeError:
            pass
        x, h = self.baseNet(x, self.hidden)
        if self.keepHidden:
            if isinstance(h, tuple):
                self.hidden = tuple(x.detach() for x in h)
            else:
                self.hidden = h.detach()

        # For better modularity, a sequence's order should be preserved
        # by each module
        if self.reverse:
            x = torch.flip(x, [1])
        return x


class CPCModel(nn.Module):

    def __init__(self,
                 encoder,
                 AR):
        super(CPCModel, self).__init__()
        self.gEncoder = encoder
        self.gAR = AR

    def forward(self, batchData, label):
        encodedData = self.gEncoder(batchData).permute(0, 2, 1)
        cFeature = self.gAR(encodedData)
        return cFeature, encodedData, label


class PredictionNetwork(nn.Module):

    def __init__(self,
                 nPredicts,
                 dimOutputAR,
                 dimOutputEncoder,
                 dropout=False):

        super(PredictionNetwork, self).__init__()
        self.predictors = nn.ModuleList()
        self.RESIDUAL_STD = 0.01
        self.dimOutputAR = dimOutputAR

        self.dropout = nn.Dropout(p=0.5) if dropout else None
        for i in range(nPredicts):
            self.predictors.append(
                nn.Linear(dimOutputAR, dimOutputEncoder, bias=False))
            if dimOutputEncoder > dimOutputAR:
                residual = dimOutputEncoder - dimOutputAR
                self.predictors[-1].weight.data.copy_(torch.cat([torch.randn(
                    dimOutputAR, dimOutputAR), self.RESIDUAL_STD * torch.randn(residual, dimOutputAR)], dim=0))

    def forward(self, c, candidates):

        assert (len(candidates) == len(self.predictors))
        out = []

        # UGLY
        # if isinstance(self.predictors[0], EqualizedConv1d):
        # c = c.permute(0, 2, 1)

        for k in range(len(self.predictors)):

            locC = self.predictors[k](c)
            if isinstance(locC, tuple):
                locC = locC[0]
            # if isinstance(self.predictors[k], EqualizedConv1d):
            # locC = locC.permute(0, 2, 1)
            if self.dropout is not None:
                locC = self.dropout(locC)
            locC = locC.view(locC.size(0), 1, locC.size(1), locC.size(2))
            outK = (locC * candidates[k]).mean(dim=3)
            out.append(outK)
        return out


class BaseCriterion(nn.Module):
    def update(self):
        return


class CPCUnsupersivedCriterion(BaseCriterion):

    def __init__(self,
                 nPredicts,  # Number of steps
                 dimOutputAR,  # Dimension of G_ar
                 dimOutputEncoder,  # Dimension of the convolutional net
                 negativeSamplingExt,  # Number of negative samples to draw
                 mode=None,
                 dropout=False):

        super(CPCUnsupersivedCriterion, self).__init__()

        self.wPrediction = PredictionNetwork(
            nPredicts, dimOutputAR, dimOutputEncoder, dropout=dropout)
        self.nPredicts = nPredicts
        self.negativeSamplingExt = negativeSamplingExt
        self.lossCriterion = nn.CrossEntropyLoss()

        if mode not in [None, "reverse"]:
            raise ValueError("Invalid mode")

        self.mode = mode

    def sampleClean(self, encodedData, windowSize):

        batchSize, nNegativeExt, dimEncoded = encodedData.size()
        outputs = []

        negExt = encodedData.contiguous().view(-1, dimEncoded)
        # Draw nNegativeExt * batchSize negative samples anywhere in the batch
        batchIdx = torch.randint(low=0, high=batchSize,
                                 size=(self.negativeSamplingExt
                                       * windowSize * batchSize,),
                                 device=encodedData.device)

        seqIdx = torch.randint(low=1, high=nNegativeExt,
                               size=(self.negativeSamplingExt
                                     * windowSize * batchSize,),
                               device=encodedData.device)

        baseIdx = torch.arange(0, windowSize, device=encodedData.device)
        baseIdx = baseIdx.view(1, 1,
                               windowSize).expand(1,
                                                  self.negativeSamplingExt,
                                                  windowSize).expand(batchSize, self.negativeSamplingExt, windowSize)
        seqIdx += baseIdx.contiguous().view(-1)
        seqIdx = torch.remainder(seqIdx, nNegativeExt)

        extIdx = seqIdx + batchIdx * nNegativeExt
        negExt = negExt[extIdx].view(batchSize, self.negativeSamplingExt,
                                     windowSize, dimEncoded)

        labelLoss = torch.zeros((batchSize * windowSize),
                                dtype=torch.long,
                                device=encodedData.device)

        for k in range(1, self.nPredicts + 1):

            # Positive samples
            if k < self.nPredicts:
                posSeq = encodedData[:, k:-(self.nPredicts - k)]
            else:
                posSeq = encodedData[:, k:]

            posSeq = posSeq.view(batchSize, 1, windowSize, dimEncoded)
            fullSeq = torch.cat((posSeq, negExt), dim=1)
            outputs.append(fullSeq)

        return outputs, labelLoss

    def getInnerLoss(self):

        return "orthoLoss", self.orthoLoss * self.wPrediction.orthoCriterion()

    def forward(self, cFeature, encodedData):

        if self.mode == "reverse":
            encodedData = torch.flip(encodedData, [1])
            cFeature = torch.flip(cFeature, [1])

        batchSize, seqSize, dimAR = cFeature.size()
        windowSize = seqSize - self.nPredicts

        cFeature = cFeature[:, :windowSize]

        sampledData, labelLoss = self.sampleClean(encodedData, windowSize)

        predictions = self.wPrediction(cFeature, sampledData)

        outLosses = [0 for _ in range(self.nPredicts)]
        outAcc = [0 for _ in range(self.nPredicts)]

        for k, locPreds in enumerate(predictions[:self.nPredicts]):
            locPreds = locPreds.permute(0, 2, 1)  # (batchSize, 1 + negativeSamplingExt, windowSize) to
            #                                       (batchSize, windowSize, 1 + negativeSamplingExt)
            locPreds = locPreds.contiguous().view(
                -1, locPreds.size(2))  # (batchSize, windowSize, 1 + negativeSamplingExt) to
            #                            (batchSize * windowSize, 1 + negativeSamplingExt)
            lossK = self.lossCriterion(locPreds, labelLoss)
            outLosses[k] += lossK.view(1, -1)
            _, predsIndex = locPreds.max(1)
            outAcc[k] += torch.sum(predsIndex == labelLoss).float().view(1, -1)

        return torch.cat(outLosses, dim=1), torch.cat(outAcc, dim=1) / (windowSize * batchSize)
