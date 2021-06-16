import torch
import torch.nn as nn
import math
import numpy as np
# from utils import getCheckpointData
import os


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


class SincConv1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False,
                 padding_mode='zeros', sampleRate=16000, minLowHz=50, minBandHz=50):
        super(SincConv1D, self).__init__()
        self.padding_mode = padding_mode
        if in_channels != 1:
            msg = "SincConv1D only support one input channel (here, in_channels = {%i})" % in_channels
            raise ValueError(msg)
        self.outChannels = out_channels
        self.kernelSize = kernel_size
        # Forcing the filters to be odd (i.e, perfectly symmetrics)
        if self.kernelSize % 2 == 0:
            self.kernelSize += 1
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        if bias:
            raise ValueError('SincConv1D does not support bias.')
        if groups > 1:
            raise ValueError('SincConv1D does not support groups.')
        self.sampleRate = sampleRate
        self.minLowHz = minLowHz
        self.minBandHz = minBandHz
        # Initialize filterbanks such that they are equally spaced in Mel scale
        lowHz = 30
        highHz = self.sampleRate / 2 - (self.minLowHz + self.minBandHz)
        mel = np.linspace(self.hz2Mel(lowHz), self.hz2Mel(highHz), self.outChannels + 1)
        hz = self.mel2Hz(mel)
        # Filter lower frequency (outChannels, 1)
        self.lowHz_ = torch.nn.Parameter(torch.Tensor(hz[:-1]).view(-1, 1))
        # Filter frequency band (outChannels, 1)
        self.bandHz_ = torch.nn.Parameter(torch.Tensor(np.diff(hz)).view(-1, 1))
        # Hamming window
        nLin = torch.linspace(0, (self.kernelSize / 2) - 1,
                              steps=int((self.kernelSize / 2)))  # computing only half of the window
        self.window_ = 0.54 - 0.46 * torch.cos(2 * math.pi * nLin / self.kernelSize)
        n = (self.kernelSize - 1) / 2.0
        self.n_ = 2 * math.pi * torch.arange(-n, 0).view(1,
                                                         -1) / self.sampleRate  # Due to symmetry, we only need half of
        # the time axes
        self.filters = None

    @staticmethod
    def hz2Mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    @staticmethod
    def mel2Hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)

    def forward(self, waveforms):
        self.n_ = self.n_.to(waveforms.device)
        self.window_ = self.window_.to(waveforms.device)
        low = self.minLowHz + torch.abs(self.lowHz_)
        high = torch.clamp(low + self.minBandHz + torch.abs(self.bandHz_), self.minLowHz, self.sampleRate / 2)
        band = (high - low)[:, 0]
        fTimesTLow = torch.matmul(low, self.n_)
        fTimesTHigh = torch.matmul(high, self.n_)
        # Equivalent of Eq.4 of the reference paper
        bandPassLeft = ((torch.sin(fTimesTHigh) - torch.sin(fTimesTLow)) / (self.n_ / 2)) * self.window_
        bandPassCenter = 2 * band.view(-1, 1)
        bandPassRight = torch.flip(bandPassLeft, dims=[1])
        bandPass = torch.cat([bandPassLeft, bandPassCenter, bandPassRight], dim=1)
        bandPass = bandPass / (2 * band[:, None])
        self.filters = bandPass.view(self.outChannels, 1, self.kernelSize)
        return torch.conv1d(waveforms, self.filters, stride=self.stride, padding=self.padding,
                            dilation=self.dilation, bias=None, groups=1)


class CPCEncoder(nn.Module):

    def __init__(self,
                 sizeHidden=512,
                 normMode="layerNorm", sincNet=False):

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
        if sincNet:
            self.conv0 = SincConv1D(1, sizeHidden, 10, stride=5, padding=3)
        else:
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


class ConcatenatedModel(nn.Module):

    def __init__(self, model_list):
        super(ConcatenatedModel, self).__init__()
        self.models = torch.nn.ModuleList(model_list)

    def forward(self, batchData, label):
        outFeatures = []
        outEncoded = []
        for model in self.models:
            cFeature, encodedData, label = model(batchData, label)
            outFeatures.append(cFeature)
            outEncoded.append(encodedData)
        return torch.cat(outFeatures, dim=2), torch.cat(outEncoded, dim=2), label


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

    def forward(self, cFeature, encodedData, label):

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


class ScaledDotProductAttention(nn.Module):
    def __init__(self,
                 sizeSeq,  # Size of the input sequence
                 dk,  # Dimension of the input sequence
                 dropout,  # Dropout parameter
                 relpos=False):  # Do we retrieve positional information ?
        super(ScaledDotProductAttention, self).__init__()

        self.drop = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=2)
        self.relpos = relpos
        self.sizeSeq = sizeSeq

        if relpos:
            self.Krelpos = nn.Parameter(torch.Tensor(dk, sizeSeq))
            self.initmat_(self.Krelpos)
            self.register_buffer('z', torch.zeros(1, sizeSeq, 1))

        # A mask is set so that a node never queries data in the future
        mask = torch.tril(torch.ones(sizeSeq, sizeSeq), diagonal=0)
        mask = 1 - mask
        mask[mask == 1] = -float('inf')
        self.register_buffer('mask', mask.unsqueeze(0))

    @staticmethod
    def initmat_(mat, dim=0):
        stdv = 1. / math.sqrt(mat.size(dim))
        mat.data.uniform_(-stdv, stdv)

    def forward(self, Q, K, V):
        # Input dim : N x sizeSeq x dk
        QK = torch.bmm(Q, K.transpose(-2, -1))

        if self.relpos:
            bsz = Q.size(0)
            QP = Q.matmul(self.Krelpos)
            # This trick with z fills QP's diagonal with zeros
            QP = torch.cat((self.z.expand(bsz, -1, -1), QP), 2)
            QK += QP.view(bsz, self.sizeSeq + 1, self.sizeSeq)[:, 1:, :]
        A = self.softmax(QK / math.sqrt(K.size(-1)) + self.mask)
        return torch.bmm(self.drop(A), V)


class MultiHeadAttention(nn.Module):
    def __init__(self,
                 sizeSeq,  # Size of a sequence
                 dropout,  # Dropout parameter
                 dmodel,  # Model's dimension
                 nheads,  # Number of heads in the model
                 abspos):  # Is positional information encoded in the input ?
        super(MultiHeadAttention, self).__init__()
        self.Wo = nn.Linear(dmodel, dmodel, bias=False)
        self.Wk = nn.Linear(dmodel, dmodel, bias=False)
        self.Wq = nn.Linear(dmodel, dmodel, bias=False)
        self.Wv = nn.Linear(dmodel, dmodel, bias=False)
        self.nheads = nheads
        self.dk = dmodel // nheads
        self.Att = ScaledDotProductAttention(sizeSeq, self.dk,
                                             dropout, not abspos)

    def trans_(self, x):
        bsz, bptt, h, dk = x.size(0), x.size(1), self.nheads, self.dk
        return x.view(bsz, bptt, h, dk).transpose(1, 2).contiguous().view(bsz * h, bptt, dk)

    def reverse_trans_(self, x):
        bsz, bptt, h, dk = x.size(
            0) // self.nheads, x.size(1), self.nheads, self.dk
        return x.view(bsz, h, bptt, dk).transpose(1, 2).contiguous().view(bsz, bptt, h * dk)

    def forward(self, Q, K, V):
        q = self.trans_(self.Wq(Q))
        k = self.trans_(self.Wk(K))
        v = self.trans_(self.Wv(V))
        y = self.reverse_trans_(self.Att(q, k, v))
        return self.Wo(y)


class FFNetwork(nn.Module):
    def __init__(self, din, dout, dff, dropout):
        super(FFNetwork, self).__init__()
        self.lin1 = nn.Linear(din, dff, bias=True)
        self.lin2 = nn.Linear(dff, dout, bias=True)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.lin2(self.drop(self.relu(self.lin1(x))))


class TransformerLayer(nn.Module):
    def __init__(self, sizeSeq=32, dmodel=512, dff=2048,
                 dropout=0.1, nheads=8,
                 abspos=False):
        super(TransformerLayer, self).__init__()
        self.multihead = MultiHeadAttention(sizeSeq, dropout,
                                            dmodel, nheads, abspos)
        self.ln_multihead = nn.LayerNorm(dmodel)
        self.ffnetwork = FFNetwork(dmodel, dmodel, dff, dropout)
        self.ln_ffnetwork = nn.LayerNorm(dmodel)

    def forward(self, x):
        y = self.ln_multihead(x + self.multihead(Q=x, K=x, V=x))
        return self.ln_ffnetwork(y + self.ffnetwork(y))


class StaticPositionEmbedding(nn.Module):
    def __init__(self, seqlen, dmodel):
        super(StaticPositionEmbedding, self).__init__()
        pos = torch.arange(0., seqlen).unsqueeze(1).repeat(1, dmodel)
        dim = torch.arange(0., dmodel).unsqueeze(0).repeat(seqlen, 1)
        div = torch.exp(- math.log(10000) * (2 * (dim // 2) / dmodel))
        pos *= div
        pos[:, 0::2] = torch.sin(pos[:, 0::2])
        pos[:, 1::2] = torch.cos(pos[:, 1::2])
        self.register_buffer('pe', pos.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


def buildTransformerAR(dimEncoded,  # Output dimension of the encoder
                       nLayers,  # Number of transformer layers
                       sizeSeq,  # Expected size of the input sequence
                       abspos):
    layerSequence = []
    if abspos:
        layerSequence += [StaticPositionEmbedding(sizeSeq, dimEncoded)]
    layerSequence += [TransformerLayer(sizeSeq=sizeSeq,
                                       dmodel=dimEncoded, abspos=abspos)
                      for _ in range(nLayers)]
    return nn.Sequential(*layerSequence)


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


def loadModel(pathCheckpoints, locArgs, loadStateDict=True):
    models = []
    hiddenGar, hiddenEncoder = 0, 0
    for path in pathCheckpoints:
        print(f"Loading checkpoint {path}")
        # _, _, locArgs = getCheckpointData(os.path.dirname(path))

        # doLoad = locArgs.load is not None and \
        #     (len(locArgs.load) > 1 or
        #      os.path.dirname(locArgs.load[0]) != os.path.dirname(path))

        # if doLoad:
        #     m_, hg, he = loadModel(locArgs.load, loadStateDict=False)
        #     hiddenGar += hg
        #     hiddenEncoder += he
        # else:
        encoderNet = CPCEncoder(locArgs.hiddenEncoder, 'layerNorm', sincNet=locArgs.encoderType == 'sinc')

        arNet = getAR(locArgs)
        m_ = CPCModel(encoderNet, arNet)

        if loadStateDict:
            print(f"Loading the state dict at {path}")
            state_dict = torch.load(path, 'cpu')
            m_.load_state_dict(state_dict["gEncoder"], strict=False)

        hiddenGar += locArgs.hiddenGar
        hiddenEncoder += locArgs.hiddenEncoder

        models.append(m_)

    if len(models) == 1:
        return models[0], hiddenGar, hiddenEncoder

    return ConcatenatedModel(models), hiddenGar, hiddenEncoder


class CategoryCriterion(BaseCriterion):

    def __init__(self,
                 hiddenGar,
                 sizeWindow,
                 downSampling,
                 numClasses,
                 pool=None):
        super(CategoryCriterion, self).__init__()
        self.pool = pool
        if pool is not None:
            kernelSize, padding, stride = pool
            self.avgPool = nn.AvgPool1d(kernelSize, stride, padding)
            self.numFeatures = hiddenGar * (((sizeWindow // downSampling) + 2 * padding - kernelSize) // stride + 1)
        else:
            self.numFeatures = hiddenGar * (sizeWindow // downSampling)
        self.numClasses = numClasses
        self.lossCriterion = nn.CrossEntropyLoss()
        # print("Num features: ", self.numFeatures)
        # print("hiddenGar: ", hiddenGar)
        # print("Seq length: ", (sizeWindow // downSampling))
        # print("Pool: ", pool)
        self.wPrediction = nn.Linear(self.numFeatures, numClasses)

    def forward(self, x, encodedData, label):
        # if not model.optimize:
        x = x.transpose(1, 2).detach()
        # print(cFeature.size())
        batchSize, dimAR, seqSize = x.size()
        if self.pool is not None:
            x = self.avgPool(x)
        # print(cFeature.size())
        # assert False
        x = x.view(batchSize, self.numFeatures)
        predictions = self.wPrediction(x)
        loss = self.lossCriterion(predictions, label)
        _, predsIndex = predictions.max(1)
        accuracy = torch.sum(predsIndex == label).float().view(1, -1) / batchSize
        return loss.view(1, -1), accuracy


class TranscriptionCriterion(BaseCriterion):

    def __init__(self,
                 hiddenGar,
                 sizeWindow,
                 downSampling,
                 numClasses=129,
                 transcript_window=10,
                 pool=None):
        super(TranscriptionCriterion, self).__init__()
        self.pool = pool
        if pool is not None:
            kernelSize, padding, stride = pool
            self.avgPool = nn.AvgPool1d(kernelSize, stride, padding)
            self.numFeatures = int(hiddenGar * (((sizeWindow // downSampling) + 2 * padding - kernelSize) // stride + 1))
        else:
            self.numFeatures = int(hiddenGar * (sizeWindow // downSampling))
        print(self.numFeatures)
        self.numClasses = numClasses
        self.lossCriterion = nn.BCEWithLogitsLoss()
        self.wPrediction = nn.Sequential(
            nn.Linear(self.numFeatures, 100), # possibly change n_neurons?
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Linear(100, numClasses * (sizeWindow // downSampling))
        )

    def forward(self, x, encodedData, label):
        # get x of size (batch, 128, feat_dim) --> (N, L, C)
        # x = x.transpose(1, 2).detach()
        # x of size (batch, feat_dim, 128) --> (N, C, L)
        batchSize, seqSize, dimAR = x.size()

        if self.pool is not None:
            x = self.avgPool(x)

        label = label.contiguous().view(batchSize, self.numClasses * seqSize)
        x = x.contiguous().view(batchSize, seqSize * dimAR)

        predictions = self.wPrediction(x)
        predictions_sigm = nn.Sigmoid()(predictions)
        predsIndex = predictions_sigm > 0.5

        loss = self.lossCriterion(predictions, label)
        accuracy = torch.sum(predsIndex == label).float().view(1, -1) / (batchSize * label.shape[1])
        return loss.view(1, -1), accuracy, predsIndex
