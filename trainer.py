import numpy as np
import torch
import time
from copy import deepcopy
from utils import saveLogs, updateLogs, showLogs, saveCheckpoint


def trainStep(dataLoader,
              cpcModel,
              cpcCriterion,
              optimizer,
              loggingStep,
              useGPU,
              log2Board,
              totalSteps,
              experiment):
    cpcModel.train()
    cpcCriterion.train()

    startTime = time.perf_counter()
    n_examples = 0
    logs, lastlogs = {}, None
    iterCtr = 0

    gradmapGEncoder = {}
    gradmapGAR = {}
    gradmapWPrediction = {}

    if log2Board > 1 and totalSteps == 0:
        logWeights(cpcModel.gEncoder, totalSteps, experiment)
        logWeights(cpcModel.gAR, totalSteps, experiment)
        logWeights(cpcCriterion.wPrediction, totalSteps, experiment)

    for step, fulldata in enumerate(dataLoader):
        batchData, label = fulldata
        n_examples += batchData.size(0)
        if useGPU:
            batchData = batchData.cuda(non_blocking=True)
            label = label.cuda(non_blocking=True)
        c_feature, encoded_data, label = cpcModel(batchData, label)
        allLosses, allAcc = cpcCriterion(c_feature, encoded_data)
        totLoss = allLosses.sum()

        totLoss.backward()

        if log2Board > 1:
            gradmapGEncoder = updateGradientMap(cpcModel.gEncoder, gradmapGEncoder)
            gradmapGAR = updateGradientMap(cpcModel.gAR, gradmapGAR)
            gradmapWPrediction = updateGradientMap(cpcCriterion.wPrediction, gradmapWPrediction)

        optimizer.step()
        optimizer.zero_grad()

        if "locLoss_train" not in logs:
            logs["locLoss_train"] = np.zeros(allLosses.size(1))
            logs["locAcc_train"] = np.zeros(allLosses.size(1))

        logs["locLoss_train"] += (allLosses.mean(dim=0)).detach().cpu().numpy()
        logs["locAcc_train"] += (allAcc.mean(dim=0)).cpu().numpy()
        iterCtr += 1

        if log2Board:
            for t in range(len(logs["locLoss_train"])):
                experiment.log_metric(f"Losses/batch/locLoss_train_{t}", logs["locLoss_train"][t] / iterCtr,
                                      step=totalSteps + iterCtr)
                experiment.log_metric(f"Accuracy/batch/locAcc_train_{t}", logs["locAcc_train"][t] / iterCtr,
                                      step=totalSteps + iterCtr)

        if (step + 1) % loggingStep == 0:
            new_time = time.perf_counter()
            elapsed = new_time - startTime
            print(f"Update {step + 1}")
            print(f"elapsed: {elapsed:.1f} s")
            print(
                f"{1000.0 * elapsed / loggingStep:.1f} ms per batch, {1000.0 * elapsed / n_examples:.1f} ms / example")
            locLogs = updateLogs(logs, loggingStep, lastlogs)
            lastlogs = deepcopy(logs)
            showLogs("Training loss", locLogs)
            startTime, n_examples = new_time, 0

            if log2Board > 1:
                # Log gradients
                logGradients(gradmapGEncoder, totalSteps + iterCtr, experiment, scaleBy=1.0 / iterCtr)
                logGradients(gradmapGAR, totalSteps + iterCtr, experiment, scaleBy=1.0 / iterCtr)
                logGradients(gradmapWPrediction, totalSteps + iterCtr, experiment, scaleBy=1.0 / iterCtr)
                # Log weights
                logWeights(cpcModel.gEncoder, totalSteps + iterCtr, experiment)
                logWeights(cpcModel.gAR, totalSteps + iterCtr, experiment)
                logWeights(cpcCriterion.wPrediction, totalSteps + iterCtr, experiment)

    logs = updateLogs(logs, iterCtr)
    logs["iter"] = iterCtr
    showLogs("Average training loss on epoch", logs)
    return logs


def valStep(dataLoader,
            cpcModel,
            cpcCriterion,
            useGPU):
    cpcCriterion.eval()
    cpcModel.eval()
    logs = {}
    cpcCriterion.eval()
    cpcModel.eval()
    iterCtr = 0

    for step, fulldata in enumerate(dataLoader):

        batchData, label = fulldata

        if useGPU:
            batchData = batchData.cuda(non_blocking=True)
            label = label.cuda(non_blocking=True)

        with torch.no_grad():
            c_feature, encoded_data, label = cpcModel(batchData, label)
            allLosses, allAcc = cpcCriterion(c_feature, encoded_data)

        if "locLoss_val" not in logs:
            logs["locLoss_val"] = np.zeros(allLosses.size(1))
            logs["locAcc_val"] = np.zeros(allLosses.size(1))

        iterCtr += 1
        logs["locLoss_val"] += allLosses.mean(dim=0).cpu().numpy()
        logs["locAcc_val"] += allAcc.mean(dim=0).cpu().numpy()

    logs = updateLogs(logs, iterCtr)
    logs["iter"] = iterCtr
    showLogs("Validation loss:", logs)
    return logs


def updateGradientMap(model, gradMap):
    for name, param in model.named_parameters():
        paramName = name.split('.')
        paramLabel = paramName[-1]
        if paramLabel not in ['weight_ih_l0', 'weight_hh_l0', 'bias_ih_l0', 'bias_hh_l0',
                              'lowHz_', 'bandHz_', 'weight', 'bias']:
            continue
        param = model
        for i in range(len(paramName)):
            param = getattr(param, paramName[i])
        gradMap.setdefault("%s/%s" % ("Gradients", name), 0)
        gradMap["%s/%s" % ("Gradients", name)] += param.grad
    return gradMap


def logGradients(gradMap, step, experiment, scaleBy=1.0):
    for k, v in gradMap.items():
        experiment.log_histogram_3d(v.cpu().detach().numpy() * scaleBy, name=k, step=step)


def logWeights(model, step, experiment):
    for name, param in model.named_parameters():
        paramName = name.split('.')
        paramLabel = paramName[-1]
        if paramLabel not in ['weight_ih_l0', 'weight_hh_l0', 'bias_ih_l0', 'bias_hh_l0',
                              'lowHz_', 'bandHz_', 'weight', 'bias']:
            continue
        param = model
        for i in range(len(paramName)):
            param = getattr(param, paramName[i])
        experiment.log_histogram_3d(param.cpu().detach().numpy(), name="%s/%s" % ("Parameters", name), step=step)


def trainingLoop(trainDataset,
                 valDataset,
                 batchSize,
                 samplingMode,
                 cpcModel,
                 cpcCriterion,
                 nEpoch,
                 optimizer,
                 pathCheckpoint,
                 logs,
                 useGPU,
                 log2Board=0,
                 experiment=None):
    print(f"Running {nEpoch} epochs")
    startEpoch = len(logs["epoch"])
    bestAcc = 0
    bestStateDict = None
    startTime = time.time()
    epoch = 0
    totalSteps = 0
    try:
        for epoch in range(startEpoch, nEpoch):
            print(f"Starting epoch {epoch}")
            trainLoader = trainDataset.getDataLoader(batchSize, samplingMode,
                                                     True, numWorkers=0)
            valLoader = valDataset.getDataLoader(batchSize, samplingMode, False,
                                                 numWorkers=0)

            print("Training dataset %d batches, Validation dataset %d batches, batch size %d" %
                  (len(trainLoader), len(valLoader), batchSize))

            locLogsTrain = trainStep(trainLoader, cpcModel, cpcCriterion, optimizer, logs["logging_step"],
                                     useGPU, log2Board, totalSteps, experiment)

            totalSteps += locLogsTrain['iter']

            locLogsVal = valStep(valLoader, cpcModel, cpcCriterion, useGPU)

            print(f'Ran {epoch + 1} epochs '
                  f'in {time.time() - startTime:.2f} seconds')

            if useGPU:
                torch.cuda.empty_cache()

            currentAccuracy = float(locLogsVal["locAcc_val"].mean())

            if log2Board:
                for t in range(len(locLogsVal["locLoss_val"])):
                    experiment.log_metric(f"Losses/epoch/locLoss_train_{t}",
                                          locLogsTrain["locLoss_train"][t] / totalSteps, step=epoch)
                    experiment.log_metric(f"Accuracy/epoch/locAcc_train_{t}",
                                          locLogsTrain["locAcc_train"][t] / totalSteps, step=epoch)
                    experiment.log_metric(f"Losses/epoch/locLoss_val_{t}", locLogsVal["locLoss_val"][t] / totalSteps,
                                          step=epoch)
                    experiment.log_metric(f"Accuracy/epoch/locAcc_val_{t}", locLogsVal["locAcc_val"][t] / totalSteps,
                                          step=epoch)

            if currentAccuracy > bestAcc:
                bestStateDict = cpcModel.state_dict()

            for key, value in dict(locLogsTrain, **locLogsVal).items():
                if key not in logs:
                    logs[key] = [None for _ in range(epoch)]
                if isinstance(value, np.ndarray):
                    value = value.tolist()
                logs[key].append(value)

            logs["epoch"].append(epoch)

            if pathCheckpoint is not None and (epoch % logs["saveStep"] == 0 or epoch == nEpoch - 1):
                modelStateDict = cpcModel.state_dict()
                criterionStateDict = cpcCriterion.state_dict()

                saveCheckpoint(modelStateDict, criterionStateDict, optimizer.state_dict(), bestStateDict,
                               f"{pathCheckpoint}_{epoch}.pt")
                saveLogs(logs, pathCheckpoint + "_logs.json")
    except KeyboardInterrupt:
        if pathCheckpoint is not None:
            modelStateDict = cpcModel.state_dict()
            criterionStateDict = cpcCriterion.state_dict()

            saveCheckpoint(modelStateDict, criterionStateDict, optimizer.state_dict(), bestStateDict,
                           f"{pathCheckpoint}_{epoch}_interrupted.pt")
            saveLogs(logs, pathCheckpoint + "_logs.json")
        return


def run(trainDataset,
        valDataset,
        batchSize,
        samplingMode,
        cpcModel,
        cpcCriterion,
        nEpoch,
        optimizer,
        pathCheckpoint,
        logs,
        useGPU,
        log2Board=0,
        experiment=None):
    if log2Board:
        with experiment.train():
            trainingLoop(trainDataset, valDataset, batchSize, samplingMode, cpcModel, cpcCriterion, nEpoch, optimizer,
                         pathCheckpoint, logs, useGPU, log2Board, experiment)
            experiment.end()
    else:
        trainingLoop(trainDataset, valDataset, batchSize, samplingMode, cpcModel, cpcCriterion, nEpoch, optimizer,
                     pathCheckpoint, logs, useGPU, log2Board, experiment)
