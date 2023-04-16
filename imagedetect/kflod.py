from trainer import Trainer
from preprocessing import get_dataset
from os import path
from torch.utils.data import DataLoader
import torch as T
import numpy as np
import random
from sklearn import metrics
from os import path
from collections import defaultdict
import pandas as pd
def get_metrics(target, pred):
    prec, recall, _, _ = metrics.precision_recall_fscore_support(target, pred, average='weighted')
    fpr, tpr, thresholds = metrics.roc_curve(target, pred)
    auc = metrics.auc(fpr, tpr)
    return prec, recall, auc


def calc_accuracy(x, y):
    # Check if the argmax index of the output matches the target
    matches = (x.argmax(dim=1) == y)
    return matches
# Reset random seed
def reset_rand():
    seed = 1000
    T.manual_seed(seed)
    T.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
# Training function
def kfold(src_path,
          batch_size,
          n_epochs,
          model_optimizer,
          loss,
          name,
          device,
          deterministic=False,
          parallel=False,
          dataset_func=get_dataset):

    print(f'Experiment {name}')
    i = 0
    f = open(path.join('results', f'{name}.txt'), 'w')
    f.write(f'{batch_size} {n_epochs} {model_optimizer}\n')
    

    reset_rand()

    # print(f'------------ fold {fold+1} ------------')
    # f.write(f'------------ fold {fold+1} ------------\n')
    trset, testset = dataset_func(path.join(src_path))
    print(f'Training Size: {len(trset)}, Validation Size: {len(testset)}')
    trset = DataLoader(trset, batch_size, shuffle=True)
    testset = DataLoader(testset, batch_size, shuffle=False)
    model,optimizer = model_optimizer()
    tr = Trainer(
        trset,
        testset,
        batch_size,
        n_epochs,
        model,
        optimizer,
        loss,
        f'{name}',
        device,
        deterministic,
        parallel
    )

    tr.run() 
    # run prediction on test set and get metrics
    pred, target = tr.predict()
    # get argmax of predictions
    preds = pred.argmax(dim=1)
    matrix = metrics.confusion_matrix(target, preds)
    print(matrix)
    # Input: probability_vector - vector of probabilities for each class
    #        level - prediction interval level
    # Output: interval - prediction interval
    def multi_class_prediction_intervals(probability_vector, level=0.8):
        # Sort categories and probabilities in descending order of probabilities
        category_labels = np.arange(len(probability_vector))
        sorted_indices = np.argsort(-probability_vector)
        sorted_labels = category_labels[sorted_indices]
        sorted_probabilities = probability_vector[sorted_indices]

        # Calculate cumulative probabilities
        cumulative_probabilities = np.cumsum(sorted_probabilities)

        # Find the prediction interval
        k = np.argmax(cumulative_probabilities >= level)
        interval = sorted_labels[: k + 1]

        return sorted(interval.tolist())
    # Get confusion matrix with 80% prediction interval
    # Input: pred - list of predictions
    #        target - list of target values
    #        level - prediction interval level
    # Output: matrix - confusion matrix with prediction intervals

    def get_confusion_matrix_intervals(pred, target,level=0.8):

        # Get the prediction intervals for each sample
        pred_intervals = [multi_class_prediction_intervals(p, level=level) for p in pred]
        
        # Initialize confusion matrix as nested dictionaries
        matrix = defaultdict(lambda: defaultdict(int))

        # Update confusion matrix
        for i in range(len(target)):
            matrix[tuple(list(target[i]))][tuple(pred_intervals[i])] += 1
        return matrix
    # Function to calculate the coverage rate of the prediction intervals (the percentage of samples that are covered by the prediction interval)
    # Input: pred_intervals - list of prediction intervals
    #        target - list of target values
    # Output: coverage - list of coverage rates for each class
    def coverage_rate(pred_intervals, target):
        # Calculate the coverage rate
        # GO through each of the unique classes and calculate the coverage rate for each class
        coverage = []
        # get all target values that are 0, and z
        coverage0 = 0
        coverage1 = 0
        coverage2 = 0
        for i in range(len(target)):
            if target[i] == 0:
                if 0 in pred_intervals[i]:
                    coverage0 += 1
            if target[i] == 1:
                if 1 in pred_intervals[i]:
                    coverage1 += 1
            if target[i] == 2:
                if 2 in pred_intervals[i]:
                    coverage2 += 1
        coverage.append(coverage0 / np.count_nonzero(target == 0))
        coverage.append(coverage1 / np.count_nonzero(target == 1))
        coverage.append(coverage2 / np.count_nonzero(target == 2))

        return coverage
    

    # Get confusion matrix 
    # Change target to numpy array with int
    
    target = target.cpu().numpy().astype(int)
    print(coverage_rate([multi_class_prediction_intervals(p, level=.5) for p in pred], target))
    print(coverage_rate([multi_class_prediction_intervals(p, level=.8) for p in pred], target))
    print(coverage_rate([multi_class_prediction_intervals(p, level=.9) for p in pred], target))
    matrixInterval = get_confusion_matrix_intervals(pred, target,level=0.8)
    matrixInterval = get_confusion_matrix_intervals(pred, target,level=0.)
    matrixInterval50  = get_confusion_matrix_intervals(pred, target,level=0.5)
    print("80% prediction interval")
    print(matrixInterval)
    print("50% prediction interval")
    print(matrixInterval50)



    del tr



