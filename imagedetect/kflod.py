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

def reset_rand():
    seed = 1000
    T.manual_seed(seed)
    T.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

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
    all_pred = T.zeros(849)
    all_targets = T.zeros(849)
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
    def get_confusion_matrix_intervals(pred, target,level=0.8):

        # Get the prediction intervals for each sample
        pred_intervals = [multi_class_prediction_intervals(p, level=level) for p in pred]
        
            # Extract unique class labels from the intervals
        unique_labels = sorted(list(set(tuple(interval) for interval in pred_intervals)))

        # Initialize confusion matrix as nested dictionaries
        matrix = defaultdict(lambda: defaultdict(int))

        # Update confusion matrix
        for i in range(len(target)):
            #print(target[i])
            # print(pred_intervals[i])
            matrix[tuple(list(target[i]))][tuple(pred_intervals[i])] += 1


            # Convert the nested defaultdict to a Pandas DataFrame
        # Convert to Pandas DataFrame
        #df_matrix = pd.DataFrame(matrix).transpose().fillna(0).astype(int)

        # Convert to numpy array
        #confusion_matrix = df_matrix.values

        return matrix
    # Function to calculate the coverage rate of the prediction intervals (the percentage of samples that are covered by the prediction interval)
    def coverage_rate(pred_intervals, target):
        # Calculate the coverage rate
        coverage = sum([1 if int(target[i]) in pred_intervals[i] else 0 for i in range(len(target))]) / len(target)

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



