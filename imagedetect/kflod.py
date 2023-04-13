from trainer import Trainer
from preprocessing import get_dataset
from os import path
from torch.utils.data import DataLoader
import torch as T
import numpy as np
import random
from sklearn import metrics
from os import path

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
        print(pred_intervals)
            # Extract unique class labels from the intervals
        unique_labels = sorted(list(set(tuple(interval) for interval in pred_intervals)))

        # Initialize confusion matrix as nested dictionaries
        matrix = {label: {label: 0 for label in unique_labels} for label in unique_labels}

        # Update confusion matrix
        for i in range(len(target)):
            matrix[tuple(target[i].tolist())][tuple(pred_intervals[i])] += 1

        return matrix
    

    # Get confusion matrix 
    # Change target to numpy array with int
    target = target.cpu().numpy().astype(int)

    matrixInterval = get_confusion_matrix_intervals(pred, target,level=0.8)
    matrixInterval50  = get_confusion_matrix_intervals(pred, target,level=0.5)
    print(matrixInterval)
    print(matrixInterval50)

    #pred, target = tr.predict()
    #all_pred[i:i+pred.shape[0]] = pred
    #all_targets[i:i+target.shape[0]] = target.view(-1)
    #i += target.shape[0]

    #prec, recall, auc = get_metrics(target, pred)
    #print(f'AUC: {auc}, precision: {prec}, Recall: {recall}')
    #f.write(f'AUC: {auc}, precision: {prec}, Recall: {recall}\n')

    del tr


    #matches = calc_accuracy(all_pred, all_targets)
    #acc = matches.float().mean()
    #all_pred = all_pred.numpy()
    #all_targets = all_targets.numpy()

    #prec, recall, auc = get_metrics(all_targets, all_pred)
    #print(f'Accuracy: {acc}, AUC: {auc}, Precision: {prec}, Recall: {recall}')
    #f.write(f'Accuracy: {acc}, AUC: {auc}, Precision: {prec}, Recall: {recall}')
    #result = {'all_pred': all_pred, 'all_targets': all_targets}
    #T.save(result, path.join('results',f'{name}_result'))
    #f.close()



