import torch as T
import torch.nn as nn
import numpy as np
import time
from sklearn import metrics
from os import path
import wandb
'''
Trainer class for training models
'''
class Trainer:
    def __init__(self, training_set,
                 validation_set,
                 batch_size,
                 n_epochs,
                 model,
                 optimizer,
                 loss,
                 name,
                 device='cuda',
                 deterministic=False,
                 parallel=False
                 ):

        T.backends.cudnn.deterministic = deterministic
        self.batch_size = batch_size
        self.dataset = training_set
        self.valid_dataset = validation_set

        self.model = model
        self.device = device
        self.parallel = parallel
        if parallel:
            self.model = nn.DataParallel(model)
        self.model.cuda(self.device)
        self.optimizer = optimizer
        self.loss = loss
        self.n_epochs = n_epochs
        self.name = name
        self.log = ''
        wandb.init(project='lidc-idri', config={
            "batch_size": batch_size,
            "n_epochs": n_epochs,
            "model": model,
            "optimizer": optimizer

        })

    # Train a single epoch  
    def train_epoch(self, epoch):
        s_time = time.time()
        self.model.train()
        all_losses = []
        all_acc = []
        outputs = []
        targets = []
        for data, target in self.dataset:
            data, target = data.cuda(self.device), target.cuda(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            # check if target 
            acc = self.calc_accuracy(output, target)
            #print(target.shape)
            target = T.squeeze(target, 1).long()
            loss = self.loss(output, target)
            loss.backward()
            self.optimizer.step()
            all_losses.append(loss.item())
            all_acc.append(acc.cpu())
            outputs.extend(output.softmax(dim=1).detach().cpu().numpy())
            targets.extend(target.detach().cpu().numpy())



        valid_acc = self.validate()
        # auc 
        auc = metrics.roc_auc_score(targets, outputs, multi_class='ovr')
        # Get training accuracy
        tr_accuacy = T.mean(T.tensor(all_acc))
        print("Training Accuracy: ")
        print(tr_accuacy)
        print("\n Validation Accuracy: ")
        print(valid_acc)
        # self.report(all_losses, all_acc, valid_acc, epoch, time.time() - s_time)
        # Calculate accuracy for each class based on the argmax of the output
        
        
        wandb.log({"loss": np.sum(all_losses) / len(all_losses), 
                   "epoch": epoch,
                    "train_acc": tr_accuacy,
                    "valid_acc": valid_acc,
                    "duration": time.time() - s_time,
                    "auc": auc
                   })

    # Predict on the validation set
    def predict(self):
        self.model.eval()
        all_pred = T.zeros(len(self.valid_dataset.dataset), 3)
        all_targets = T.zeros(len(self.valid_dataset.dataset))
        for batch_idx, (data, target) in enumerate(self.valid_dataset):
            with T.no_grad():
                data, target = data.cuda(self.device), target.cuda(self.device)
                output = self.model(data)
            st = batch_idx * self.batch_size

            all_pred[st:st + output.shape[0]] = output.cpu()
            all_targets[st:st + output.shape[0]] = target.squeeze().long().cpu()

        
        all_targets = all_targets.view(-1, 1)
        # print(all_pred, all_targets)
        return all_pred, all_targets

    # Validate the model on the validation set
    def validate(self):
        all_pred, all_targets = self.predict()
        # print(all_pred, all_targets)
        matches = self.calc_accuracy(all_pred, all_targets)
        return matches
    
    # Calculate the accuracy of the model
    # Input: output - the output of the model
    #        target - the target of the model
    # Output: accuracy - the accuracy of the model
    def calc_accuracy(self, output, target):
        # Check that the argmax of softmax is the same as the target
        #print(output.data)
        predicted = T.argmax(output, dim=1)
        # print(predicted, target)
        # compare with the target tensor to get a tensor of correct predictions
        correct = (predicted == target.to(predicted.dtype))
        # calculate the accuracy as the percentage of correct predictions
        accuracy = correct.float().mean()
        return accuracy
    # Run the training
    def run(self):
        start_t = time.time()
        for epoch in range(self.n_epochs):
            self.train_epoch(epoch)
        diff = time.time() - start_t
        print(f'took {diff} seconds')
        with open(path.join('results',f'{self.name}.txt'),'w') as f:
            f.write(self.log)


