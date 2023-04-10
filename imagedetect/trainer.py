import torch as T
import torch.nn as nn
import numpy as np
import time
from sklearn import metrics
from os import path
import wandb

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


    def train_epoch(self, epoch):
        s_time = time.time()
        self.model.train()
        all_losses = []
        all_acc = []
        for data, target in self.dataset:
            data, target = data.cuda(self.device), target.cuda(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            # check if target 
            acc = self.calc_accuracy(output, target)
            print(target.shape)
            target = T.squeeze(target, 1).long()
            loss = self.loss(output, target)
            loss.backward()
            self.optimizer.step()
            all_losses.append(loss.item())
            all_acc.append(acc.cpu())

        valid_acc = self.validate()
        
        # Get training accuracy
        tr_accuacy = T.mean(all_acc)
        # self.report(all_losses, all_acc, valid_acc, epoch, time.time() - s_time)
        # Calculate accuracy for each class based on the argmax of the output
        
        
        wandb.log({"loss": np.sum(all_losses) / len(all_losses), 
                   "epoch": epoch,
                    "train_acc": tr_accuacy,
                    "valid_acc": valid_acc,
                    "duration": time.time() - s_time
                   })

    def report(self, all_losses, all_acc, valid_acc, epoch, duration):
        n_train = len(all_losses)
        loss = np.sum(all_losses) / n_train

        def summery(data):
            n = 0.0
            s_dist = 0
            for dist in data:
                s_dist += T.sum(dist)
                n += len(dist)

            return s_dist.float() / n

        tr_dist = summery(all_acc)
        va_dist = summery(valid_acc)

        pred, target = self.predict()
        # print(pred, target)
        # auc = metrics.auc(fpr, tpr)

        msg = f'epoch {epoch}: loss {loss:.3f} Tr Acc {tr_dist:.2f} Val Acc {va_dist:.2f} AUC {3:.2f} duration {duration:.2f}'
        print(msg)
        self.log += msg + '\n'


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

        all_pred = all_pred.view(-1, 3).mean(dim=0)
        all_targets = all_targets.view(-1, 1)
        return all_pred, all_targets


    def validate(self):
        all_pred, all_targets = self.predict()
        matches = self.calc_accuracy(all_pred, all_targets)
        return [matches]

    def calc_accuracy(self, output, target):
        # Check that the argmax of softmax is the same as the target
        #print(output.data)
        _, predicted = T.max(output.data, 1)
        # compare with the target tensor to get a tensor of correct predictions
        correct = (predicted == target)
        # calculate the accuracy as the percentage of correct predictions
        accuracy = correct.float().mean()
        print(accuracy)
        return accuracy

    def run(self):
        start_t = time.time()
        for epoch in range(self.n_epochs):
            self.train_epoch(epoch)
        diff = time.time() - start_t
        print(f'took {diff} seconds')
        with open(path.join('results',f'{self.name}.txt'),'w') as f:
            f.write(self.log)


