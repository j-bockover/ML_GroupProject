import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import time
from networks.VGG import *
from sklearn.metrics import roc_auc_score, roc_curve

class VGG_Trainer():
    def __init__(self, train_set, test_set, lr = 1e-4, epochs=100,
                 batch_size=32, weight_decay=1e-6) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.weight_decay = weight_decay
        self.lr_milestones = [int(epochs*0.5),int(epochs*0.75)]

        self.train_set = train_set
        self.test_set = test_set

        self.model = VGG16().to(self.device)

    def train(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        criterion = nn.CrossEntropyLoss()
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.lr_milestones, gamma=0.1)
        print('----- Start Training ... -----')
        start_time = time.time()
        self.model.train()

        for epoch in range(self.epochs):
            epoch_loss = 0.0
            n_batches = 0
            correct, total = 0, 0
            epoch_start_time = time.time()

            for data in self.train_set:
                inputs, targets = data
                inputs, targets = inputs.to(self.device).float(), targets.to(self.device).long()
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                epoch_loss += loss
                n_batches += 1

                _, pred = torch.max(outputs, dim=1)
                correct += (pred == targets).sum()
                total += pred.size(0)
            scheduler.step()
            if epoch in self.lr_milestones:
                print('  LR scheduler: new learning rate is %g' % float(scheduler.get_lr()[0]))
            epoch_train_time = time.time() - epoch_start_time
            print(f'     | Epoch: {epoch + 1:03}/{self.epochs:03} | Train Time: {epoch_train_time:.3f}s '
                  f'| Train ACC: {correct/total:.6f} | Train Loss: {epoch_loss / n_batches:.6f} |')
            self.test()
        print('----- Finished training. Time: {:.3f}s -----'.format(time.time() - start_time))
    
    def test(self):
        # print('----- Start testing... -----')
        # epoch_loss = 0.0
        # n_batches = 0
        correct, total = 0, 0
        
        start_time = time.time()
        self.model.eval()

        with torch.no_grad():
            for data in self.test_set:
                inputs, targets = data
                inputs = inputs.to(self.device).float()
                targets = targets.to(self.device).long()
                outputs = self.model(inputs)
                _, pred = torch.max(outputs, dim=1)
                correct += (pred == targets).sum()
                total += pred.size(0)
        test_time = time.time() - start_time
        # print('Test ACC: {:.5f}'.format(correct/total))
        # print('Test Time: {:.3f}s'.format(test_time))
        print(f'     | Test Time: {test_time:.3f}s | Test ACC: {correct/total:.6f} | ')