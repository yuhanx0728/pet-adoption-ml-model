import pandas as pd
import torch
import torch.utils.data as data
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
import matplotlib.pyplot as plt
import random
import numpy as np
import os, sys, time

EPOCHS = 1000
Hidden = [100, 200, 200, 100]
LR = 0.001

class CSVDataset(data.Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        label = self.data[index][-1].long().cuda()
        return (self.data[index][:-1], label)

    def __len__(self):
        return len(self.data)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_model = nn.Sequential(torch.nn.Linear(19, Hidden[0]),
                                        torch.nn.ReLU(),
                                        torch.nn.Linear(Hidden[0], Hidden[1]))
        self.classification_layer1 = nn.Sequential(torch.nn.ReLU(),
                                                  torch.nn.Linear(Hidden[1], Hidden[2]))
        self.classification_layer2 = nn.Sequential(torch.nn.ReLU(),
                                                  torch.nn.Linear(Hidden[2], Hidden[3]))
        self.classification_layer3 = nn.Sequential(torch.nn.ReLU(),
                                                  torch.nn.Linear(Hidden[3], 5))

        
    def forward(self, input):
        # input : 19 x 1 matrix
        hidden1 = self.base_model.forward(input)
        hidden2 = self.classification_layer1(hidden1)
        hidden3 = self.classification_layer2(hidden2)
        output = self.classification_layer3(hidden3)
        # output : 5 adoption speeds
        return output


df = pd.read_csv('data/train.csv')
df = df.drop(['Name', 'RescuerID', 'PetID', 'Description'], axis=1)
d = torch.FloatTensor(df.values).cuda()
random.shuffle(d)
partition = {}
validation = d[:len(d)//10]
partition['validation'] = CSVDataset(validation)
train_set = d[len(d)//10:]
partition['train'] = CSVDataset(train_set)
data_train_loader = data.DataLoader(partition['train'], shuffle=True, batch_size=32)
data_val_loader = data.DataLoader(partition['validation'], batch_size=32)

# data_test = CSVDataset('data/test.csv')
# data_test_loader = data.DataLoader(data_test, shuffle=True, batch_size=1)

def train(model):
    print("Training...")
    training_dir = 'training_{}'.format(time.time)
    os.mkdir(training_dir)
	os.mkdir(training_dir+'/misclassified')
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    loss = nn.CrossEntropyLoss().cuda()
    epoch = 0
    store_epoch_loss = []
    store_epoch_loss_val = []
    store_epoch_acc_val = []
#    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(EPOCHS/2), int(0.75*EPOCHS)], gamma=0.1)
    try:
        for e in tqdm(range(EPOCHS)):
            #scheduler.step()
            epoch = e + 1
            epoch_loss = 0
            store_batch_loss = []
            
            for batch_num, (X, y) in enumerate(data_train_loader):
                optimizer.zero_grad()
                if batch_num==1:
                    print(X,y)
                prediction = model.forward(X.cuda())
                batch_loss = loss(prediction, y.cuda())
                batch_loss.backward()
                optimizer.step()
                store_batch_loss.append(batch_loss.clone().cpu())
                epoch_loss = torch.FloatTensor(store_batch_loss).mean()
            store_epoch_loss.append(epoch_loss)
            torch.save(model.state_dict(), "{}/checkpoint_{}.pth".format(training_dir, epoch))
            plt.plot(store_epoch_loss[1:], label="Training Loss")

            model.eval()
            epoch_loss_val = 0
            epoch_acc_val = 0
            store_batch_loss_val = []
            store_batch_acc_val = []
            misclassified_images = []
            for batch_num, (X, y) in enumerate(data_val_loader):
                with torch.no_grad():
                    prediction = model.forward(X.cuda())
                batch_loss = loss(prediction, y.cuda())
                misclassified = prediction.max(-1)[-1].squeeze().cpu() != y.cpu()
                misclassified_images.append(X[misclassified==1].cpu())
                batch_acc = misclassified.float().mean()
                store_batch_loss_val.append(batch_loss.cpu())
                store_batch_acc_val.append(batch_acc)
                epoch_loss_val = torch.FloatTensor(store_batch_loss_val).mean()
                epoch_acc_val = torch.FloatTensor(store_batch_acc_val).mean()
            store_epoch_loss_val.append(epoch_loss_val)
            store_epoch_acc_val.append(1-epoch_acc_val)
            plt.plot(store_epoch_loss_val[1:], label="Validation Loss")
            plt.plot(store_epoch_acc_val[1:], label="Validation Accuracy")
            plt.legend()
            plt.grid()
            plt.savefig("{}/Loss.png".format(training_dir))
            plt.close()
            if len(misclassified_images) > 0:
                misclassified_images = np.concatenate(misclassified_images,axis=0)
                validation_dir = training_dir+'/misclassified/checkpoint_{}'.format(epoch)
                os.mkdir(validation_dir)
            model.train()
        most_acc = max(store_epoch_acc_val)
        min_loss = min(store_epoch_loss_val)
        print("\nHighest accuracy of {} occured at {}%...Minimum loss occured at {}%...".format(most_acc, store_epoch_acc_val.index(most_acc)+1, store_epoch_loss_val.index(min_loss)+1))
        t = Timer(3*60, interrupt_handler, ["{}/checkpoint_{}.pth".format(training_dir, store_epoch_loss_val.index(min_loss)+1), "{}/checkpoint_{}.pth".format(training_dir, store_epoch_acc_val.index(most_acc)+1)])
        t.start()
        user_pick = input("Which checkpoint do you want to use ?\n")
        t.cancel()
        model.load_state_dict(torch.load("{}/checkpoint_{}.pth".format(training_dir, user_pick)))
    except KeyboardInterrupt:
        most_acc = max(store_epoch_acc_val)
        min_loss = min(store_epoch_loss_val)
        print("\nHighest accuracy of {} occured at {}%...Minimum loss occured at {}%...".format(most_acc, store_epoch_acc_val.index(most_acc)+1, store_epoch_loss_val.index(min_loss)+1))
        user_pick = input("Which checkpoint do you want to use ?\n")
        model.load_state_dict(torch.load("{}/checkpoint_{}.pth".format(training_dir, user_pick)))

    return model.cuda(), training_dir

if __name__ == "__main__":
        train(Model().cuda())