EPOCHS = 1000
H = 100
LR = 0.001

data_train = CSVDataset('data/train.csv')
data_train_loader = data.DataLoader(data_train, shuffle=True, batch_size=16)
data_test = CSVDataset('data/test.csv')
data_test_loader = data.DataLoader(data_test, shuffle=True, batch_size=1)

import pandas as pd
import hyperparameters as hyp
import torch.utils.data as data
import torch.nn as nn
import torch
from tqdm import tqdm
import torch.optim as optim
import matplotlib.pyplot as plt

class CSVDataset(data.Dataset):
    # exclude some fields that are non-numeric
    def __init__(self, file_path):
        df = pd.read_csv(file_path)
        df = df.drop(['Name', 'RescuerID', 'PetID', 'Description'], axis=1)
        self.data = torch.FloatTensor(df.values).cuda()
        
    """  
    (  tensor([[    2,     3,   299,     0,     1,     1,     7,     0,     1,     1,
                  2,     2,     2,     1,     1,   100, 41326,     0,     1]],   dtype=torch.int32)  ,
       tensor([2], dtype=torch.int32)  )
    """  
    def __getitem__(self, index):
        label = self.data[index][-1].long().cuda()
        return (self.data[index][:-1], label)

    def __len__(self):
        return len(self.data)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_model = nn.Sequential(torch.nn.Linear(19, H),
                                        torch.nn.ReLU(),
                                        torch.nn.Linear(H, H))
        self.classification_layer = nn.Sequential(torch.nn.ReLU(),
                                                  torch.nn.Linear(H, 5))

    def forward(self, input):
        # input : 19 x 1 matrix
        hidden = self.base_model.forward(input)
        output = self.classification_layer(hidden)
        # output : 5 adoption speeds
        return output

    def train(self):
        self.base_model.train()

    def eval(self):
        self.base_model.eval()

model = Model().cuda()

for epoch in range(EPOCHS):  # trains the NN 1,000 times
    print("Training...")
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    loss = nn.CrossEntropyLoss()
    loss = loss.cuda()
    epoch = 0
    store_epoch_loss = []
    store_epoch_loss_val = []
    store_epoch_acc_val = []
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(EPOCHS/2), int(0.75*EPOCHS)], gamma=0.1)
    try:
        for e in tqdm(range(EPOCHS)):
            scheduler.step()
            epoch = e + 1
            epoch_loss = 0
            store_batch_loss = []
            for batch_num, (X, y) in enumerate(data_train_loader):
                optimizer.zero_grad()
                prediction = model.forward(X)
                batch_loss = loss(prediction, y)
                batch_loss.backward()
                optimizer.step()
                store_batch_loss.append(batch_loss.clone().cpu())
                epoch_loss = torch.FloatTensor(store_batch_loss).mean()
            store_epoch_loss.append(epoch_loss)
            torch.save(model.state_dict(), "dcheckpoint_{}.pth".format(epoch))
            plt.plot(store_epoch_loss[1:], label="Training Loss")

            model.eval()

            plt.legend()
            plt.grid()
            plt.savefig("Loss.png".format())
            plt.close()
            model.train()
        most_acc = max(store_epoch_acc_val)
        min_loss = min(store_epoch_loss_val)
        print("\nHighest accuracy of {} occured at {}%...Minimum loss occured at {}%...".format(most_acc, store_epoch_acc_val.index(most_acc)+1, store_epoch_loss_val.index(min_loss)+1))
        t = Timer(3*60, interrupt_handler, ["checkpoint_{}.pth".format(store_epoch_loss_val.index(min_loss)+1), "{}/checkpoint_{}.pth".format(training_dir, store_epoch_acc_val.index(most_acc)+1)])
        t.start()
        user_pick = input("Which checkpoint do you want to use ?\n")
        t.cancel()
        model.load_state_dict(torch.load("checkpoint_{}.pth".format(user_pick)))
    except KeyboardInterrupt:
        most_acc = max(store_epoch_acc_val)
        min_loss = min(store_epoch_loss_val)
        print("\nHighest accuracy of {} occured at {}%...Minimum loss occured at {}%...".format(most_acc, store_epoch_acc_val.index(most_acc)+1, store_epoch_loss_val.index(min_loss)+1))
        user_pick = input("Which checkpoint do you want to use ?\n")
        model.load_state_dict(torch.load("checkpoint_{}.pth".format(user_pick)))