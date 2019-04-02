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
from sklearn.metrics import cohen_kappa_score

EPOCHS = 1000
LR = 0.0001
HIDDEN_LIST = [ [125, 250, 250, 125],
                [250, 500, 500, 250],
                [500, 1000, 1000, 500],
                [125, 250, 500, 250, 125],
                [250, 500, 1000, 500, 250],
                [125, 250, 500, 500, 250, 125],
                [250, 500, 1000, 1000, 500, 250] ]

# state GDP: https://en.wikipedia.org/wiki/List_of_Malaysian_states_by_GDP
state_gdp = {
    41336: 116.679,
    41325: 40.596,
    41367: 23.02,
    41401: 190.075,
    41415: 5.984,
    41324: 37.274,
    41332: 42.389,
    41335: 52.452,
    41330: 67.629,
    41380: 5.642,
    41327: 81.284,
    41345: 80.167,
    41342: 121.414,
    41326: 280.698,
    41361: 32.270
}

# state population: https://en.wikipedia.org/wiki/Malaysia
state_population = {
    41336: 33.48283,
    41325: 19.47651,
    41367: 15.39601,
    41401: 16.74621,
    41415: 0.86908,
    41324: 8.21110,
    41332: 10.21064,
    41335: 15.00817,
    41330: 23.52743,
    41380: 2.31541,
    41327: 15.61383,
    41345: 32.06742,
    41342: 24.71140,
    41326: 54.62141,
    41361: 10.35977
}

state_area ={
    41336:19102,
    41325:9500,
    41367:15099,
    41401:243,
    41415:91,
    41324:1664,
    41332:6686,
    41335:36137,
    41330:21035,
    41380:821,
    41327:1048,
    41345:73631,
    41342:124450,
    41326:8104,
    41361:13035
}


class CSVDataset(data.Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        label = self.data[index][-1].long().cuda()
        return (self.data[index][:-1], label)

    def __len__(self):
        return len(self.data)


class Model(nn.Module):
    def __init__(self, HIDDEN, ONE_HOT, DATA_AUG):
        super().__init__()
        self.HIDDEN = HIDDEN
        if ONE_HOT:
            if DATA_AUG:
                initial = 664
            else:
                initial = 753
        else:
            if DATA_AUG:
                initial = 655
            else:
                initial = 744
        self.base_model = nn.Sequential(torch.nn.Linear(initial, self.HIDDEN[0]),
                                        torch.nn.ReLU(),
                                        torch.nn.Linear(self.HIDDEN[0], self.HIDDEN[1]))
        self.classification_layer1 = nn.Sequential(torch.nn.ReLU(),
                                                   torch.nn.Linear(self.HIDDEN[1], self.HIDDEN[2]))
        self.classification_layer2 = nn.Sequential(torch.nn.ReLU(),
                                                   torch.nn.Linear(self.HIDDEN[2], self.HIDDEN[3]))
        self.output_layer = nn.Sequential(torch.nn.ReLU(),
                                         torch.nn.Linear(HIDDEN[-1], 5))
        if (len(self.HIDDEN) == 5):
            self.classification_layer3 = nn.Sequential(torch.nn.ReLU(),
                                                       torch.nn.Linear(self.HIDDEN[3], self.HIDDEN[4]))
        elif (len(self.HIDDEN) == 6):
            self.classification_layer3 = nn.Sequential(torch.nn.ReLU(),
                                                       torch.nn.Linear(self.HIDDEN[3], self.HIDDEN[4]))
            self.classification_layer4 = nn.Sequential(torch.nn.ReLU(),
                                                       torch.nn.Linear(self.HIDDEN[4], self.HIDDEN[5]))
    def forward(self, input):
        # input : some number x some number matrix
        hidden1 = self.base_model.forward(input)
        hidden2 = self.classification_layer1(hidden1)
        hidden3 = self.classification_layer2(hidden2)
        if (len(self.HIDDEN) == 4):
            output = self.output_layer(hidden3)
        elif (len(self.HIDDEN) == 5):
            hidden4 = self.classification_layer3(hidden3)
            output = self.output_layer(hidden4)
        elif (len(self.HIDDEN) == 6):
            hidden4 = self.classification_layer3(hidden3)
            hidden5 = self.classification_layer4(hidden4)
            output = self.output_layer(hidden5)
        # output : 5 adoption speeds
        return output
    
def prep_dataset(ONE_HOT, DATA_AUG):
    df = pd.read_csv('data/train.csv')
    df = df.drop(['Name', 'RescuerID', 'PetID', 'Description'], axis=1)
    d = torch.FloatTensor(df.values)

    # Type,Name,Age,Breed1,Breed2,Gender,Color1,Color2,Color3,MaturitySize,
    #    0,   x,  1,     2,     3,     4,     5,     6,     7,           8,
    # FurLength,Vaccinated,Dewormed,Sterilized,Health,Quantity,Fee,State,RescuerID,
    #         9,        10,      11,        12,    13,      14, 15,   16,        x,      
    # VideoAmt,Description,PetID,PhotoAmt,AdoptionSpeed
    #       17,          x,    x,      18,           19

    # df['Type']
    nType = np.array([[0.5,0.5]]*d.size(0)).astype(float)
    nType[df['Type'].values.astype(int)==1] = [0.,1.]
    nType[df['Type'].values.astype(int)==2] = [1.,0.]

    # df['Breed1']
    idx = d[:,2]
    nBreed1 = torch.zeros(len(idx), int(idx.max())+1).scatter_(1, idx.view(len(idx)).long().unsqueeze(1), 1.)[:,1:]

    # df['Breed2']
    idx = d[:,3]
    nBreed2 = torch.zeros(len(idx), int(idx.max())+1).scatter_(1, idx.view(len(idx)).long().unsqueeze(1), 1.)[:,1:]

    # df['Gender']
    nGender = np.array([[0.5,0.5]]*d.size(0)).astype(float)
    nGender[df['Gender'].values.astype(int)==1] = [0.,1.]
    nGender[df['Gender'].values.astype(int)==2] = [1.,0.]

    # df['Color1']
    idx = d[:,5]
    nColor1 = torch.zeros(len(idx), int(idx.max())+1).scatter_(1, idx.view(len(idx)).long().unsqueeze(1), 1.)[:,1:]

    # df['Color2']
    idx = d[:,6]
    nColor2 = torch.zeros(len(idx), int(idx.max())+1).scatter_(1, idx.view(len(idx)).long().unsqueeze(1), 1.)[:,1:]

    # df['Color3']
    idx = d[:,7]
    nColor3 = torch.zeros(len(idx), int(idx.max())+1).scatter_(1, idx.view(len(idx)).long().unsqueeze(1), 1.)[:,1:]

    # df['Vaccinated']
    nVaccinated = np.array([[0.5,0.5]]*d.size(0)).astype(float)
    nVaccinated[df['Vaccinated'].values.astype(int)==1] = [0.,1.]
    nVaccinated[df['Vaccinated'].values.astype(int)==2] = [1.,0.]

    # df['Dewormed']
    nDewormed = np.array([[0.5,0.5]]*d.size(0)).astype(float)
    nDewormed[df['Dewormed'].values.astype(int)==1] = [0.,1.]
    nDewormed[df['Dewormed'].values.astype(int)==2] = [1.,0.]

    # df['Sterilized']
    nSterilized = np.array([[0.5,0.5]]*d.size(0)).astype(float)
    nSterilized[df['Sterilized'].values.astype(int)==1] = [0.,1.]
    nSterilized[df['Sterilized'].values.astype(int)==2] = [1.,0.]

    # df['State']
    idx = d[:,16]
    nState = torch.zeros(len(idx), int(idx.max()-idx.min())+1).scatter_(1, idx.view(len(idx)).long().unsqueeze(1)-idx.min(), 1.)

    if DATA_AUG:
        state_data = {}
        for k, v in state_gdp.items():
            state_data[k] = np.array([v, state_population[k], state_area[k]]).astype(float)

        nState = np.array([[0, 0, 0]]*d.size(0)).astype(float)
        for k,v in state_data.items():
            nState[df['State'].values.astype(int)==k] = v

    if ONE_HOT:
        idx = d[:,8]
        nMaturitySize = torch.zeros(len(idx), int(idx.max())+1).scatter_(1, idx.view(len(idx)).long().unsqueeze(1), 1.)[:,1:]
        idx = d[:,9]
        nFurLength = torch.zeros(len(idx), int(idx.max())+1).scatter_(1, idx.view(len(idx)).long().unsqueeze(1), 1.)[:,1:]
        idx = d[:,13]
        nHealth = torch.zeros(len(idx), int(idx.max())+1).scatter_(1, idx.view(len(idx)).long().unsqueeze(1), 1.)[:,1:]
        d = torch.cat([torch.FloatTensor(nType),
                       d[:,1:2],
                       torch.FloatTensor(nBreed1),
                       torch.FloatTensor(nBreed2),
                       torch.FloatTensor(nGender),
                       torch.FloatTensor(nColor1),
                       torch.FloatTensor(nColor2),
                       torch.FloatTensor(nColor3),
                       torch.FloatTensor(nMaturitySize),
                       torch.FloatTensor(nFurLength),
                       torch.FloatTensor(nVaccinated),
                       torch.FloatTensor(nDewormed),
                       torch.FloatTensor(nSterilized),
                       torch.FloatTensor(nHealth),
                       d[:,13:16],
                       torch.FloatTensor(nState),
                       d[:,17:]
                      ], dim=1).cuda()
    else:
        d = torch.cat([torch.FloatTensor(nType),
                       d[:,1:2],
                       torch.FloatTensor(nBreed1),
                       torch.FloatTensor(nBreed2),
                       torch.FloatTensor(nGender),
                       torch.FloatTensor(nColor1),
                       torch.FloatTensor(nColor2),
                       torch.FloatTensor(nColor3),
                       d[:,8:10],
                       torch.FloatTensor(nVaccinated),
                       torch.FloatTensor(nDewormed),
                       torch.FloatTensor(nSterilized),
                       d[:,14:16],
                       torch.FloatTensor(nState),
                       d[:,17:]
                      ], dim=1).cuda()

    random.shuffle(d)
    partition = {}
    validation = d[:len(d)//10]
    partition['validation'] = CSVDataset(validation)
    train_set = d[len(d)//10:]
    partition['train'] = CSVDataset(train_set)
    data_train_loader = data.DataLoader(partition['train'], shuffle=True, batch_size=32)
    data_val_loader = data.DataLoader(partition['validation'], batch_size=32)
    return data_train_loader, data_val_loader


def train(model, HIDDEN, ONE_HOT, DATA_AUG, data_train_loader, data_val_loader):
    print("Training...")
    training_dir = './training_{}+{}_{}_{}_{}'.format(ONE_HOT, DATA_AUG, len(HIDDEN), max(HIDDEN), time.time())
    os.mkdir(training_dir)
    os.mkdir(training_dir+'/misclassified')
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    
    qwk_loss = cohen_kappa_score
    ce_loss = nn.CrossEntropyLoss().cuda()
    epoch = 0
    store_epoch_loss = []
    store_qwk_epoch_loss = []
    store_epoch_loss_val = []
    store_qwk_epoch_loss_val = []
    store_epoch_acc_val = []
    try:
        for e in tqdm(range(EPOCHS)):
            #scheduler.step()
            epoch = e + 1
            epoch_loss = 0
            qwk_epoch_loss = 0
            store_batch_loss = []
            store_qwk_batch_loss = []
            
            for batch_num, (X, y) in enumerate(data_train_loader):
                optimizer.zero_grad()
                prediction = model.forward(X.cuda())
                batch_loss = ce_loss(prediction, y)
                batch_loss.backward()
                qwk_batch_loss = qwk_loss(y.clone().detach().cpu().numpy(), 
                                          np.argmax(prediction.clone().detach().cpu().numpy(), axis=1), 
                                          weights="quadratic")
                optimizer.step()
                store_batch_loss.append(batch_loss.clone().cpu())
                store_qwk_batch_loss.append(qwk_batch_loss)
                epoch_loss = torch.FloatTensor(store_batch_loss).mean()
                qwk_epoch_loss = torch.FloatTensor(store_qwk_batch_loss).mean()
                
            store_epoch_loss.append(epoch_loss)
            store_qwk_epoch_loss.append(qwk_epoch_loss)
            torch.save(model.state_dict(), "{}/checkpoint_{}.pth".format(training_dir, epoch))
#             plt.plot(store_epoch_loss[1:], label="Training Loss")
#             plt.plot(store_qwk_epoch_loss[1:], label="Training Metric(QWK)")

            model.eval()
            epoch_loss_val = 0
            qwk_epoch_loss_val = 0
            epoch_acc_val = 0
            store_batch_loss_val = []
            store_qwk_batch_loss_val = []
            store_batch_acc_val = []
            misclassified_images = []
            for batch_num, (X, y) in enumerate(data_val_loader):
                with torch.no_grad():
                    prediction = model.forward(X.cuda())
                batch_loss = ce_loss(prediction, y)
                qwk_batch_loss = qwk_loss(y.clone().detach().cpu().numpy(), 
                                          np.argmax(prediction.clone().detach().cpu().numpy(), axis=1), 
                                          weights="quadratic")
                misclassified = prediction.max(-1)[-1].squeeze().cpu() != y.cpu()
                misclassified_images.append(X[misclassified==1].cpu())
                batch_acc = misclassified.float().mean()
                store_batch_loss_val.append(batch_loss)
                store_qwk_batch_loss_val.append(qwk_batch_loss)
                store_batch_acc_val.append(batch_acc)
                epoch_loss_val = torch.FloatTensor(store_batch_loss_val).mean()
                qwk_epoch_loss_val = torch.FloatTensor(store_qwk_batch_loss_val).mean()
                epoch_acc_val = torch.FloatTensor(store_batch_acc_val).mean()
            store_epoch_loss_val.append(epoch_loss_val)
            store_qwk_epoch_loss_val.append(qwk_epoch_loss_val)
            store_epoch_acc_val.append(1-epoch_acc_val)
            plt.plot(store_epoch_loss_val[1:], label="Validation Loss")
            plt.plot(store_qwk_epoch_loss_val[1:], label="Validation Metric(QWK)")
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
        qwk_max_loss = max(store_qwk_epoch_loss_val)
        print("\nHighest accuracy of {} occured at {}...\nMinimum loss occured at {}... \nMaximum QWK metric of {} occured at {}".format(
            most_acc, store_epoch_acc_val.index(most_acc)+1, 
            store_epoch_loss_val.index(min_loss)+1, 
            qwk_max_loss, store_qwk_epoch_loss_val.index(qwk_max_loss)+1))
        with open(training_dir+"/HYP.txt","w+") as f:
            f.write("EPOCH = {} \n".format(EPOCHS))
            f.write("LR = {} \n".format(LR))
            f.write("HIDDEN_LAYERS = {} \n".format(HIDDEN))
            f.write("ONE_HOT = {} \n".format(ONE_HOT))
            f.write("DATA_AUG = {} \n".format(DATA_AUG))
            f.write("Highest accuracy of {} occured at {}...\nMinimum loss of {} occured at {}... \nMaximum QWK metric of {} occured at {}".format(
            most_acc, store_epoch_acc_val.index(most_acc)+1, 
            min_loss, store_epoch_loss_val.index(min_loss)+1, 
            qwk_max_loss, store_qwk_epoch_loss_val.index(qwk_max_loss)+1))
        checkpoints = os.listdir(training_dir)
        for checkpoint in checkpoints:
            if "checkpoint" in checkpoint:
                checkpoint_num = int(checkpoint[checkpoint.index("_")+1:checkpoint.index(".")])
                if checkpoint_num not in [store_qwk_epoch_loss_val.index(qwk_max_loss)+1,
                                          store_epoch_loss_val.index(min_loss)+1,
                                          store_epoch_acc_val.index(most_acc)+1]:
                    os.remove(training_dir+"/"+checkpoint)        
    except KeyboardInterrupt:
        most_acc = max(store_epoch_acc_val)
        min_loss = min(store_epoch_loss_val)
        qwk_max_loss = max(store_qwk_epoch_loss_val)
        print("\nHighest accuracy of {} occured at {}...\nMinimum loss of {} occured at {}... \nMaximum QWK metric of {} occured at {}".format(
            most_acc, store_epoch_acc_val.index(most_acc)+1, 
            min_loss, store_epoch_loss_val.index(min_loss)+1, 
            qwk_max_loss, store_qwk_epoch_loss_val.index(qwk_max_loss)+1))
        with open(training_dir+"/HYP.txt","w+") as f:
            f.write("EPOCH = {} \n".format(EPOCHS))
            f.write("LR = {} \n".format(LR))
            f.write("HIDDEN_LAYERS = {} \n".format(HIDDEN))
            f.write("ONE_HOT = {} \n".format(ONE_HOT))
            f.write("DATA_AUG = {} \n".format(DATA_AUG))
            f.write("Highest accuracy of {} occured at {}...\nMinimum loss of {} occured at {}... \nMaximum QWK metric of {} occured at {}\n".format(
            most_acc, store_epoch_acc_val.index(most_acc)+1,
            min_loss, store_epoch_loss_val.index(min_loss)+1,
            qwk_max_loss, store_qwk_epoch_loss_val.index(qwk_max_loss)+1))
            f.write("TRAINING INCOMPLETE")
        checkpoints = os.listdir(training_dir)
        for checkpoint in checkpoints:
            if "checkpoint" in checkpoint:
                checkpoint_num = int(checkpoint[checkpoint.index("_")+1:checkpoint.index(".")])
                if checkpoint_num not in [store_qwk_epoch_loss_val.index(qwk_max_loss)+1,
                                          store_epoch_loss_val.index(min_loss)+1,
                                          store_epoch_acc_val.index(most_acc)+1]:
                    os.remove(training_dir+"/"+checkpoint)

if __name__ == "__main__":
    for ONE_HOT in [0,1]: # for MaturitySize, FurLength, Health
        for DATA_AUG in [0,1]: # for state data
            data_train_loader, data_val_loader = prep_dataset(ONE_HOT, DATA_AUG)
            train(Model(HIDDEN_LIST[2], ONE_HOT, DATA_AUG).cuda(), HIDDEN_LIST[2], ONE_HOT, DATA_AUG, data_train_loader, data_val_loader)