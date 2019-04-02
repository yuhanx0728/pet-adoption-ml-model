import torch
import torch.nn as nn
import torch.utils.data as data

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
        self.base_model = nn.Sequential(nn.Linear(initial, self.HIDDEN[0]),
                                        nn.ReLU(),
                                        nn.Linear(self.HIDDEN[0], self.HIDDEN[1]))
        self.classification_layer1 = nn.Sequential(nn.ReLU(),
                                                   nn.Linear(self.HIDDEN[1], self.HIDDEN[2]))
        self.classification_layer2 = nn.Sequential(nn.ReLU(),
                                                   nn.Linear(self.HIDDEN[2], self.HIDDEN[3]))
        self.output_layer = nn.Sequential(nn.ReLU(),
                                          nn.Linear(HIDDEN[-1], 5))
        if (len(self.HIDDEN) == 5):
            self.classification_layer3 = nn.Sequential(nn.ReLU(),
                                                       nn.Linear(self.HIDDEN[3], self.HIDDEN[4]))
        elif (len(self.HIDDEN) == 6):
            self.classification_layer3 = nn.Sequential(nn.ReLU(),
                                                       nn.Linear(self.HIDDEN[3], self.HIDDEN[4]))
            self.classification_layer4 = nn.Sequential(nn.ReLU(),
                                                       nn.Linear(self.HIDDEN[4], self.HIDDEN[5]))
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