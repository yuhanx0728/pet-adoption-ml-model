import pandas as pd
import numpy as np
import torch, random, models
import torch.utils.data as data
import state_data as aug

def preprocess_data(ONE_HOT, DATA_AUG):
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
        for k, v in aug.state_gdp.items():
            state_data[k] = np.array([v, aug.state_population[k], aug.state_area[k]]).astype(float)

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
    partition['validation'] = models.CSVDataset(validation)
    train_set = d[len(d)//10:]
    partition['train'] = models.CSVDataset(train_set)
    data_train_loader = data.DataLoader(partition['train'], shuffle=True, batch_size=32)
    data_val_loader = data.DataLoader(partition['validation'], batch_size=32)
    return data_train_loader, data_val_loader
