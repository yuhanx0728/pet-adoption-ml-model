import pandas as pd
import numpy as np
import torch, random, models
import torch.utils.data as data
import state_data as aug
import pandas as pd
import numpy as np
import torch, random, models, os, json
import torch.utils.data as data
import state_data as aug

def normalize(X, x_min, x_max):
    nom = (X-X.min(axis=0))*(x_max-x_min)
    denom = X.max(axis=0) - X.min(axis=0)
    denom[denom==0] = 1
    return x_min + nom/denom 

def fold_divider(data_loader, NUM_FOLD, NUM_FOLDS):
    partition = {}
    one_fold = len(data_loader)//NUM_FOLDS
    validation = data_loader[one_fold*NUM_FOLD:one_fold*(NUM_FOLD+1)]
    partition['validation'] = models.CSVDataset(validation)
    train_set = torch.cat((data_loader[:one_fold*NUM_FOLD],
                           data_loader[one_fold*(NUM_FOLD+1):]), 0)
    partition['train'] = models.CSVDataset(train_set)
    data_train_loader = data.DataLoader(partition['train'], shuffle=True, batch_size=32)
    data_val_loader = data.DataLoader(partition['validation'], batch_size=32)
    return data_train_loader, data_val_loader

def add_sentiment_data(df):
    json_files = os.listdir('data/train_sentiment')
    columns = ['PetID', 'text_mag', 'text_score']
    add_df = pd.DataFrame(columns=columns)
    for j in json_files:
        petID = j[:j.index('.')]
        with open('data/train_sentiment/'+j) as f:
            data = json.load(f)
            mag = data['documentSentiment']['magnitude']
            score = data['documentSentiment']['score']
            add_df = add_df.append({'PetID':petID, 'text_mag':mag, 'text_score':score}, ignore_index=True)
    df = pd.merge(df, add_df, on='PetID', how='outer')
    df['text_mag'] = df['text_mag'].fillna(0)
    df['text_score'] = df['text_score'].fillna(0)
    return df

def preprocess_data(ONE_HOT, DATA_AUG):
    print("Data preparing...")
    df = pd.read_csv('data/train.csv')
    
    df = add_sentiment_data(df)
    
    df = df.drop(['Name', 'RescuerID', 'PetID', 'Description'], axis=1)

    # Type,Name,Age,Breed1,Breed2,Gender,Color1,Color2,Color3,MaturitySize,
    #    0,   x,  1,     2,     3,     4,     5,     6,     7,           8,
    # FurLength,Vaccinated,Dewormed,Sterilized,Health,Quantity,Fee,State,RescuerID,
    #         9,        10,      11,        12,    13,      14, 15,   16,        x,
    # VideoAmt,Description,PetID,PhotoAmt,AdoptionSpeed, (text_mag), (text_score)
    #       17,          x,    x,      18,           19

    # df['Age']
    nAge = np.column_stack((df['Age'], df['Age']))
    nAge = normalize(nAge, 0, 1)
    df['Age'] = nAge[:,0]

    # df['Quantity']
    nQuantity = np.column_stack((df['Quantity'], df['Quantity']))
    nQuantity = normalize(nQuantity, 0, 1)
    nQuantity = nQuantity[:,0]

    # df['Fee']
    nFee = np.column_stack((df['Fee'], df['Fee']))
    nFee = normalize(nFee, 0, 1)
    df['Fee'] = nFee[:,0]

    # df['VideoAmt']
    nVideoAmt = np.column_stack((df['VideoAmt'], df['VideoAmt']))
    nVideoAmt = normalize(nVideoAmt, 0, 1)
    df['VideoAmt'] = nVideoAmt[:,0]

    # df['PhotoAmt']
    nPhotoAmt = np.column_stack((df['PhotoAmt'], df['PhotoAmt']))
    nPhotoAmt = normalize(nPhotoAmt, 0, 1)
    df['PhotoAmt'] = nPhotoAmt[:,0]

    # df['text_mag']
    ntext_mag = np.column_stack((df['text_mag'], df['text_mag']))
    ntext_mag = normalize(ntext_mag, -1, 1)
    df['text_mag'] = ntext_mag[:,0]

    # df['text_score']
    ntext_score = np.column_stack((df['text_score'], df['text_score']))
    ntext_score = normalize(ntext_score, 0, 1)
    df['text_score'] = ntext_score[:,0]
    
    if not ONE_HOT:
        nMaturitySize = np.column_stack((df['MaturitySize'], df['MaturitySize']))
        nMaturitySize = normalize(nMaturitySize, 0, 1)
        df['MaturitySize'] = nMaturitySize[:,0]
        
        nFurLength = np.column_stack((df['FurLength'], df['FurLength']))
        nFurLength = normalize(nFurLength, 0, 1)
        df['FurLength'] = nFurLength[:,0]
        
        nHealth = np.column_stack((df['Health'], df['Health']))
        nHealth = normalize(nHealth, 0, 1)
        df['Health'] = nHealth[:,0]
    
    d = torch.FloatTensor(df.values)

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

    # df['AdoptionSpeed']
    nSpeed = df['AdoptionSpeed']
    
    if DATA_AUG:
        state_data = {}
        for k, v in aug.state_gdp.items():
            state_data[k] = np.array([v, aug.state_population[k], aug.state_area[k]]).astype(float)

        nState = np.array([[0, 0, 0]]*d.size(0)).astype(float)
        for k,v in state_data.items():
            nState[df['State'].values.astype(int)==k] = v
            nState = normalize(nState, 0, 1)

    if ONE_HOT:
        idx = d[:,8]
        nMaturitySize = torch.zeros(len(idx), int(idx.max())+1).scatter_(1, idx.view(len(idx)).long().unsqueeze(1), 1.)[:,1:]
        idx = d[:,9]
        nFurLength = torch.zeros(len(idx), int(idx.max())+1).scatter_(1, idx.view(len(idx)).long().unsqueeze(1), 1.)[:,1:]
        idx = d[:,13]
        nHealth = torch.zeros(len(idx), int(idx.max())+1).scatter_(1, idx.view(len(idx)).long().unsqueeze(1), 1.)[:,1:]
        
        d = torch.cat([torch.FloatTensor(nType), # cat
                       d[:,1].view(-1,1), # Age
                       torch.FloatTensor(nBreed1), # cat
                       torch.FloatTensor(nBreed2), # cat
                       torch.FloatTensor(nGender), # cat
                       torch.FloatTensor(nColor1), # cat
                       torch.FloatTensor(nColor2), # cat
                       torch.FloatTensor(nColor3), # cat
                       torch.FloatTensor(nMaturitySize),
                       torch.FloatTensor(nFurLength),
                       torch.FloatTensor(nHealth),
                       torch.FloatTensor(nVaccinated), # cat
                       torch.FloatTensor(nDewormed), # cat
                       torch.FloatTensor(nSterilized), # cat
                       d[:,14].view(-1,1), # Quantity
                       d[:,15].view(-1,1), # Fee 
                       torch.FloatTensor(nState), # special
                       d[:,17].view(-1,1), # VideoAmt
                       d[:,18].view(-1,1), # PhotoAmt
                       d[:,20].view(-1,1), # text_mag
                       d[:,21].view(-1,1), # text_score
                       d[:,19].view(-1,1) # adoptionspeed
                      ], dim=1).cuda()
    else:
        
        d = torch.cat([torch.FloatTensor(nType),
                       d[:,1].view(-1,1), # Age
                       torch.FloatTensor(nBreed1),
                       torch.FloatTensor(nBreed2),
                       torch.FloatTensor(nGender),
                       torch.FloatTensor(nColor1),
                       torch.FloatTensor(nColor2),
                       torch.FloatTensor(nColor3),
                       d[:,8:10], # MaturitySize, FurLength
                       torch.FloatTensor(nVaccinated),
                       torch.FloatTensor(nDewormed),
                       torch.FloatTensor(nSterilized),
                       d[:,13].view(-1,1), # Health
                       d[:,14].view(-1,1), # Quantity
                       d[:,15].view(-1,1), # Fee 
                       torch.FloatTensor(nState), # special
                       d[:,17].view(-1,1), # VideoAmt
                       d[:,18].view(-1,1), # PhotoAmt
                       d[:,20].view(-1,1), # text_mag
                       d[:,21].view(-1,1), # text_score
                       d[:,19].view(-1,1) # adoptionspeed
                      ], dim=1).cuda()

    random.shuffle(d)
    print("Data prepared.")
    return d