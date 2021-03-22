import os
import pandas
import numpy as np
from sklearn.preprocessing import normalize
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# create dataset using electricity and weather data


class Dataset(BaseDataset):
    def __init__(
        self,
        train_data_electricity,
        train_data_weather,
    ):
        # prepare the training data and label
        self.train_label = np.array(train_data_electricity, dtype=np.float32)
        self.train_data_electricity = np.array(
            train_data_electricity,  dtype=np.float32)
        self.train_data_weather = np.array(
            train_data_weather,  dtype=np.float32)

        # self.train_data_electricity = np.array(normalize([train_data_electricity], axis=1)[0],  dtype=np.float32)
        # self.train_data_weather = np.array(normalize([train_data_weather], axis=1)[0])

    def __getitem__(self, i):
        x = np.concatenate(
            [self.train_data_electricity[i:10+i],  self.train_data_weather[i:10+i]])
        y = self.train_label[10+i:17+i]

        return x, y

    def __len__(self):
        return len(self.train_data_electricity) - 16

# build model


class LinearModel(nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()

        self.fc0 = nn.Linear(20, 128)
        self.fc1 = nn.Linear(128, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc = nn.Linear(128, 7)
        self.flatten = nn.Flatten()

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc0(x)
        x = self.relu(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc(x)

        return x


def getdataset(train_data, train_data2=None, train_data3=None, train_data4=None):
    # read eletricity data
    eletricities = None

    edf1 = pandas.read_csv(train_data)
    if train_data2 != None:
        edf2 = pandas.read_csv(train_data2)
        eletricities = np.concatenate(
            [edf1['備轉容量(萬瓩)']*10, edf2['備轉容量(萬瓩)']*10])

    else:
        eletricities = edf1['備轉容量(萬瓩)']*10

    avgs_temperature = None
    # read weather data
    if train_data3 and train_data4:
        wdf1 = pandas.read_csv(train_data3)
        # eletricity data only to 2020-12-15
        dates2020 = sorted(list(set(wdf1['日期'])))[:273]
        dates2021 = sorted(list(set(wdf1['日期'])))[289:]

        avgs_temperature = []

        for i in dates2020:
            avgs_temperature.append(np.average(
                wdf1[wdf1['日期'] == i]['當日平均溫度(°C)']))

        for i in dates2021:
            avgs_temperature.append(np.average(
                wdf1[wdf1['日期'] == i]['當日平均溫度(°C)']))

        # read forcast weather data
        wdf2 = pandas.read_csv(train_data4)

        for i in range(len(wdf2)):
            avgs_temperature.append(np.average(wdf2.iloc[i][1:4]))

    # check length of weather datas and electricity datas
    if avgs_temperature:
        if len(avgs_temperature) < len(eletricities):
            eletricities = eletricities[-len(avgs_temperature):]
        elif len(avgs_temperature) > len(eletricities):
            avgs_temperature = avgs_temperature[-len(eletricities):]

    print('length of electricity datas: {}\nlength of weather datas: {}'.format(
        len(eletricities), len(avgs_temperature)))

    # init dataset
    dataset = Dataset(train_data_electricity=eletricities,
                      train_data_weather=avgs_temperature)

    return dataset, eletricities, avgs_temperature

# predict the next 7 days


def predict(model, x):
    x = torch.tensor(x, dtype=torch.float).unsqueeze(dim=0)
    if torch.cuda.is_available():
        x = x.cuda()
    pred = model(x).detach().cpu().numpy().squeeze()

    # save result
    f = open('submission.csv', 'w')
    f.write('date,operating_reserve(MW)\n')
    print('date,operating_reserve(MW)')
    for i, date in enumerate(range(20210323, 20210330)):
        f.write('{},{}\n'.format(date, int(pred[i])))
        print(('{},{}\n'.format(date, int(pred[i]))))

    f.close()


def training(train_data, train_data2=None, train_data3=None, train_data4=None):

    dataset, eletricities, avgs_temperature = getdataset(
        train_data, train_data2, train_data3, train_data4)

    # training
    linear = LinearModel()
    if torch.cuda.is_available():
        linear.cuda()

    # loss function
    loss = torch.nn.MSELoss(reduction='mean')

    # optimizer
    optimizer = torch.optim.Adam([
        dict(params=linear.parameters(), lr=4e-4),
    ])

    # data loader
    train_loaders = torch.utils.data.DataLoader(
        dataset, batch_size=64, shuffle=False, num_workers=8)

    # train model for N epochs
    max_valid_score = 0
    EPOCH = 300
    for epoch in range(EPOCH):

        torch.cuda.empty_cache()

        total_loss = 0
        for x, y in train_loaders:
            optimizer.zero_grad()  # optimizer gradient to zero

            # Get data
            if torch.cuda.is_available():
                x = x.float().cuda()
                y = y.float().cuda()
            else:
                x = x.float()
                y = y.float()

            # Train

            # 1. forward propagationreal_labels
            # -----------------------

            pred = linear(x)

            # 2. loss calculation
            # -----------------------
            _loss = loss(pred, y)

            # track loss
            total_loss += _loss.item()

            # 3. backward propagation
            # -----------------------
            _loss.backward()

            # 4. weight optimization
            # -----------------------
            optimizer.step()

        print("Epoch:", epoch, "Training Loss: {}".format(total_loss))

        total_loss = 0

    torch.save(linear.state_dict(), 'save_model.pth')

    # predict the next 7 days with electricity and weather data from past 10 days
    predict(linear, np.concatenate(
        [eletricities[-10:], avgs_temperature[-10:]]))
