import os
import pandas as pd
import time
import math
from torch.utils.data import Dataset, DataLoader
from torch import nn
import torch
import glob
import csv
from torch.distributions import Normal
from torch.utils.tensorboard import SummaryWriter


class NeuralNetwork(nn.Module):
    def __init__(self,num_in,num_out,size_one,size_two):
        super(NeuralNetwork, self).__init__()
    
        self.layer1 =  nn.Linear(num_in,   size_one)
        self.layer2 =  nn.Linear(size_one, size_two)
        self.layer3 =  nn.Linear(size_two, num_out)
        
    
    def forward(self, x):
        x = self.layer1(x)
        x = nn.functional.relu(x)
        x = self.layer2(x)
        x = nn.functional.relu(x)
        res = self.layer3(x)

        return res

def init_normal(m):
    if type(m) == nn.Linear:
        nn.init.uniform_(m.weight)

    
    
    
class PivotData(Dataset):
    def __init__(self,folder):
        path = os.getcwd() + '/' + folder
        list_files = path + "/*.csv"
        files = glob.glob(list_files)
        state = []
        action = []
        for file_name in files:
            file_out = pd.read_csv(file_name)
            state.extend(file_out.iloc[1:120,0:4].values)
            action.extend(file_out.iloc[1:120,8:11].values)

        self.state_train = torch.tensor(state, dtype = torch.float32)
        self.action_train = torch.tensor(action, dtype=torch.float32)

    def __len__(self):
        return len(self.action_train)
    
    def __getitem__(self, index):
        return self.state_train[index], self.action_train[index]
    
    def get_file(self):
        pass

def sqrt_loss(pred,action):
    loss =  torch.sqrt(torch.mean((pred-action)**2))
    return torch.sqrt(loss)
    
if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"
    times = 0
    Alpha = [0.7]
    SIZE_ONE = [8,16,32,64,128]
    torch.manual_seed(0)
   


    Names = ['size_8_min','size_16_min','size_32_min','size_64_min','size_128_min']
    
    for name in Names:
        #Logging the Loss's
        
        f = open(name + '_progress.csv','w')
        writer_csv = csv.writer(f)
        Header = ['epoch','Loss','Validaton Loss']
        writer_csv.writerow(Header)

        D = PivotData('ExpertDataset')
        #alpha = Alpha[times]
        alpha = 0.5
        
        size_of_set = len(D.state_train)
        size_train = round(alpha * size_of_set)
        size_validation = size_of_set - size_train
        train_set, val_set = torch.utils.data.random_split(D, [size_train, size_validation])
    
        model = NeuralNetwork(num_in=4,num_out=3,size_one=SIZE_ONE[times],size_two=SIZE_ONE[times]).to(device)
       
        train_loader = DataLoader(train_set,batch_size = 512,shuffle=True)
        validation_loader = DataLoader(val_set,batch_size = 512,shuffle=True)
    
        epoch = range(5000)

        loss_fn = nn.L1Loss()
        
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
        writer = SummaryWriter(name)

        for i in epoch:
            state,action = next(iter(train_loader))
            state = state.to(device)
            action = action.to(device)
            row = []
            for _ in range(256):    
                pred = model(state)
                loss = loss_fn(pred,action)

                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        
            #validation
            with torch.no_grad():
                state_val,action_val = next(iter(validation_loader))
                state_val = state_val.to(device)
                action_val = action_val.to(device)
                pred_val = model(state_val)
                loss_val = loss_fn(pred_val,action_val)
            
            row.append(i)
            row.append(loss.detach().numpy())
            row.append(loss_val.detach().numpy())
            writer_csv.writerow(row)

            writer.add_scalar("Loss/train", loss, i)
            writer.add_scalar("LossVal/train", loss_val, i)
            writer.flush()
    
        torch.save(model.state_dict(),'./' + name + '_model' + '.pt')
        times += 1