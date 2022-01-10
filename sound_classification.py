import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

import librosa
import numpy as np
import pandas as pd

from tqdm import tqdm

import argparse

parser = argparse.ArgumentParser(description='sound_classification')

parser.add_argument('--seed', type=int, default=117, help='seed')
parser.add_argument('--epoch', type=int, default=1000, help='epoch')
parser.add_argument('--batch_size', type=int, default=32, help='batch_size')
parser.add_argument('--lr', type=float, default=1e-4, help='learning_rate')
parser.add_argument('--dropout', type=float, default=0.2, help='learning_rate')
parser.add_argument('--fold', type=int, default=1, help='k_fold')
parser.add_argument('--n_class', type=int, default=50, help='number of class')

args = parser.parse_args()

#seed
torch.manual_seed(args.seed)

#GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Let's use", torch.cuda.device_count(), "GPUs!")
print('device:', device)

#load_data
metadata_df = pd.read_csv('./ESC-50/meta/esc50.csv')
filename = metadata_df['filename'].values.tolist()
category = metadata_df['category'].values.tolist()
fold = metadata_df['fold'].values.tolist()

label_list = list(set(category))
label_map = {label: i for i, label in enumerate(label_list)}

#features
features = []
for fi, label, fo in tqdm(zip(filename, category, fold), total=len(metadata_df)):
    file_path = f'./ESC-50/audio/{fi}'
    audio, sr = librosa.load(file_path, duration=2.95, res_type='kaiser_fast')
    melspectrogram = librosa.feature.melspectrogram(y=audio, 
                                                    sr=sr, 
                                                    hop_length=512,
                                                    win_length=512, 
                                                    n_mels=128)
    features.append([melspectrogram, label_map[label], fo])
dataset_df = pd.DataFrame(features, columns=['melspectrogram', 'label', 'fold'])

#transform
train_transforms = None
test_transforms = None

#dataset
class SoundDataset(Dataset):
    def __init__(self, features, transform=None):
        self.features = features
        self.transform = transform

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        spectrogram, label, fold = self.features.iloc[index]
        spectrogram = np.expand_dims(spectrogram, axis=0)

        if self.transform is not None:
            spectrogram = self.transform(spectrogram)

        return spectrogram, label

def process_fold(dataset_df, fold_k):
    train_df = dataset_df[dataset_df['fold'] != fold_k]
    test_df = dataset_df[dataset_df['fold'] == fold_k]

    train_dataset = SoundDataset(train_df, transform=train_transforms)
    train_loader = DataLoader(train_dataset, 
                            batch_size=args.batch_size,
                            shuffle = True,
                            pin_memory=True,
                            num_workers=1)
    
    test_dataset = SoundDataset(test_df, transform=test_transforms)
    test_loader = DataLoader(test_dataset, 
                          batch_size=args.batch_size,
                          shuffle = False,
                          pin_memory=True,
                          num_workers=1)

    return train_loader, test_loader

train_loader, test_loader = process_fold(dataset_df, args.fold)

#model
class CNN(nn.Module):
    def __init__(self, num_class, drop_prob):
        super(CNN, self).__init__()
        # input is 128x128
        self.dropout = nn.Dropout(p=drop_prob)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=24, kernel_size=5, padding=0)  #(input_channel, output_channel, filter_size, padding_size)
        self.conv2 = nn.Conv2d(in_channels=24, out_channels=36, kernel_size=4, padding=0)
        self.conv3 = nn.Conv2d(in_channels=36, out_channels=48, kernel_size=3, padding=0)
        '''
        add here
        '''
        self.fc1 = nn.Linear(in_features=48, out_features=60)
        self.fc2 = nn.Linear(in_features=60, out_features=num_class)
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 3)  #(batch_size, 24, 41, 41)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.relu(self.conv3(x))
        '''
        add here
        '''
        x = F.avg_pool2d(x, kernel_size=x.size()[2:]) #self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        x = x.view(-1, 48)  #flatten
        x = self.dropout(F.relu(self.fc1(x)))
        output = self.fc2(x)

        return output

model = CNN(args.n_class, args.dropout)
model.to(device)

#loss, optimizer
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=args.lr)

#load_weight
##model.load_state_dict(torch.load(f'./path/{args.model}.pth'))

#train
model.train()
total_loss = 0
total_acc = 0
train_loss = []
train_accuracy = []
i = 1
for epoch in range(args.epoch):
    for data, target in train_loader:
        data = data.to(device)        
        target = target.to(device)

        output = model(data)
        loss = criterion(output, target)

        total_loss += loss
        train_loss.append(total_loss.detach().cpu().numpy()/i)

        prediction = output.max(1)[1]  #tensor.max(dim=1)[max, argmax]
        accuracy = prediction.eq(target).sum()/args.batch_size*100

        total_acc += accuracy
        train_accuracy.append(total_acc.detach().cpu().numpy()/i)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            print(f'Epoch: {epoch+1}\t Train Step: {i:3d}\t Loss: {loss:.3f}\t Accuracy: {accuracy:.3f}%')
        i += 1
    print(f'Epoch: {epoch+1} finished')

#save_weight
##torch.save(model.state_dict(), f'./path/{args.model}.pth')

#validation
with torch.no_grad():
    model.eval()
    test_acc_sum = 0
    for data, target in test_loader:
        data = data.to(device)
        target = target.to(device)

        output = model(data)

        prediction = output.max(1)[1]  #tensor.max(dim=1)[max, argmax]
        test_acc_sum += prediction.eq(target).sum()

print(f'\nTest set: Accuracy: {100 * test_acc_sum / len(test_loader.dataset):.3f}%')