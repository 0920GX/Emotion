from getData import textDataset
from model import LSTMClassifier
from padding import vectorize_batch
from train import TrainModel

from torchtext.data.utils import get_tokenizer
from torch.optim import AdamW
from torch.utils.data.dataset import random_split
from torchtext.data.functional import to_map_style_dataset
from torch.utils.data import DataLoader
import torch

train_path = "./pre_train.xlsx"
test_path = "./pre_train.xlsx"
train_iter = textDataset(train_path)
test_iter = textDataset(test_path)

tokenizer = get_tokenizer("basic_english")


device = torch.device("cpu")

model = LSTMClassifier()
model = model.to(device)


EPOCHS = 50
LR = 0.01
BATCH_SIZE = 96
criterion = torch.nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.001)

train_dataset = to_map_style_dataset(train_iter)

test_dataset = to_map_style_dataset(test_iter)
num_train = int(len(train_dataset) * 0.8)   
split_train_, split_valid_ = \
    random_split(train_dataset, [num_train, len(train_dataset) - num_train])


train_dataloader = DataLoader(split_train_, batch_size=BATCH_SIZE, shuffle=True, collate_fn=vectorize_batch)

vaild_dataloader = DataLoader(split_valid_, batch_size=BATCH_SIZE, shuffle=True, collate_fn=vectorize_batch)

test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=vectorize_batch)

TrainModel(model, criterion, optimizer, train_dataloader, vaild_dataloader, EPOCHS)

