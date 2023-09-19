import torch
import gc
from torch.nn import functional as F
from getData import textDataset
from torch.utils.data import DataLoader
from torchtext.data.functional import to_map_style_dataset
from padding import vectorize_batch

model = torch.load("./model.pt")
device = torch.device("cpu")
test_path = "./pre_test.xlsx"
test_iter = textDataset(test_path)
BATCH_SIZE = 96
test_dataset = to_map_style_dataset(test_iter)

html_label = ["Safe","Reflect","Storage"]
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=vectorize_batch)



def MakePredictions(model, loader):
    Y_shuffled, Y_preds = [], []
    for X, Y in loader:
        X = X.to(device)
        Y = Y.to(device)
        preds = model(X)
        Y_preds.append(preds)
        Y_shuffled.append(Y)
    gc.collect()
    Y_preds, Y_shuffled = torch.cat(Y_preds), torch.cat(Y_shuffled)
    #print(Y_preds, Y_shuffled)

    #return Y_shuffled.detach().cpu().numpy(), F.softmax(Y_preds, dim=-1).argmax(dim=-1).detach().cpu().numpy()
    return F.softmax(Y_preds, dim=-1).argmax(dim=-1).detach().cpu().numpy()

preds = MakePredictions(model, test_dataloader)
for paras in preds:
    print("預測類型:", html_label[paras])