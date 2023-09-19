import torch
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import matplotlib.pyplot as plt
valid_loss = []
valid_acc = [0] * 1
device = torch.device("cpu")
train_loss = []
EPOCHS = 50

def CalcValLossAndAccuracy(model, loss_fn, val_loader):
    with torch.no_grad():
        Y_shuffled, Y_preds, losses = [],[],[]
        for X, Y in val_loader:
            X = X.to(device)
            #X = X.float().to(device)
            Y = Y.to(device)
            #print(model)
            preds = model(X)
            #print(preds)
            loss = loss_fn(preds, Y)
            l2_reg = torch.tensor(0.).to(device)
            for param in model.parameters():
                l2_reg += torch.norm(param, p=2)
            loss += 0.001 * l2_reg
            
            losses.append(loss.item())

        
            Y_shuffled.append(Y)
            Y_preds.append(preds.argmax(dim=-1))
            #print(Y_preds)

        Y_shuffled = torch.cat(Y_shuffled)
        Y_preds = torch.cat(Y_preds)

        print(Y_shuffled)
        print(Y_preds)
        
        valid_loss.append(torch.tensor(losses).mean())
        if accuracy_score(Y_shuffled.detach().cpu().numpy(), Y_preds.detach().cpu().numpy()) > max(valid_acc):
            torch.save(model, "./model.pt")
        valid_acc.append(accuracy_score(Y_shuffled.detach().cpu().numpy(), Y_preds.detach().cpu().numpy()))
        print("Valid Loss : {:.3f}".format(torch.tensor(losses).mean()))
        print("Valid Acc  : {:.3f}".format(accuracy_score(Y_shuffled.detach().cpu().numpy(), Y_preds.detach().cpu().numpy())))

def TrainModel(model, loss_fn, optimizer, train_loader, val_loader, epochs):
    for i in range(1, epochs+1):
        losses = []
        #print(next(model.parameters()).device)
        for X, Y in tqdm(train_loader):
            X = X.to(device)
            #X = X.float().to(device)
            Y = Y.to(device)
            #print(X.shape)
            Y_preds = model(X) ## Make Predictions
            loss = loss_fn(Y_preds, Y) ## Calculate Loss
            l2_reg = torch.tensor(0.).to(device)
            for param in model.parameters():
                l2_reg += torch.norm(param, p=2)
            loss += 0.001 * l2_reg

            losses.append(loss.item())

            optimizer.zero_grad() ## Clear previously calculated gradients
            loss.backward() ## Calculates Gradients
            optimizer.step() ## Update network weights.

        train_loss.append(torch.tensor(losses).mean())
        print("Train Loss : {:.3f}".format(torch.tensor(losses).mean()))
        CalcValLossAndAccuracy(model, loss_fn, val_loader)


