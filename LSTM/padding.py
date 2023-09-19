from preprocess import tokenize_form
import torch
from getData import textDataset
from torchtext.vocab import build_vocab_from_iterator
from preprocess import yield_tokens


train_path = "./pre_train.xlsx"
test_path = "./pre_test.xlsx"
train_iter = textDataset(train_path)
test_iter = textDataset(test_path)
max_words = 100

vocab = build_vocab_from_iterator(yield_tokens([train_iter,test_iter]),min_freq=1, specials=["<unk>","<pad>"]) 
vocab.set_default_index(vocab['<unk>'])

def vectorize_batch(batch):
    Y, X = list(zip(*batch))
    X = [vocab(tokenize_form(text)) for text in X]
    X = [tokens+([0]* (max_words-len(tokens))) if len(tokens)<max_words else tokens[:max_words] for tokens in X] 


    return torch.tensor(X, dtype=torch.int32), torch.tensor(Y) - 1 ## We have deducted 1 from target names to get them in range [0,1,2,3] from [1,2,3,4]