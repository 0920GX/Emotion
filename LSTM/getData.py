import pandas as pd
from torch.utils.data import Dataset


class textDataset(Dataset):
    def __init__(self,csv_path):
        super().__init__()
        self.install_data = pd.read_excel(csv_path)


    def __len__(self):
        return len(self.install_data)
    def __getitem__(self, idx):
        
        label = pd.DataFrame([])

        htmls = self.install_data.iloc[idx, 0]
        #print(htmls)

        label = self.install_data.iloc[idx, 1]
        #print(label)
        return label,htmls
