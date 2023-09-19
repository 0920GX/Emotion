import pandas as pd
import re

#讀取資料集，並讀入HTML部分
path = "./train_raw.xlsx"


df = pd.read_excel(path, usecols="A")
label = pd.read_excel(path, usecols="B")
#label_data = label['label']
#print(df)
#print(label)

html = []
pre_html = []
label_data = []

#正規表示法，符合HTML內所有的<form~</form>
role = re.compile(r'<form\b[^>]*>[\s\S]*?<\/form>', flags=re.M)

#輸入資料集可以取得每個HTML
def getRow(df):
    for index, row in df.items():
        return row


def getLabel(label):
    for index, labels in label.items():
        return labels

def find_label(idx):
    return labels[idx]

rows = getRow(df)
labels = getLabel(label)
total = 1
loss = 0


###主要任務：將篩出來的form填回資料集。或是重寫一份資料集，漏洞分類記得也要寫回去。
###Tips:在下方迭代的部分寫應該就可以寫完了。
#迭代整份資料集的HTML部分
for row in rows:
    #用正規表示法篩選目前的row
    list_of_form = role.findall(row)
    if list_of_form != []:
        # print(total+1, 'Found!', list_of_form)
        html.append(list_of_form)
        label_data.append(find_label(total-1))
        total += 1
    else:
        # print(total+1, 'Not found!')
        loss += 1
        total += 1


#計算經過篩選後，剩餘幾筆資料
print('剩下:', total-loss, '/', total)

#print(html)

def preprocess_text(sentence):
    # 將tag轉換成小寫
    sentence = sentence.lower()
    # 將所有tag轉換成單詞(token)
    sentence = sentence.replace("\n","")
    sentence = sentence.replace("\t","")
    sentence = re.sub(r'>[\s]+<',"><", sentence)
    #print(sentence)
    
    

    return sentence



for htmls in html:
    raw = ''.join(htmls)
    pre_html.append(preprocess_text(raw))

print(pre_html)


dict = {'html': pre_html,'label': label_data}
# print(pre_html[23])
# print(dict.get())

dataframe = pd.DataFrame(data=dict)
dataframe.set_index('html',inplace=True)

dataframe.to_excel("./pre_train.xlsx")
