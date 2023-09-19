import re

def tokenize_form(html):
    pattern = r'<[^<]+>?'
    #text = '<div class="example">This is an example.</div>'

    tokens = re.findall(pattern, html)
    return tokens


def yield_tokens(datasets):
    for dataset in datasets:
        for _, text in dataset:   #取得每一條的標籤Label和內容Text
            #print(text)
            yield tokenize_form(text)   #將內容用分詞器分詞並返回，yield的返回值會是迭代器型態