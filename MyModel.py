import json
import pickle as pkl
from pycparser import c_generator
from torch.nn.utils.rnn import pad_sequence

import ForPragmaExtractor.visitors as visitor
# from Model import tokenizer

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm

VAR_PREFIX = "var"
ARR_PREFIX = "arr"
FUNC_PREFIX = "func"
STRUCT_PREFIX = "struct"
generator = c_generator.CGenerator()
id_v = visitor.CounterIdVisitor()
replacer = visitor.ReplaceIdsVisitor(VAR_PREFIX, ARR_PREFIX, STRUCT_PREFIX, FUNC_PREFIX)


def db_read_string_from_file(file_path):
    try:
        with open(file_path, "r") as f:
            return "".join(f.readlines())   # readlines()是读全部
    except:
        return ""


def normalize_code_as_ast(pickle_file):
    # print (pickle_file)
    with open(pickle_file, 'rb') as f:
        pragmafor_tuple = pkl.load(f)  #
        for_ast = pragmafor_tuple.for_node
        # for_ast.show()
        # print(normalize_code_as_string.generator.visit(for_ast))
        # for_ast.show()
        # counts in an array the name and identifiers of the code
        id_v.reset()
        id_v.visit(for_ast)
        # Replace the names..
        replacer.reset(id_v.ids, id_v.array,id_v.struct, id_v.func)
        replacer.visit(for_ast)
        with open('temp.txt', 'w') as f:
            for_ast.show(buf=f)
        with open('temp.txt', 'r') as f:
            ast = f.readlines()

        ast_no_whitespaces = [a.strip() for a in ast] # kill all whitespaces

        ast_one_line = " " + " ".join(ast_no_whitespaces)
        return ast_one_line


# str1 = normalize_code_as_ast('DB_TEST/DB_TEST/database/PolyBench-ACC-master_.gitignore_2mm.c_5/code_pickle.pkl')
# print(str1)

def code_as_ast(pickle_file):
    # print (pickle_file)
    with open(pickle_file, 'rb') as f:
        pragmafor_tuple = pkl.load(f)  #
        for_ast = pragmafor_tuple.for_node
        with open('temp.txt', 'w') as f:
            for_ast.show(buf=f)
        with open('temp.txt', 'r') as f:
            ast = f.readlines()

        ast_no_whitespaces = [a.strip() for a in ast] # kill all whitespaces and \n
        ast_one_line = " " + " ".join(ast_no_whitespaces)
        return ast_one_line


str2 = code_as_ast('DB_TEST/DB_TEST/database/PolyBench-ACC-master_.gitignore_2mm.c_5/code_pickle.pkl')
# print('code_as_ast: ', str2)
# print('str2len: ', len(str2))

data1 = {'text':[str2]}

# text, _ = tokenizer.deepscc_tokenizer(data1['text'])
# print('deepscc_token: ', text)
# print('tokenlen: ', len(text.input_ids[0]), len(text.attention_mask[0]))


# ------------------------------------------------------------------------

data_set = {'text':[], 'label':[], 'ast':[]}
jsonpath = 'D:\CodeLibrary\PragFormer-main\DB_TEST\DB_TEST\database.json'
with open(jsonpath, 'r') as f:
    file_data = json.load(f)
    # print('file_data: ', file_data)
    for i, key in enumerate(file_data):
        # print('file_data[key]["code"]: ', file_data[key]["code"])
        code = db_read_string_from_file(file_data[key]["code"])
        ast_str = code_as_ast(file_data[key]['code_pickle'])
        if file_data[key]['pragma']:
            pragma = db_read_string_from_file(file_data[key]['pragma'])
        else:
            pragma = "0"

        data_set['text'].append(code)
        data_set['ast'].append(ast_str)
        data_set['label'].append(pragma)

# print(data_set['label'])

# print("data_set['text'][0]: \n", data_set['text'][0])
# print("data_set['text'][10]: \n", data_set['text'][10])
# print('-' * 100)
# print("data_set['ast'][0]: \n", data_set['ast'][0])
# print("data_set['ast'][10]: \n", data_set['ast'][10])
# print('-' * 100)
# print("data_set['label'][0]: \n", data_set['label'][0])
# print("data_set['label'][10]: \n", data_set['label'][10])


# Define your dataset class
class MyDataset(Dataset):
    def __init__(self, texts, asts, labels, tokenizer, max_len):
        self.texts = texts
        self.asts = asts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        ast = str(self.asts[idx])
        label = int(self.labels[idx])

        inputs = self.tokenizer(text, ast, return_tensors='pt', max_length=self.max_len, truncation=True, padding='max_length')
        inputs['labels'] = torch.tensor(label)
        print('mydata: ', inputs)
        return inputs

# Encode labels 这种编码不适合当前的任务
# label_encoder = LabelEncoder()
# data_set['label'] = label_encoder.fit_transform(data_set['label'])
# print('\n', data_set['label'])

def label_encoder(labels:list):
    new_labels = []
    for label in labels:
        if 'omp parallel for' in label and 'private' not in label and 'reduction' not in label:
            new_labels.append(1)
        elif 'omp parallel for' in label and 'private' in label and 'reduction' not in label:
            new_labels.append(2)
        elif 'omp parallel for' in label and 'reduction' in label and 'private' not in label:
            new_labels.append(3)
        elif 'omp parallel for' in label and 'reduction' in label and 'private' in label:
            new_labels.append(4)
        else:
            new_labels.append(0)
    return new_labels


# def collate_fn(batch):
#     return {
#         'input_ids': pad_sequence([item['input_ids'] for item in batch], batch_first=True),
#         'attention_mask': pad_sequence([item['attention_mask'] for item in batch], batch_first=True),
#         'labels': torch.tensor([item['labels'] for item in batch])
#     }
#
# data_set['label'] = label_encoder(data_set['label'])
# print(data_set['label'])

# Split the dataset
train_texts, val_texts, train_asts, val_asts, train_labels, val_labels = train_test_split(
    data_set['text'], data_set['ast'], data_set['label'], test_size=0.2, random_state=42
)

# Define tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=5)

# Define datasets and dataloaders
train_dataset = MyDataset(train_texts, train_asts, train_labels, tokenizer, max_len=123)
val_dataset = MyDataset(val_texts, val_asts, val_labels, tokenizer, max_len=123)

train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False)

# Define training parameters
optimizer = AdamW(model.parameters(), lr=1e-5)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Training loop
num_epochs = 3
for epoch in range(num_epochs):
    model.train()
    for batch in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}"):
        inputs = {key: value.to(device) for key, value in batch.items()}
        # print('training inputs: ', inputs)
        optimizer.zero_grad()
        outputs = model(**inputs)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

    # Validation
    model.eval()
    val_preds = []
    val_true = []
    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc=f"Epoch {epoch + 1} - Validation"):
            inputs = {key: value.to(device) for key, value in batch.items()}
            outputs= model(**inputs)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)
            val_preds.extend(preds.cpu().numpy())
            val_true.extend(inputs['label'].cpu().numpy())

    # Calculate validation accuracy and print metrics
    accuracy = accuracy_score(val_true, val_preds)
    report = classification_report(val_true, val_preds, target_names=label_encoder.classes_)
    print(f"Epoch {epoch + 1} - Validation Accuracy: {accuracy:.4f}")
    print(report)


