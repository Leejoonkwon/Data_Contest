# https://dacon.io/en/competitions/official/236037/overview/description

import pandas as pd
import numpy as np
import torch
import os
import random
from sklearn.model_selection import train_test_split
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import nn
from tqdm import tqdm

# for graphing
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings(action='ignore') 
path = 'C:\data_set\\text_dacon/'
train_original = pd.read_csv(path + 'train.csv')
train_original.drop(columns=['ID'], inplace=True)
test = pd.read_csv(path + 'test.csv')
test.drop(columns=['ID'], inplace=True)
submission = pd.read_csv(path + 'sample_submission.csv')


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

CFG = {
    'EPOCHS':20,
    'LEARNING_RATE':1e-5,
    'BATCH_SIZE':32,
    'SEED':41
}

seed_everything(CFG['SEED']) # Seed 고정
device = torch.device('mps')

train, val, _, _ = train_test_split(train_original, train_original['label'], test_size=0.2, random_state=CFG['SEED'])
train = train.reset_index(drop=True)
val = val.reset_index(drop=True)

model_nm = 'klue/roberta-small'
base_model = AutoModel.from_pretrained(model_nm)
tokenizer = AutoTokenizer.from_pretrained(model_nm)

tokenizer_len = [len(tokenizer(s)['input_ids']) for s in train['문장']]
sns.histplot(tokenizer_len)
plt.show()

print(f'log value : {np.mean(tokenizer_len)+3*np.std(tokenizer_len)}')

tokenizer_log = np.log(tokenizer_len)
sns.histplot(tokenizer_log)
plt.show()

print(f'log value : {np.mean(tokenizer_log)+3*np.std(tokenizer_log)}')
print(f'original value : {np.exp(np.mean(tokenizer_log)+3*np.std(tokenizer_log))}')

class SentenceTypeDataset(Dataset):
    def __init__(self, dataframe, tokenizer, labels=None):
        texts = dataframe['문장'].values.tolist()

        self.texts = [tokenizer(text, padding='max_length', max_length=90, truncation=True, return_tensors='pt') for text in texts]
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]

        if self.labels is not None:
            type_tmp = self.labels['type'][idx]
            polarity_tmp = self.labels['polarity'][idx]
            tense_tmp = self.labels['tense'][idx]
            certainty_tmp = self.labels['certainty'][idx]
            return text, torch.Tensor(type_tmp), torch.Tensor(polarity_tmp), torch.Tensor(tense_tmp), torch.Tensor(certainty_tmp)
        else:
            return text, torch.Tensor([-1,-1,-1,-1]), torch.Tensor([-1,-1,-1]), torch.Tensor([-1,-1,-1]), torch.Tensor([-1,-1])
        
class SentenceClassifier(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.klue = base_model # from transformers package

        self.fc1 = nn.Linear(768, 32)
        self.relu = nn.ReLU()
        self.type_clf = nn.Linear(32,4)
        self.polarity_clf = nn.Linear(32,3)
        self.tense_clf = nn.Linear(32,3)
        self.certainty_clf = nn.Linear(32,2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_ids, attention_mask):
        # input_ids : token's id / attention_mask : make a model to focus on which token
        klue_out = self.klue(input_ids= input_ids, attention_mask = attention_mask)[0][:,0]

        x = self.fc1(klue_out)
        x = self.relu(x)

        type_output = self.type_clf(x)
        type_output = self.softmax(type_output)
        polarity_output = self.polarity_clf(x)
        polarity_output = self.softmax(polarity_output)
        tense_output = self.tense_clf(x)
        tense_output = self.softmax(tense_output)
        certainty_output = self.certainty_clf(x)
        certainty_output = self.softmax(certainty_output)

        return type_output, polarity_output, tense_output, certainty_output
def sentence_train(model, train_dataloader, val_dataloader, learning_rate, epochs, model_nm):
    best_val_loss = 99999999999999 # setting max (act as infinity)
    early_stopping_threshold_count = 0

    criterion = {
        'type' : nn.CrossEntropyLoss().to(device),
        'polarity' : nn.CrossEntropyLoss().to(device),
        'tense' : nn.CrossEntropyLoss().to(device),
        'certainty' : nn.CrossEntropyLoss().to(device)
    }

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model = model.to(device)

    for epoch in range(epochs):
        total_acc_train = 0
        total_loss_train = 0
        
        model.train() # sets into the training mode
        
        for train_input, type_label, polarity_label, tense_label, certainty_label in tqdm(train_dataloader):
            attention_mask = train_input['attention_mask'].to(device)
            input_ids = train_input['input_ids'].squeeze(1).to(device)
            type_label = type_label.to(device)
            polarity_label = polarity_label.to(device)
            tense_label = tense_label.to(device)
            certainty_label = certainty_label.to(device)

            optimizer.zero_grad()
            
            type_output, polarity_output, tense_output, certainty_output = model(input_ids, attention_mask) # from the forward function
            
            loss = 0.25*criterion['type'](type_output, type_label.float()) + \
                   0.25*criterion['polarity'](polarity_output, polarity_label.float()) + \
                   0.25*criterion['tense'](tense_output, tense_label.float()) + \
                   0.25*criterion['certainty'](certainty_output, certainty_label.float())
            total_loss_train += loss.item()

            loss.backward()
            optimizer.step()


        with torch.no_grad(): # since we should not change gradient for validation 
            total_acc_val = 0
            total_loss_val = 0
            
            model.eval() # deactivate training
            
            # same process as the above
            for val_input, vtype_label, vpolarity_label, vtense_label, vcertainty_label in tqdm(val_dataloader):
                attention_mask = val_input['attention_mask'].to(device)
                input_ids = val_input['input_ids'].squeeze(1).to(device)

                vtype_label = vtype_label.to(device)
                vpolarity_label = vpolarity_label.to(device)
                vtense_label = vtense_label.to(device)
                vcertainty_label = vcertainty_label.to(device)
                
                vtype_output, vpolarity_output, vtense_output, vcertainty_output = model(input_ids, attention_mask) # from the forward function

                loss = 0.25*criterion['type'](vtype_output, vtype_label.float()) + \
                        0.25*criterion['polarity'](vpolarity_output, vpolarity_label.float()) + \
                        0.25*criterion['tense'](vtense_output, vtense_label.float()) + \
                        0.25*criterion['certainty'](vcertainty_output, vcertainty_label.float())

                total_loss_val += loss.item()

            
            print(f'Epochs: {epoch + 1} '
                  f'| Train Loss: {total_loss_train / len(train_dataloader): .3f} '
                  f'| Train Accuracy: {total_acc_train / (len(train_dataloader.dataset)): .3f} '
                  f'| Val Loss: {total_loss_val / len(val_dataloader): .3f} '
                  f'| Val Accuracy: {total_acc_val / len(val_dataloader.dataset): .3f}')
            
            if best_val_loss > total_loss_val:
                best_val_loss = total_loss_val # saving only the best one
                torch.save(model, f"model/{model_nm}.pt")
                print("Saved model")
                early_stopping_threshold_count = 0
            else:
                early_stopping_threshold_count += 1 # checking how many epochs have passed that val_loss didn't increase
                
            if early_stopping_threshold_count >= 3: # ==> patience=1
                print("Early stopping")
                break
            
train_tmp = train[['문장', '유형', '극성', '시제', '확실성']]
train_tmp = pd.get_dummies(train_tmp, columns=['유형', '극성', '시제', '확실성'])
print(train_tmp)

train_type = train_tmp.iloc[:,1:5].values.tolist()
train_polarity = train_tmp.iloc[:,5:8].values.tolist()
train_tense = train_tmp.iloc[:,8:11].values.tolist()
train_certainty = train_tmp.iloc[:,11:13].values.tolist()
train_labels = {
    'type': train_type,
    'polarity': train_polarity,
    'tense': train_tense,
    'certainty': train_certainty
}
val_tmp = val[['문장', '유형', '극성', '시제', '확실성']]
val_tmp = pd.get_dummies(val_tmp, columns=['유형', '극성', '시제', '확실성'])

val_type = val_tmp.iloc[:,1:5].values.tolist()
val_polarity = val_tmp.iloc[:,5:8].values.tolist()
val_tense = val_tmp.iloc[:,8:11].values.tolist()
val_certainty = val_tmp.iloc[:,11:13].values.tolist()
val_labels = {
    'type': val_type,
    'polarity': val_polarity,
    'tense': val_tense,
    'certainty': val_certainty
}
train_dataloader = DataLoader(SentenceTypeDataset(train_tmp, tokenizer, train_labels), batch_size=CFG['BATCH_SIZE'], shuffle=True, num_workers=0) # num_workers: how many subprocesses to use for data loading  
val_dataloader = DataLoader(SentenceTypeDataset(val_tmp, tokenizer, val_labels), batch_size=CFG['BATCH_SIZE'], num_workers=0)

model = SentenceClassifier(base_model)

sentence_train(model, train_dataloader, val_dataloader, CFG['LEARNING_RATE'], CFG['EPOCHS'], 'kclue')

def get_type_predictions(model, loader):

    device = torch.device('mps')
    model = model.to(device)
    
    type_probs, polarity_probs, tense_probs, clarity_probs = [], [], [], []
    with torch.no_grad():
        model.eval()
        for data_input, _, _, _, _ in tqdm(loader):
            attention_mask = data_input['attention_mask'].to(device)
            input_ids = data_input['input_ids'].squeeze(1).to(device)


            type_output, polarity_output, tense_output, clarity_output = model(input_ids, attention_mask)
            type_probs.append(type_output)
            polarity_probs.append(polarity_output)
            tense_probs.append(tense_output)
            clarity_probs.append(clarity_output)
    
    return torch.cat(type_probs).cpu().detach().numpy(), \
            torch.cat(polarity_probs).cpu().detach().numpy(), \
            torch.cat(tense_probs).cpu().detach().numpy(), \
            torch.cat(clarity_probs).cpu().detach().numpy()
model = torch.load("model/kclue.pt")
test_dataloader = DataLoader(SentenceTypeDataset(test, tokenizer), batch_size=CFG['BATCH_SIZE'], shuffle=False)

#val_pred_type, val_pred_polarity, val_pred_tense, val_pred_certainty = get_type_predictions(model, val_dataloader)

#val_type = ['대화형' if i==0 else '사실형' if i==1 else '예측형' if i==2 else '추론형' for i in [np.argmax(p) for p in val_pred_type]]
#val_polarity = ['긍정' if i==0 else '미정' if i==1 else '부정' for i in [np.argmax(p) for p in val_pred_polarity]]
#val_type = ['과거' if i==0 else '미래' if i==1 else '현재' for i in [np.argmax(p) for p in val_pred_tense]]
#val_type = ['불확실' if i==0 else '확실' for i in [np.argmax(p) for p in val_pred_certainty]]

test_pred_type, test_pred_polarity, test_pred_tense, test_pred_certainty = get_type_predictions(model, test_dataloader)
print(test_pred_tense)