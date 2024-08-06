import logging;
logging.disable(logging.WARNING);

import os;
import numpy as np;
import pandas as pd;

# downloading weights
os.system("mkdir -p ~/.kaggle");
os.system("mv kaggle.json ~/.kaggle/");
os.system("chmod 600 ~/.kaggle/kaggle.json");

df = pd.read_csv("./weights.csv");
print(df['title']);

print("Enter Label");
index = int(input("Waiting for the input.. "));

if index == 3 or index == 6:
    print("Unfotunately weights for setting 3 and 6 are not available for inference");
    exit(0);

print("Downloading weights..");
os.system(f"kaggle kernels output {df.iloc[index]['location']} -p /content/Clickbait-2/weights/");



from transformers import BartTokenizer, BartForConditionalGeneration;
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer;

import torch;
import torch.nn as nn;
import torch.optim as optim;
import torch.nn.functional as F;
from torch.utils.data import Dataset, DataLoader;

device = torch.device("cuda" if torch.cuda.is_available() else "cpu");
print(device);

model = BartForConditionalGeneration.from_pretrained('/content/Clickbait-2/weights/checkpoint-500');
tokenizer = BartTokenizer.from_pretrained('facebook/bart-base');

import numpy as np;
import pandas as pd;

from tqdm import tqdm;

class TestDataset(nn.Module):
    def __init__(self,path_x,path_tags,tokenizer,max_seq_length,max_target_length):
        super().__init__();

        self.max_seq_length = max_seq_length;
        self.max_target_length = max_target_length;

        with open(path_x,'r') as f:
            self.x = f.readlines();

        self.df = pd.read_csv(path_tags);

    def __len__(self):
        return len(self.x);

    def __getitem__(self,idx):
        x = tokenizer(self.df['spoilerType'][idx],self.x[idx],max_length=self.max_seq_length,truncation=True,padding='max_length',return_tensors='pt');
        
        return {
            'input_ids':x['input_ids'].flatten(),
            'attention_mask':x['attention_mask'].flatten(),
        };


test_dataset = TestDataset(
    path_x = f"./DATA/DATA/{df.iloc[index]['input_x']}",
    path_tags = f"./DATA/DATA/{df.iloc[index]['label']}",
    tokenizer=tokenizer,
    max_seq_length=512,
    max_target_length=512, 
);

test_dataloader = DataLoader(test_dataset,batch_size=8);

buffer = [];
model.eval()
for batch in tqdm(test_dataloader,ascii=True,desc="Processing..."):
    input_ids = batch['input_ids'].to(device);
    attention_mask = batch['attention_mask'].to(device);
    
    output = model.generate(input_ids,attention_mask=attention_mask,max_length=300,num_beams=5,early_stopping=True);
    txt = tokenizer.batch_decode(output,skip_special_tokens=True);
    buffer.append(txt);


tmp = [];
for batch in buffer:
    for line in batch:
        tmp.append(line.rstrip());


data = {
    "id" : np.arange(400),
    "spoiler" : tmp
}; 
df = pd.DataFrame(data);   

df.to_csv("output.csv",index=False);