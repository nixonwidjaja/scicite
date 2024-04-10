import pandas as pd
import torch 
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification
from utils import train_model, eval_model, save_model, augment_data_multiclass, preprocessing

train_df = pd.read_json('../train.jsonl', lines=True)
X_train = train_df['string']
y_train = train_df['label']

test_df = pd.read_json('../test.jsonl', lines=True)
X_test = test_df['string']
y_test = test_df['label']

# Upsample the training data
X_train, y_train = augment_data_multiclass(X_train, y_train)

# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Fit label encoder and transform string column
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.transform(y_test)

model_name = 'gpt2'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2ForSequenceClassification.from_pretrained(model_name, num_labels=3)

# Add pad_token to tokenizer and model
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = model.config.eos_token_id

# train the model
model = train_model(model, tokenizer, 10, 4e-5, 16, X_train, y_train, use_preprocess=False)

# save the model, make it "CPU-friendly" first
model = model.to('cpu')
save_model(model, 'base-gpt-model.pth')

# get the score
f1, acc = eval_model(model, tokenizer, X_test, y_test)
print(f1, acc)




