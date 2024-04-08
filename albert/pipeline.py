import pandas as pd
import torch 
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from transformers import AlbertTokenizer, AlbertForSequenceClassification
from utils import train_model, eval_model, save_model, augment_data_multiclass, preprocessing

train_df = pd.read_json('../train.jsonl', lines=True)
X_train = train_df['string']
y_train = train_df['label']

test_df = pd.read_json('../test.jsonl', lines=True)
X_test = test_df['string']
y_test = test_df['label']

# Upsample the training data
X_train, y_train = augment_data_multiclass(X_train, y_train)

# Preprocess X_train and X_test first
X_train = X_train.apply(preprocessing)
X_test = X_test.apply(preprocessing)

# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Fit label encoder and transform string column
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.transform(y_test)

model_name = 'albert-base-v2'
tokenizer = AlbertTokenizer.from_pretrained(model_name)
model = AlbertForSequenceClassification.from_pretrained(model_name, num_labels=3)

# Only train the classifier and embeddings layer
'''
for param in model.parameters():
    param.requires_grad = False
for param in model.classifier.parameters():
    param.requires_grad = True
for param in model.albert.embeddings.parameters():
    param.requires_grad = True
'''

# train the model
model = train_model(model, tokenizer, 10, 4e-5, 16, X_train, y_train)

# save the model, make it "CPU-friendly" first
model = model.to('cpu')
save_model(model, 'base-albert-model.pth')

# get the score
f1, acc = eval_model(model, tokenizer, X_test, y_test)
print(f1, acc)
