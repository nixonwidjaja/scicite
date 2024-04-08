import pandas as pd
import torch 
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.optim import Adam
from sklearn.metrics import f1_score, accuracy_score
from sklearn.utils import resample

def augment_data_multiclass(X, y):
    df = pd.concat([X, y], axis=1)
    majority_class_size = df['label'].value_counts().max()
    upsampled_dataframes = []
    for class_label in df['label'].unique():
        class_df = df[df['label'] == class_label]
        if len(class_df) < majority_class_size:
            class_df_upsampled = resample(class_df, replace=True, n_samples=majority_class_size, random_state=10)
            upsampled_dataframes.append(class_df_upsampled)
        else:
            upsampled_dataframes.append(class_df)
    upsampled_df = pd.concat(upsampled_dataframes)
    return upsampled_df['string'], upsampled_df['label']

def cleaning(text):
    stop_words = stopwords.words('english')
    text = text.lower()
    text = ' '.join(x for x in text.split() if x not in stop_words)
    return text

def lemmatize(text):
    lemmatizer = WordNetLemmatizer()
    words = []
    for x in text.split():
        x = lemmatizer.lemmatize(x)
        words.append(x)
    text = ' '.join(words)
    return text

def preprocessing(text):
    # Tokenization
    tokenizer = RegexpTokenizer(r'[a-zA-Z0-9]+')
    text = cleaning(text)
    text = lemmatize(text)
    text = ' '.join(tokenizer.tokenize(text))
    return text

# train the model for a given number of epochs
def train_model(model, tokenizer, num_epoch, learning_rate, batch_size, X_train, y_train):
    # Encode the training data
    encoded_data_train = tokenizer.batch_encode_plus(
        X_train,
        add_special_tokens=True, 
        return_attention_mask=True, 
        pad_to_max_length=True, 
        max_length=512, 
        return_tensors='pt'
    )
    labels_train = torch.tensor(y_train)

    # Create data loader for training
    dataset_train = TensorDataset(encoded_data_train['input_ids'], encoded_data_train['attention_mask'], labels_train)
    dataloader_train = DataLoader(dataset_train, sampler=RandomSampler(dataset_train), batch_size=batch_size)

    # Connect to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Define optimizer for training data
    optimizer = Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epoch):
        model.train()

        curr_total_loss = 0.
        count = 0
        
        for train_batch in dataloader_train:
            optimizer.zero_grad()

            id, mask, label = train_batch
            id = id.to(device)
            mask = mask.to(device)
            label = label.to(device)

            outputs = model(id, attention_mask=mask, labels=label)

            loss = outputs.loss

            curr_total_loss += loss.item()
            count += 1

            loss.backward()
            
            optimizer.step()

        avg_loss = curr_total_loss / count
        print(epoch, avg_loss)       
    
    return model 

# return f1 macro and accuracy of the model
def eval_model(model, tokenizer, X_test, y_test):
    encoded_data_test = tokenizer.batch_encode_plus(
        X_test,
        add_special_tokens=True, 
        return_attention_mask=True, 
        pad_to_max_length=True, 
        max_length=512, 
        return_tensors='pt'
    )
    labels_test = torch.tensor(y_test)

    # Create data loader for test data
    batch_size = 16
    test_dataset = TensorDataset(encoded_data_test['input_ids'], encoded_data_test['attention_mask'], labels_test)
    test_dataloader = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset), batch_size=batch_size)

    # Connect to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Evaluate the model
    model.eval()
    predictions = []
    labels = []
    with torch.no_grad():
        for test_batch in test_dataloader:
            id, mask, label = test_batch
            id = id.to(device)
            mask = mask.to(device)
            label = label.to(device)

            outputs = model(id, attention_mask=mask, labels=label)
            logits = outputs.logits
            _, prediction  = torch.max(logits, dim=1)

            predictions.extend(prediction.tolist())
            labels.extend(label.tolist())
            
    f1 = f1_score(labels, predictions, average='macro')
    acc = accuracy_score(labels, predictions)

    return f1, acc


def save_model(model, save_path):
    torch.save(model.state_dict(), save_path)