import fm  # for development with RNA-FM
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import numpy as np
from tqdm import tqdm 
import os
import random
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import StratifiedKFold
import pickle
import pandas as pd
from models import TextCNNBilstmWithAttention

def seed_torch(seed=2024):
    '''
        Setting the random seed
    '''
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

class RNATypeDataset(Dataset):
    '''
        Defining RNA data structures
    '''
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]

def load_data(data_dir, train_file, test_file):
    train_path = os.path.join(data_dir, train_file)
    test_path = os.path.join(data_dir, test_file)
    with open(train_path, "rb") as f:
        train_tempseq = pickle.load(f)
        train_names = pickle.load(f)
        train_labels = pickle.load(f)
    with open(test_path, "rb") as f:
        test_tempseq = pickle.load(f)
        test_names = pickle.load(f)
        test_labels = pickle.load(f)
    
    train_seqs = [(seq_name, seq) for seq_name, seq in zip(train_names, train_tempseq)]
    test_seqs = [(seq_name, seq) for seq_name, seq in zip(test_names, test_tempseq)]
    
    return train_seqs, train_labels, test_seqs, test_labels

def pretrained(fm_model, alphabet, seqs, device):
    '''
        Load the pre-trained model and get the embedding results
    '''
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        fm_model = nn.DataParallel(fm_model)
    
    fm_model.to(device)  
    batch_converter = alphabet.get_batch_converter()
    max_len = fm_model.module.embed_positions.weight.size(0) - 2 if isinstance(fm_model, nn.DataParallel) else fm_model.embed_positions.weight.size(0) - 2

    token_embeddings = np.zeros((len(seqs), max_len, 640))

    chunk_size = 50

    for i in tqdm(range(0, len(seqs), chunk_size)):
        data = seqs[i:i+chunk_size]
        batch_labels, batch_strs, batch_tokens = batch_converter(data)
        
        if batch_tokens.shape[1] > max_len:
            batch_tokens = batch_tokens[:, :max_len]

        with torch.no_grad():
            results = fm_model(batch_tokens.to(device), repr_layers=[12])

        emb = results["representations"][12].cpu().numpy()
        token_embeddings[i:i+chunk_size, :emb.shape[1], :] = emb
    return token_embeddings

def class_distribution(labels):
    classes, counts = np.unique(labels, return_counts=True)
    distribution = counts / counts.sum()
    return dict(zip(classes, distribution))

def evaluate(y_trues, y_preds):
    if isinstance(y_trues, torch.Tensor):
        y_trues = y_trues.detach().cpu().numpy()
    if isinstance(y_preds, torch.Tensor):
        y_preds = y_preds.detach().cpu().numpy()

    acc = accuracy_score(y_trues, y_preds)
    f1 = f1_score(y_trues, y_preds, average='macro')
    precision = precision_score(y_trues, y_preds, average='macro')
    recall = recall_score(y_trues, y_preds, average='macro')
    
    report = classification_report(y_trues, y_preds, output_dict=True)
    print(report)
    return acc, f1, precision, recall

class FocalLoss(nn.Module):
    '''
        Defining the loss function
    '''
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean', device='cpu'):
        super(FocalLoss, self).__init__()
        self.alpha = torch.tensor(alpha).to(device) if alpha is not None else torch.tensor(1.0).to(device)
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        BCE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        alpha = self.alpha[targets]
        F_loss = alpha * ((1-pt)**self.gamma) * BCE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss

def train(rna_type, model, train_loader, val_loader, lr, epochs, model_path, device, logger, i, scaler, num_class):
    '''
        Model training process
    '''
    all_labels = []
    for _, labels in train_loader:
        all_labels.extend(labels.tolist()) # 
    class_ratios = list(class_distribution(all_labels).values())
    weights = [1.0 / ratio for ratio in class_ratios]
    normalized_weights = torch.tensor([w / sum(weights) for w in weights], dtype=torch.float)

    criterion = FocalLoss(alpha=normalized_weights, gamma=1.5, reduction='mean', device=device) if rna_type == 'lncRNA' else nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    # Apply gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, verbose=True)

    best_model, best_val_acc = model, 0.0
    best_val_metrics = []
    for epoch in tqdm(range(epochs)):
        logger.info(f'Epoch [{epoch+1}/{epochs}]')
        model.train()
        train_losses, train_preds, train_targets, train_probs = [], [], [], []
        for x, y in train_loader:
            x, y = x.to(device).float(), y.to(device).long()
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                output, attention_weights = model(x)
                y_probs = torch.sigmoid(output).squeeze()
                y_preds = (y_probs > 0.5).float()
                y_preds = torch.argmax(y_preds, dim=1)
                loss = criterion(output, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_losses.append(loss.item())
            train_preds.append(y_preds.detach())
            train_targets.append(y.detach())
            train_probs.append(y_probs.detach())

        val_losses, val_preds, val_targets, val_probs = [], [], [], []
        model.eval()
        for x, y in val_loader:
            x, y = x.to(device).float(), y.to(device).long()
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    output, attention_weights = model(x)
                    y_probs = torch.sigmoid(output).squeeze()
                    y_preds = (y_probs > 0.5).float()
                    y_preds = torch.argmax(y_preds, dim=1)
                    loss = criterion(output, y)
            val_losses.append(loss.item())
            val_preds.append(y_preds.detach())
            val_targets.append(y.detach())
            val_probs.append(y_probs.detach())

        train_preds, train_targets, train_probs = torch.cat(train_preds), torch.cat(train_targets), torch.cat(train_probs)
        val_preds, val_targets, val_probs = torch.cat(val_preds), torch.cat(val_targets), torch.cat(val_probs)

        train_metrics = evaluate(train_targets, train_preds)
        val_metrics = evaluate(val_targets, val_preds)
        logger.info(f"Train metrics: {train_metrics}")
        logger.info(f"Train losses: {np.mean(train_losses)}")
        logger.info(f"Validation metrics: {val_metrics}")

        if val_metrics[0] > best_val_acc:
            best_model = model
            best_val_acc = val_metrics[0]
            best_val_metrics = val_metrics
            torch.save(best_model, f'{model_path}_cv{i+1}.pkl')
            print("Better model saved")

        scheduler.step(np.mean(val_losses))
        torch.cuda.empty_cache()
    
    return best_val_acc, best_val_metrics

def test(rna_type, test_loader, y_test, model_name, device, logger, num_class):
    '''
        Model testing process
    '''
    test_preds, test_probs = [], []
    model = torch.load(model_name, map_location=device).to(device)
    model.eval()
    test_attention_weights = []

    for x, _ in test_loader:
        x = x.to(device).float()
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                output, attention_weights = model(x)
                y_probs = torch.sigmoid(output).squeeze().detach()
                y_preds = (y_probs > 0.5).float()
                y_preds = torch.argmax(y_preds, dim=1)  # 将二维转为一维
                
        test_preds.append(y_preds.cpu().numpy())
        test_probs.append(y_probs.cpu().numpy())
        test_attention_weights.append(attention_weights.cpu().numpy())

    test_preds = np.concatenate(test_preds)
    logger.info(f"Test preds: {test_preds}")
    test_probs = np.concatenate(test_probs)
    test_metrics = evaluate(y_test, test_preds)
    logger.info(f"Test metrics: {test_metrics}")
    test_attention_weights = np.concatenate(test_attention_weights).squeeze()
    pd.DataFrame(test_attention_weights).to_csv(f'{rna_type}_test_textcnnbilstmattention_weights.csv', mode='a')
    return test_metrics


def run(rna_type, data_dir, train_file, test_file, pretrained_dir, device, embedding_file, test_embedding_file, logger, batch_size, lr, epochs, model_path, seed, input_dim, num_filters, filter_sizes, hidden_dim, num_layers, dropout, num_heads):
    train_seqs, train_labels, test_seqs, test_labels = load_data(data_dir, train_file, test_file)
    num_class = len(set(train_labels))

    fm_model, alphabet = fm.pretrained.rna_fm_t12(Path(pretrained_dir, 'RNA-FM_pretrained.pth'))
    fm_model.to(device)
    fm_model.eval()

    embedding_path = os.path.join(data_dir, embedding_file)
    if os.path.exists(embedding_path):
        train_embeddings = np.load(embedding_path)
        logger.info("Train Embeddings loaded successfully.")
    else:
        train_embeddings = pretrained(fm_model, alphabet, train_seqs, device)
        np.save(embedding_path, train_embeddings)
        logger.info("Train Embeddings saved successfully.")

    test_embedding_path = os.path.join(data_dir, test_embedding_file)
    if os.path.exists(test_embedding_path):
        test_embeddings = np.load(test_embedding_path)
        logger.info("Test Embeddings loaded successfully.")
    else:
        test_embeddings = pretrained(fm_model, alphabet, test_seqs, device)
        np.save(test_embedding_path, test_embeddings)
        logger.info("Test Embeddings saved successfully.")

    train_dist = class_distribution(train_labels)
    test_dist = class_distribution(test_labels)
    print(train_dist)
    print(test_dist)
    # print(train_embeddings.shape, test_embeddings.shape)
    # print(train_labels)

    val_metrics, test_metrics = [], []
    test_dataset = RNATypeDataset(test_embeddings, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)  

    scaler = torch.cuda.amp.GradScaler()

    skf = StratifiedKFold(n_splits=5, random_state=seed, shuffle=True)
    for i, (train_index, val_index) in enumerate(skf.split(train_embeddings, train_labels)): 
        model = TextCNNBilstmWithAttention(num_class, input_dim, num_filters, filter_sizes, hidden_dim, num_layers, dropout, num_heads).to(device)

        logger.info(f"Start training CV fold {i+1}:")
        train_sampler, val_sampler = SubsetRandomSampler(train_index), SubsetRandomSampler(val_index)

        train_dataset = RNATypeDataset(train_embeddings, train_labels) 
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=2, pin_memory=True)
        val_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=val_sampler, num_workers=2, pin_memory=True)
        
        best_val_acc, best_val_metrics = train(rna_type, model, train_loader, val_loader, lr, epochs, model_path, device, logger, i, scaler, num_class)
        val_metrics.append(best_val_metrics)

        logger.info(f"Start testing model {i+1}:")
        model_name = f'{model_path}_cv{i+1}.pkl'
        test_metrics.append(test(rna_type, test_loader, test_labels, model_name, device, logger, num_class))
    
    logger.info('\n===================================================================================================================')
    logger.info(f'Validation metrics: {val_metrics}')
    val_means = np.mean(np.array(val_metrics), axis=0)
    logger.info(f'Validation metrics mean: {val_means}')
    
    logger.info(f'Test metrics: {test_metrics}')
    test_means = np.mean(np.array(test_metrics), axis=0)
    logger.info(f'Test metrics mean: {test_means}')
