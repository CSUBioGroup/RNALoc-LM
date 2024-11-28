import fm
from pathlib import Path
import os
import numpy as np
from torch.utils.data import DataLoader
import pickle
from base import pretrained, RNATypeDataset, test

def load_test_data(data_dir, test_file):
    test_path = os.path.join(data_dir, test_file)
    with open(test_path, "rb") as f:
        test_tempseq = pickle.load(f)
        test_names = pickle.load(f)
        test_labels = pickle.load(f)
    
    test_seqs = [(seq_name, seq) for seq_name, seq in zip(test_names, test_tempseq)]
    
    return test_seqs, test_labels

def predict(rna_type, data_dir, test_file, pretrained_dir, device, test_embedding_file, logger, batch_size, model_path):
    test_seqs, test_labels = load_test_data(data_dir, test_file)
    num_class = len(set(test_labels))
    
    fm_model, alphabet = fm.pretrained.rna_fm_t12(Path(pretrained_dir, 'RNA-FM_pretrained.pth'))
    fm_model.to(device)
    fm_model.eval()

    test_embedding_path = os.path.join(data_dir, test_embedding_file)
    if os.path.exists(test_embedding_path):
        test_embeddings = np.load(test_embedding_path)
        logger.info("Test Embeddings loaded successfully.")
    else:
        test_embeddings = pretrained(fm_model, alphabet, test_seqs, device)
        np.save(test_embedding_path, test_embeddings)
        logger.info("Test Embeddings saved successfully.")

    test_metrics = []
    test_dataset = RNATypeDataset(test_embeddings, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)  # 设置 num_workers 和 pin_memory # False
 
    for i in range(5):
        logger.info(f"Start testing model {i+1}:")
        model_name = f'{model_path}_cv{i+1}.pkl'
        test_metrics.append(test(rna_type, test_loader, test_labels, model_name, device, logger, num_class))

    
    logger.info('\n===================================================================================================================')
    logger.info(f'Test metrics: {test_metrics}')
    test_means = np.mean(np.array(test_metrics), axis=0)
    logger.info(f'Test metrics mean: {test_means}')



