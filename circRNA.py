import torch
from logger import setup_logger
from base import seed_torch, run

data_dir = 'data'
output_dir = 'logs'
pretrained_dir = 'pretrained'
seed = 522
gpu_id = 0

rna_type = 'circRNA'
train_file = 'circTrain.pkl'
test_file = 'circTest.pkl'
embedding_file = 'circ_train_embeddings.npy'
test_embedding_file = 'circ_test_embeddings.npy'
model_path = 'models/circRNA/circRNA_textcnnbilstmAtten'
log_name = 'circ_textcnnbilstmAtten'
#===============================================
epochs = 60
batch_size = 32
lr = 1e-3
#===============================================
input_dim = 640
num_filters = 50 
filter_sizes = [3, 4, 5]
dropout = 0.5
hidden_dim = 32
num_layers = 2
num_heads = 4

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

print(f'using {device} device')

logger = setup_logger(log_name, output_dir, log_name, 0)
if device == "cuda":
    torch.cuda.set_device(gpu_id)
    logger.info("Using GPU ID: {}".format(gpu_id))
else:
    logger.info("Using CPU")

def main():
    run(rna_type, data_dir, train_file, test_file, pretrained_dir, device, embedding_file, test_embedding_file, logger, batch_size, lr, epochs, model_path, seed, input_dim, num_filters, filter_sizes, hidden_dim, num_layers, dropout, num_heads)
    

if __name__ == '__main__':
    seed_torch(seed)
    main()

