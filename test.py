import torch
from logger import setup_logger
from base import seed_torch
from predict import predict

data_dir = 'data'
output_dir = 'logs'
pretrained_dir = 'pretrained'
seed = 522
gpu_id = 0
batch_size = 64

rna_type = 'lncRNA'
test_file = 'lncTest.pkl'
test_embedding_file = 'lnc_test_embeddings.npy'
model_path = 'models/lncRNA/lncRNA_textcnnbilstmAtten'
log_name = 'lnc_test'
#===============================================
# rna_type = 'miRNA'
# test_file = 'miTest.pkl'
# test_embedding_file = 'mi_test_embeddings.npy'
# model_path = 'models/miRNA/miRNA_textcnnbilstmAtten'
# log_name = 'mi_test'
#===============================================
# rna_type = 'circRNA'
# test_file = 'circTest.pkl'
# test_embedding_file = 'circ_test_embeddings.npy'
# model_path = 'models/circRNA/circRNA_textcnnbilstmAtten'
# log_name = 'circ_test'


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
    predict(rna_type, data_dir, test_file, pretrained_dir, device, test_embedding_file, logger, batch_size, model_path)
    

if __name__ == '__main__':
    seed_torch(seed)
    main()
