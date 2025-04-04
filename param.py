"""Params for AAD."""

PAD = 0
UNK = 1
SOS = 2
EOS = 3

# params for dataset and data loader
data_root = "data"

# params for source dataset
src_encoder_path = "source-encoder.pt"
src_classifier_path = "source-classifier.pt"

# params for target dataset
tgt_encoder_path = "target-encoder.pt"

# params for setting up models
model_root = "snapshots"
d_model_path = "critic.pt"

# params for training network
num_gpu = 3
manual_seed = None

# params for optimizing models
c_learning_rate = 5e-5
d_learning_rate = 1e-5

n_vocab = 30522
hidden_size = 768
intermediate_size = 3072
embed_dim = 300
kernel_num = 20
kernel_sizes = [3, 4, 5]
pretrain = True
embed_freeze = True
class_num = 2
dropout = 0.1
num_labels = 2
d_hidden_dims = 384
d_output_dims = 2
num_exits = 12


imdb_train_path = r"C:\Users\divya\Desktop\DAdEE-main\DAdEE-main\data\imdb\imdb.csv"
imdb_test_path = r"path to imdb test data"
qnli_train_path = r"path to qnli train data"
qnli_test_path = r"path to qnli test data"
sst_train_path = r"/home/iitb/Buffer_based_threshold/sst2_data/train.csv"
sst_test_path = r"/home/iitb/Buffer_based_threshold/sst2_data/dev.csv"
snli_train_path = r"path to snli train data"
snli_test_path = r"path to snli test data"

dataframe_save_path = r"/home/iitb/Buffer_based_threshold/csv_files/df.csv"


# Simulation Parameters
arrival_rate = 1  # Poisson arrival rate (λ)
max_queue_size = 10  # Maximum queue size
e_process_time = 0
c_process_time = 0.5
exit_layer = 4