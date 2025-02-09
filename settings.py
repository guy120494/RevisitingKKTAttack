import torch

global device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# dataset parameters
global im_size
im_size = 28

global padded_im_size
padded_im_size = 32

global C
C = 1

global input_ch
input_ch = 1

global output_dim
input_dim = 2

# Optimization hyperparameters

global lr_decay
lr_decay = 0.995

global lr
lr = 0.1

global epochs
epochs = 200000

global find_lambdas_epochs
find_lambdas_epochs = 2000

global num_samples
num_samples = 4

global batch_size
batch_size = num_samples

global momentum
momentum = 0.995

global weight_decay
weight_decay = 0  # 5e-15
