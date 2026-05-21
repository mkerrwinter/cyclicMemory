training_dataset_size = 5000
input_dim = 10
width = 10
n_hidden = 3

n_params = (input_dim * width) + (n_hidden * width * width) + (width * 1)

print('Number of parameters in the model:', n_params)
print('Network size / training dataset size:', n_params / training_dataset_size)