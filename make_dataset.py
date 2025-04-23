import torch
from torch.utils.data import DataLoader, TensorDataset
import pickle

def generate_dataset(N, w_input):
    return torch.randn(N, w_input)

def create_dataloader(N, w_input, batch_size):
    dataset = generate_dataset(N, w_input)
    threshold = torch.mean(torch.norm(dataset, dim=1))
    targets = (torch.norm(dataset, dim=1) > threshold).float().unsqueeze(1)
    targets[targets==0] = -1
    tensor_dataset = TensorDataset(dataset, targets)
    dataloader = DataLoader(tensor_dataset, batch_size=batch_size, shuffle=True)
    return dataloader, tensor_dataset

if __name__=='__main__':
    # Example usage
    N = 1000  # Number of vectors
    w_input = 4 # Length of vectors == input layer width
    batch_size = 10
    train_dataloader, train_dataset = create_dataloader(N, w_input, batch_size)

    N = 40  # Number of vectors
    w_input = 4 # Length of vectors == input layer width
    batch_size = 10
    test_dataloader, test_dataset = create_dataloader(N, w_input, batch_size)

    # Save for later use
    outpath = '/Users/maxkerrwinter/Documents/python/cyclicMemory/dataset/train_dataset.pkl'
    with open(outpath, 'wb') as outfile:
        pickle.dump(train_dataset, outfile)

    outpath = '/Users/maxkerrwinter/Documents/python/cyclicMemory/dataset/test_dataset.pkl'
    with open(outpath, 'wb') as outfile:
        pickle.dump(test_dataset, outfile)