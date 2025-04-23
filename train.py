import torch
from torch.utils.data import DataLoader, TensorDataset
import pickle
from copy import deepcopy
from NN import FeedForwardNN
from matplotlib import pyplot as plt
import time
import os
import argparse

def make_quadratic_hinge_loss():
    def quadratic_hinge(output, target):
        Delta = 1.0 - target * output
        zeros = torch.zeros_like(Delta)

        max_Delta = torch.max(zeros, Delta)

        sq_max_Delta = max_Delta * max_Delta

        return 0.5 * torch.mean(sq_max_Delta)

    return quadratic_hinge


def train(dataloader, model, loss_fn, optimizer, device):
    model.train()

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)
        for param in model.parameters():
            param.grad = None
        loss.backward()
        optimizer.step()

def evaluate_loss(model, dataloader, loss_fn, device):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0.0
    with torch.no_grad():  # Disable gradient computation
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss = loss_fn(pred, y)
            total_loss += loss.item()
    return total_loss / len(dataloader)

def set_decreasing_save_rate(t):
    if t<10:
        return 1
    if t<100:
        return 10
    if t<1000:
        return 100
    if t<1e4:
        return 1000

    return 1e4

if __name__=='__main__':
    start = time.time()

    ### Set up parameters ###
    if torch.cuda.is_available():
        device = "cuda"
        cpu_or_gpu = 'gpu'
        print('Training with CUDA GPU.')
    else:
        device = "cpu"
        cpu_or_gpu = 'cpu'
        print('Training with CPU.')

    cwd = os.getcwd()
    if cwd[:20] == '/Users/maxkerrwinter':
        # Running on my laptop
        base_path = '/Users/maxkerrwinter/Documents/python/'
        proj_path = base_path
        scratch_path = None
        running_locally = True
    else:
        # Running on Snellius (or equiv.)
        base_path = '/home/mkerrwinter/python/'
        proj_path = '/home/mkerrwinter/link_project_space/python/'
        scratch_path = '/scratch-shared/mkerrwinter/python/'
        running_locally = False

    parser = argparse.ArgumentParser(description="Train feed forward networks.")
    parser.add_argument('--N_epochs', type=int, default=10, help="Number of training epochs (default: 10).")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size (default: 64).")
    parser.add_argument('--learning_rate', type=float, default=0.0001, help="Learning rate (default: 0.0001).")
    parser.add_argument('--N_reps', type=int, default=1, help="Number of repeats (default: 1).")
    parser.add_argument('--depth', type=int, default=3, help="Depth of network (default: 3).")
    parser.add_argument('--width_strings', action='append', help="Width to add to the list of widths (required).",
                        required=True)

    args = parser.parse_args()
    N_epochs = args.N_epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    N_reps = args.N_reps
    depth = args.depth
    width_strings = args.width_strings
    widths = [int(width_string) for width_string in width_strings]

    ### Load data ###
    train_inpath = '{}dataset/train_dataset.pkl'.format(proj_path)
    test_inpath = '{}dataset/test_dataset.pkl'.format(proj_path)

    with open(train_inpath, 'rb') as infile:
        train_dataset = pickle.load(infile)

    with open(test_inpath, 'rb') as infile:
        test_dataset = pickle.load(infile)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=True)

    ### Define models, optimiser, loss function ###
    input_size = train_dataset[0][0].shape[0]
    output_size = 1
    models_by_width = {}
    for width in widths:
        model_list = []
        for rep in range(N_reps):
            model = FeedForwardNN(input_size, width, depth, output_size).to(device)
            model_list.append(model)

        models_by_width[width] = deepcopy(model_list)

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    loss_fn = make_quadratic_hinge_loss()

    ### Train and save ###
    for width, model_list in models_by_width.items():
        for rep in range(len(model_list)):
            losses = {}
            model = model_list[rep]
            for t in range(N_epochs):
                save_rate = set_decreasing_save_rate(t)
                if t%save_rate==0:
                    print('Training Width: {}, Depth: {}, Rep: {}, Epoch: {}'.format(width, depth, rep, t))
                    current_loss = evaluate_loss(model, train_dataloader, loss_fn, device)
                    losses[t] = current_loss
                    outpath = '{}models/{}/FF_w{}_d{}_rep{}_epoch{}.pth'.format(proj_path, cpu_or_gpu, width, depth, rep, t)
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'epoch': t,
                        'loss': current_loss
                    }, outpath)

                train(train_dataloader, model, loss_fn, optimizer, device)

            # Save the final model if not already saved
            if t%save_rate!=0:
                current_loss = evaluate_loss(model, train_dataloader, loss_fn, device)
                losses[t] = current_loss
                outpath = '{}models/{}/FF_w{}_d{}_rep{}_epoch{}.pth'.format(proj_path, cpu_or_gpu, width, depth, rep, t)
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': t,
                    'loss': current_loss
                }, outpath)

            # Save the loss plot
            loss_plot_path = '{}plots/loss_w{}_d{}_rep{}.pdf'.format(proj_path, width, depth, rep)
            plt.figure(figsize=(12, 6))
            plt.plot(list(losses.keys()), list(losses.values()))
            plt.yscale('log')
            plt.xlabel('Epochs')
            plt.ylabel('Training Loss (log scale)')
            plt.title('Loss vs Epochs (Width: {}, Depth: {}, Rep: {})'.format(width, depth, rep))
            plt.savefig(loss_plot_path)
            plt.close()

    print('Running time = ', time.time()-start)


