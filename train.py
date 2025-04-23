import torch
from torch.utils.data import DataLoader, TensorDataset
import pickle
from copy import deepcopy
from NN import FeedForwardNN
from matplotlib import pyplot as plt
import time

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

if __name__=='__main__':
    start = time.time()
    if torch.cuda.is_available():
        device = "cuda"
        print('Training with CUDA GPU.')
    else:
        device = "cpu"
        print('Training with CPU.')

    basepath = '/Users/maxkerrwinter/Documents/python/cyclicMemory/'

    ### Load data ###
    train_inpath = '{}dataset/train_dataset.pkl'.format(basepath)
    test_inpath = '{}dataset/test_dataset.pkl'.format(basepath)

    with open(train_inpath, 'rb') as infile:
        train_dataset = pickle.load(infile)

    with open(test_inpath, 'rb') as infile:
        test_dataset = pickle.load(infile)

    batch_size = 200
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=True)

    ### Define models, optimiser, loss function ###
    input_size = train_dataset[0][0].shape[0]
    depth = 3
    output_size = 1
    # widths = [2, 8, 32]
    widths = [32]
    N_reps = 1
    models_by_width = {}
    for width in widths:
        model_list = []
        for rep in range(N_reps):
            model = FeedForwardNN(input_size, width, depth, output_size).to(device)
            model_list.append(model)

        models_by_width[width] = deepcopy(model_list)

    learning_rate = 1e-3
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    loss_fn = make_quadratic_hinge_loss()

    ### Train and save ###
    N_epochs = 1000
    save_rate = 100
    for width, model_list in models_by_width.items():
        for rep in range(len(model_list)):
            losses = {}
            model = model_list[rep]
            for t in range(N_epochs):
                if t%save_rate==0:
                    print('Training Width: {}, Depth: {}, Rep: {}, Epoch: {}'.format(width, depth, rep, t))
                    current_loss = evaluate_loss(model, train_dataloader, loss_fn, device)
                    losses[t] = current_loss
                    outpath = '{}models/FF_w{}_d{}_rep{}_epoch{}.pth'.format(basepath, width, depth, rep, t)
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'epoch': t,
                        'loss': current_loss
                    }, outpath)

                train(train_dataloader, model, loss_fn, optimizer, device)

            # Save the final model if not already saved
            if t%save_rate!=0:
                losses[t] = evaluate_loss(model, train_dataloader, loss_fn, device)
                outpath = '{}models/FF_w{}_d{}_rep{}_epoch{}.pth'.format(basepath, width, depth, rep, t)
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': t,
                    'loss': current_loss
                }, outpath)

            # Save the loss plot

            loss_plot_path = '{}plots/loss_w{}_d{}_rep{}.pdf'.format(basepath, width, depth, rep)
            plt.figure(figsize=(12, 6))
            plt.plot(list(losses.keys()), list(losses.values()))
            plt.yscale('log')
            plt.xlabel('Epochs')
            plt.ylabel('Loss (log scale)')
            plt.title('Loss vs Epochs (Width: {}, Depth: {}, Rep: {})'.format(width, depth, rep))
            plt.savefig(loss_plot_path)
            plt.close()

    print('Running time = ', time.time()-start)


