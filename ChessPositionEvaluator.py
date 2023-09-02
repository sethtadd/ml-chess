from time import time

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn, optim
from torch.utils.data import random_split, TensorDataset


class ResNetEvaluator(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        # self.model = resnet18(in_channels, n_classes)
        # self.encoder = nn.Sequential(
        #     nn.Conv2d(in_channels=12, out_channels=32, kernel_size=3, padding=1, stride=1),
        #     nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, stride=1),
        #     nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=0, stride=1),
        #     nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=0, stride=1),
        # )
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=13, out_channels=32, kernel_size=1, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, padding=2, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=0, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=0, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
        )
        # self.encoder = nn.Sequential(
        #     nn.Conv2d(in_channels=12, out_channels=32, kernel_size=3, padding=1, stride=1),
        #     nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, stride=1),
        #     nn.Conv2d(in_channels=64, out_channels=128, kernel_size=2, padding=0, stride=1),
        #     nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1, padding=0, stride=1),
        #     nn.Conv2d(in_channels=64, out_channels=16, kernel_size=1, padding=0, stride=1),
        # )
        self.decoder = nn.Sequential(
            nn.LazyLinear(out_features=512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.LazyLinear(out_features=1),
        )
        self.loss_fn = nn.MSELoss()
        # self.loss_fn = nn.NLLLoss()
        self.optimizer = optim.SGD(self.parameters(), lr=0.1, momentum=0.5)
        # self.optimizer = optim.Adam(self.parameters(), lr=0.001)

        self.score_compression = 1

    def forward(self, x):
        # turn_indicators = x[:, -1, 0, 0].view(-1, 1)
        # piece_values = x[:, -1, 0, 1].view(-1, 1)
        # x = x[:, :-1, :, :]
        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1)
        # x = torch.cat((x, turn_indicators, piece_values), dim=1)
        x = self.decoder(x)
        x = torch.sigmoid(x / self.score_compression)
        return x

    def train_step(self, x_train, y_train):
        self.train()
        yhat = self(x_train)
        y_train = torch.sigmoid(y_train / self.score_compression)
        loss = self.loss_fn(input=yhat, target=y_train)
        # l2 regularization
        for name, tensor in self.named_parameters():
            loss += 0.01 * torch.flatten(tensor).norm(2) ** 2
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def evaluate(self, x_test, y_test):
        with torch.no_grad():
            self.eval()
            # print('shape x_test:', x_test.shape)
            yhat = self(x_test)
            y_test = torch.sigmoid_(y_test / self.score_compression)
            # print('y shape:', y_test.shape)
            # print('yhat shape:', yhat.shape)
            loss = self.loss_fn(input=yhat, target=y_test)
            # l2 regularization
            for name, tensor in self.named_parameters():
                loss += 0.01 * torch.flatten(tensor).norm(2) ** 2
            return loss.item()

    def train_loop(self, num_epochs, train_loader, test_loader):
        train_losses = []
        test_losses = []

        self.to(self.device)

        start = time()

        for epoch in range(num_epochs):

            if epoch % 10 == 0:
                print(f'{100 * epoch / num_epochs}% trained (epoch {epoch})')
                if not epoch == 0:
                    print(f'train loss: {train_losses[-1]}')
                    print(f'test loss: {test_losses[-1]}')

            epoch_train_loss = 0
            epoch_test_loss = 0

            # evaluating
            for x_batch, y_batch in test_loader:
                # send data to device
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                # calculate loss
                test_loss = self.evaluate(x_batch, y_batch)
                epoch_test_loss += test_loss

            # training
            for x_batch, y_batch in train_loader:
                # send data to device
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                # calculate loss
                train_loss = self.train_step(x_batch, y_batch)
                epoch_train_loss += train_loss

            train_losses.append(epoch_train_loss)
            test_losses.append(epoch_test_loss)

            print(f'completed epoch {epoch}')

        end = time()
        train_time = end - start

        return train_time, train_losses, test_losses


def main():
    device = 'cuda:0'
    batch_size = 1024
    epochs = 500

    # seeding
    seed = 42
    torch.manual_seed(seed)

    # load data
    print('Loading data')
    board_tensors = torch.tensor(np.array(np.load(file='./board_tensors.npy')), dtype=torch.float32)
    board_scores = torch.tensor(np.load(file='./board_scores.npy'), dtype=torch.float32)
    board_scores = torch.reshape(board_scores, (-1, 1))
    dataset = TensorDataset(board_tensors, board_scores)

    # train/test split
    train_len = int(0.9 * len(dataset))  # 90|10 split
    test_len = len(dataset) - train_len
    train_set, test_set = random_split(dataset=dataset, lengths=[train_len, test_len])

    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, num_workers=2,
                                               pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2,
                                              pin_memory=True)

    model = ResNetEvaluator(device=device)
    # model.load_state_dict(torch.load(r'.\last_net.pth'))

    # training loop
    print('training...')
    train_time, train_losses, test_losses = model.train_loop(
        num_epochs=epochs, train_loader=train_loader, test_loader=test_loader)
    print('finished training')
    print(f'finished training in {train_time} seconds')

    print(f'Saving model parameters')
    torch.save(model.state_dict(), r'.\last_net.pth')

    # normalize train loss
    norm_const = (len(test_loader) * batch_size) / (len(train_loader) * batch_size)
    train_losses_normalized = [x * norm_const for x in train_losses]

    # plot loss
    plt.figure(1)
    plt.plot(np.log(train_losses_normalized))
    plt.plot(np.log(test_losses))
    plt.title('chess evaluator LOG LOSS')
    plt.legend(['training loss', 'testing loss'])
    plt.xlabel('epochs')
    plt.ylabel('log loss')

    plt.figure(2)
    plt.plot(train_losses_normalized)
    plt.plot(test_losses)
    plt.title('chess evaluator')
    plt.legend(['training loss', 'testing loss'])
    plt.xlabel('epochs')
    plt.ylabel('loss')

    plt.show()


if __name__ == '__main__':
    main()
