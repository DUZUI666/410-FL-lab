import copy

import torch
from torch.utils.data import DataLoader, Dataset
from torch import nn


class Client:
    class DatasetSplit(Dataset):
        def __init__(self, dataset, idxs):
            self.dataset = dataset
            self.idxs = list(idxs)

        def __len__(self):
            return len(self.idxs)

        def __getitem__(self, item):
            image, label = self.dataset[self.idxs[item]]
            return image, label

    def __init__(self, args, dataset=None, idxs=None, index=None):
        self.local_DataLoader = DataLoader(self.DatasetSplit(dataset, idxs), batch_size=args.local_bs, shuffle=True)
        self.local_model = None
        self.history_weights = []
        self.history_losses = []
        self.index = index
        self.args = args

    # client local train
    def local_train(self):
        self.local_model.to(self.args.device)
        self.local_model.train()
        optimizer = torch.optim.SGD(self.local_model.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        loss_func = nn.CrossEntropyLoss()

        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.local_DataLoader):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                self.local_model.zero_grad()
                log_probs = self.local_model(images)
                loss = loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.local_DataLoader.dataset),
                              100. * batch_idx / len(self.local_DataLoader), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        w = self.local_model.state_dict()

        self.history_losses.append(epoch_loss)
        self.history_weights.append(w)

        return w, sum(epoch_loss) / len(epoch_loss)

    # update local model
    def update_local_model(self, net):
        self.local_model = copy.deepcopy(net)