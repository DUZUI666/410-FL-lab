import random
import matplotlib
import copy
import torch
from utils.options import args_parser
import utils.EnvInit as env
from models.Nets import MLP, CNNMnist, CNNCifar
from utils.DataTest import test_img
from models.Client import Client
from models.Server import Server
matplotlib.use('Agg')


def FL_train(args):
    for iter in range(args.epochs):
        train_clients = []
        loss_locals = []
        w_locals = []
        if args.all_clients:
            for i in range(args.num_users):
                train_clients.append(Clients[i])
        else:
            sample = random.sample(range(0, args.num_users), max(int(args.frac * args.num_users), 1))
            for i in range(len(sample)):
                train_clients.append(Clients[sample[i]])

        print(f"clients of this epoch: {[train_clients[i].index for i in range(len(train_clients))]}")
        for client in train_clients:
            client.update_local_model(Server.global_model)
            w, loss = client.local_train()
            loss_locals.append(copy.deepcopy(loss))
            w_locals.append(w)

        Server.FedAvg(w_locals)

        for i in range(len(train_clients)):
            w_locals.append(train_clients[i].local_model.state_dict())
        loss_avg = sum(loss_locals) / len(loss_locals)
        print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
        Server.history_losses.append(loss_avg)


if __name__ == '__main__':
    # config args in options.py or set args here
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    # init data
    dataset_train, dataset_test, dict_users = env.generate_data(args)

    # init server
    Server = Server(args)

    if args.model == 'cnn' and args.dataset == 'cifar':
        Server.global_model = CNNCifar(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        Server.global_model = CNNMnist(args=args).to(args.device)
    elif args.model == 'mlp':
        len_in = 1
        for x in dataset_train[0][0].shape:
            len_in *= x
        Server.global_model = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)
    else:
        exit('Error: unrecognized model')

    print(Server.global_model)

    # init clients
    Clients = []

    for i in range(args.num_users):
        Clients.append(Client(args, dataset=dataset_train, idxs=dict_users[i], index=i))
        Clients[i].local_model = Server.global_model

    # train
    FL_train(args)

    # testing
    test_net = Server.global_model
    test_net.eval()
    acc_train, loss_train = test_img(test_net, dataset_train, args)
    acc_test, loss_test = test_img(test_net, dataset_test, args)
    print("Training accuracy: {:.2f}".format(acc_train))
    print("Testing accuracy: {:.2f}".format(acc_test))







