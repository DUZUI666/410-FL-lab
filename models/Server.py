import copy
import torch


class Server:
    def __init__(self, args=None):
        self.global_model = None
        self.updates = []
        self.history_losses = []
        self.history_weights = []
        self.args = None

    def update_clear(self):
        self.updates = []

    def FedAvg(self, w):
        w_avg = copy.deepcopy(w[0])
        for k in w_avg.keys():
            for i in range(1, len(w)):
                w_avg[k] += w[i][k]
            w_avg[k] = torch.div(w_avg[k], len(w))
        self.global_model.load_state_dict(w_avg)
        return w_avg

