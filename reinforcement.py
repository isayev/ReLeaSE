import torch.nn.functional as F


class Reinforcement(object):
    def __init__(self, generator, data):
        super(Reinforcement, self).__init__()
        self.generator = generator
        self.data = data
        raise NotImplementedError

    def get_reward(self, smiles):
        raise NotImplementedError

    def policy_gradient_replay(self, n_batch=100):

        rl_loss = 0
        reward = 0
        n_samples = 0

        for i in range(n_batch):

            hidden = self.generator.init_hidden()
            cell = self.generator.init_cell()
            stack = self.generator.initStack()

            seq = self.data.sample_from_replay_memory()
            inp = self.data.char_tensor(seq)
            cur_loss = 0

            for p in range(len(inp)-1):
                output, hidden, cell, stack = self.generator(inp[p], hidden, cell, stack)
                top_i = inp.data[p+1]
                log_dist = F.log_softmax(output)
                cur_loss += log_dist[0, top_i]
                if seq[p+1] == '>':
                    reward = self.get_reward(seq[1:-1])

            rl_loss += cur_loss * reward
            n_samples += 1

        rl_loss = -rl_loss / n_samples
        rl_loss.backward()
        self.generator.optimizer.step()

        return rl_loss.data[0]
