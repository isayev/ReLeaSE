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

        for _ in range(n_batch):

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

    def policy_gradient(self, prime_str='<', end_token='>', predict_len=200, temperature=0.8, n_batch=100):

        rl_loss = 0
        reward = 0
        n_samples = 0

        for _ in range(n_batch):

            hidden = self.generator.init_hidden()
            cell = self.generator.init_cell()
            stack = self.generator.initStack()
            prime_input = self.data.char_tensor(prime_str)
            predicted = prime_str

            # Use priming string to "build up" hidden state
            for p in range(len(prime_str)):
                _, hidden, cell, stack = self.generator(prime_input[p], hidden, cell, stack)
            inp = prime_input[-1]

            cur_loss = 0
            for p in range(predict_len):
                output, hidden, cell, stack = self.generator(inp, hidden, cell, stack)

                # Sample from the network as a multinomial distribution
                output_dist = output.data.view(-1).div(temperature).exp()
                top_i = torch.multinomial(output_dist, 1)[0]
                log_dist = F.log_softmax(output)
                cur_loss += log_dist[0, top_i]

                # Add predicted character to string and use as next input
                predicted_char = self.data.all_characters[top_i]
                predicted += predicted_char
                inp = char_tensor(predicted_char)

                if predicted_char == end_token:
                    reward = self.get_reward(predicted[1:-1])
                else:
                    reward = -10

            if reward != 0.0:
                rl_loss += cur_loss * reward
                n_samples += 1

        rl_loss = -rl_loss / n_samples
        rl_loss.backward()
        self.generator.optimizer.step()

        return rl_loss.data[0]

    def transfer_learning(self):
        raise NotImplementedError



