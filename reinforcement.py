import torch
import torch.nn.functional as F


class Reinforcement(object):
    def __init__(self, generator, predictor):
        super(Reinforcement, self).__init__()
        self.generator = generator
        self.predictor = predictor

    def get_reward(self, smiles, threshold, invalid_reward=-2.0):
        # Add continuous reward
        mol, prop, nan_smiles = self.predictor.predict([smiles])
        if len(nan_smiles) == 1:
            return invalid_reward
        if prop[0] >= threshold:
            return 1.0
        else:
            return -1.0

    def policy_gradient_replay(self, data, replay_memory, threshold, n_batch=100):

        rl_loss = 0
        reward = 0
        n_samples = 0
        self.generator.optimizer.zero_grad()

        for _ in range(n_batch):

            hidden = self.generator.init_hidden()
            if self.generator.has_cell:
                cell = self.generator.init_cell()
            if self.generator.has_stack:
                stack = self.generator.init_stack()

            seq = replay_memory.sample()[0]
            inp = data.char_tensor(seq)
            cur_loss = 0

            for p in range(len(inp)-1):
                if self.generator.has_stack and self.generator.has_cell:
                    output, hidden, cell, stack = self.generator(inp[p], hidden, cell, stack)
                elif self.generator.has_stack and not self.generator.has_cell:
                    output, hidden, stack = self.generator(inp[p], hidden, stack)
                elif not self.generator.has_stack and self.generator.has_cell:
                    output, hidden, cell = self.generator(inp[p], hidden, cell)
                elif not self.generator.has_stack and not self.generator.has_cell:
                    output, hidden = self.generator(inp[p], hidden)
                top_i = inp.data[p+1]
                log_dist = F.log_softmax(output, dim=1)
                cur_loss += log_dist[0, top_i]
                if seq[p+1] == '>':
                    reward = self.get_reward(seq[1:-1],  threshold)

            rl_loss += cur_loss * reward
            n_samples += 1

        rl_loss = -rl_loss / n_samples
        rl_loss.backward()
        self.generator.optimizer.step()

        return rl_loss.item()

    def policy_gradient(self, data,  threshold, prime_str='<', end_token='>', predict_len=200, temperature=0.8, n_batch=100):

        rl_loss = 0
        reward = 0
        n_samples = 0
        self.generator.zero_grad()
        
        for _ in range(n_batch):

            hidden = self.generator.init_hidden()
            if self.generator.has_cell:
                cell = self.generator.init_cell()
            if self.generator.has_stack:
                stack = self.generator.init_stack()

            prime_input = data.char_tensor(prime_str)
            predicted = prime_str

            # Use priming string to "build up" hidden state
            for p in range(len(prime_str)):
                if self.generator.has_stack and self.generator.has_cell:
                    _, hidden, cell, stack = self.generator(prime_input[p], hidden, cell, stack)
                elif self.generator.has_stack and not self.generator.has_cell:
                    _, hidden, stack = self.generator(prime_input[p], hidden, stack)
                elif not self.generator.has_stack and self.generator.has_cell:
                    _, hidden, cell = self.generator(prime_input[p], hidden, cell)
                elif not self.generator.has_stack and not self.generator.has_cell:
                    _, hidden = self.generator(prime_input[p], hidden)
            inp = prime_input[-1]

            cur_loss = 0
            for p in range(predict_len):

                if self.generator.has_stack and self.generator.has_cell:
                    output, hidden, cell, stack = self.generator(inp, hidden, cell, stack)
                elif self.generator.has_stack and not self.generator.has_cell:
                    output, hidden, stack = self.generator(inp, hidden, stack)
                elif not self.generator.has_stack and self.generator.has_cell:
                    output, hidden, cell = self.generator(inp, hidden, cell)
                elif not self.generator.has_stack and not self.generator.has_cell:
                    output, hidden = self.generator(inp, hidden)

                # Sample from the network as a multinomial distribution
                output_dist = output.data.view(-1).div(temperature).exp()
                top_i = torch.multinomial(output_dist, 1)[0]
                log_dist = F.log_softmax(output, dim=1)
                cur_loss += log_dist[0, top_i]

                # Add predicted character to string and use as next input
                predicted_char = data.all_characters[top_i]
                predicted += predicted_char
                inp = data.char_tensor(predicted_char)

                if predicted_char == end_token:
                    reward = self.get_reward(predicted[1:-1], threshold)
                else:
                    reward = -10

            if reward != 0.0:
                rl_loss += cur_loss * reward
                n_samples += 1

        rl_loss = -rl_loss / n_samples
        rl_loss.backward()
        self.generator.optimizer.step()

        return rl_loss.item()

    def transfer_learning(self, data, n_epochs, augment=False):
        _ = self.generator.fit(data, n_epochs, augment=augment)
