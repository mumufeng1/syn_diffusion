import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import time
import collections
import os
import json

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Define the NgramLanguageModel class
class NgramLanguageModel(nn.Module):
    def __init__(self, n, samples, tokenize=False):
        super(NgramLanguageModel, self).__init__()

        if tokenize:
            tokenized_samples = []
            for sample in samples:
                tokenized_samples.append(self.tokenize_string(sample))
            samples = tokenized_samples

        self.n = n
        self.samples = samples
        self.ngram_counts = collections.defaultdict(int)
        self.total_ngrams = 0
        for ngram in self.ngrams():
            self.ngram_counts[ngram] += 1
            self.total_ngrams += 1

    def tokenize_string(self, sample):
        return tuple(sample.lower().split(' '))

    def ngrams(self):
        n = self.n
        for sample in self.samples:
            for i in range(len(sample) - n + 1):
                yield sample[i:i + n]

    def unique_ngrams(self):
        return set(self.ngram_counts.keys())

    def log_likelihood(self, ngram):
        if ngram not in self.ngram_counts:
            return -np.inf
        else:
            return np.log(self.ngram_counts[ngram]) - np.log(self.total_ngrams)

    def kl_to(self, p):
        log_likelihood_ratios = []
        for ngram in p.ngrams():
            log_likelihood_ratios.append(p.log_likelihood(ngram) - self.log_likelihood(ngram))
        return np.mean(log_likelihood_ratios)

    def cosine_sim_with(self, p):
        p_dot_q = 0.
        p_norm = 0.
        q_norm = 0.
        for ngram in p.unique_ngrams():
            p_i = np.exp(p.log_likelihood(ngram))
            q_i = np.exp(self.log_likelihood(ngram))
            p_dot_q += p_i * q_i
            p_norm += p_i**2
        for ngram in self.unique_ngrams():
            q_i = np.exp(self.log_likelihood(ngram))
            q_norm += q_i**2
        return p_dot_q / (np.sqrt(p_norm) * np.sqrt(q_norm))

    def precision_wrt(self, p):
        num = 0.
        denom = 0
        p_ngrams = p.unique_ngrams()
        for ngram in self.unique_ngrams():
            if ngram in p_ngrams:
                num += self.ngram_counts[ngram]
            denom += self.ngram_counts[ngram]
        return float(num) / denom

    def recall_wrt(self, p):
        return p.precision_wrt(self)

    def js_with(self, p):
        log_p = np.array([p.log_likelihood(ngram) for ngram in p.unique_ngrams()])
        log_q = np.array([self.log_likelihood(ngram) for ngram in p.unique_ngrams()])
        log_m = np.logaddexp(log_p - np.log(2), log_q - np.log(2))
        kl_p_m = np.sum(np.exp(log_p) * (log_p - log_m))

        log_p = np.array([p.log_likelihood(ngram) for ngram in self.unique_ngrams()])
        log_q = np.array([self.log_likelihood(ngram) for ngram in self.unique_ngrams()])
        log_m = np.logaddexp(log_p - np.log(2), log_q - np.log(2))
        kl_q_m = np.sum(np.exp(log_q) * (log_q - log_m))

        return 0.5 * (kl_p_m + kl_q_m) / np.log(2)

def load_dataset(max_length, max_n_examples, tokenize=False, max_vocab_size=2048, data_dir='/home/ishaan/data/1-billion-word-language-modeling-benchmark-r13output'):
    print("loading dataset...")

    lines = []

    finished = False

    path = os.path.join(data_dir, "promoter_1.txt")
    with open(path, 'r') as f:
        for line in f:
            line = line[:-1]
            if tokenize:
                line = NgramLanguageModel.tokenize_string(line)
            else:
                line = tuple(line)

            if len(line) > max_length:
                line = line[:max_length]

            lines.append(line + (("`",) * (max_length - len(line))))

            if len(lines) == max_n_examples:
                finished = True
                break

    np.random.shuffle(lines)

    counts = collections.Counter(char for line in lines for char in line)

    charmap = {'unk': 0}
    inv_charmap = ['unk']

    for char, count in counts.most_common(max_vocab_size - 1):
        if char not in charmap:
            charmap[char] = len(inv_charmap)
            inv_charmap.append(char)

    filtered_lines = []
    for line in lines:
        filtered_line = []
        for char in line:
            if char in charmap:
                filtered_line.append(char)
            else:
                filtered_line.append('unk')
        filtered_lines.append(tuple(filtered_line))

    for i in range(100):
        print(filtered_lines[i])
    print("loaded {} lines in dataset".format(len(lines)))
    return filtered_lines, charmap, inv_charmap

# Define the custom dataset class
class MyDataset(Dataset):
    def __init__(self, lines, charmap):
        self.lines = lines
        self.charmap = charmap

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        line = self.lines[idx]
        line_indices = [self.charmap[c] for c in line]
        return torch.tensor(line_indices, dtype=torch.int64)


# Define the generator and discriminator models
class Generator(nn.Module):
    def __init__(self, n_samples, seq_len, dim, charmap_size):
        super(Generator, self).__init__()
        self.dim = dim
        self.seq_len = seq_len
        self.fc = nn.Linear(n_samples, seq_len * dim)
        self.conv1 = nn.Conv1d(dim, dim, 5, padding=2)
        self.conv2 = nn.Conv1d(dim, dim, 5, padding=2)
        self.conv3 = nn.Conv1d(dim, dim, 5, padding=2)
        self.conv4 = nn.Conv1d(dim, dim, 5, padding=2)
        self.conv5 = nn.Conv1d(dim, charmap_size, 1)

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, self.dim, self.seq_len)
        x = F.relu(x)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.conv5(x)
        x = x.permute(0, 2, 1)
        return F.softmax(x, dim=2)


class Discriminator(nn.Module):
    def __init__(self, seq_len, dim, charmap_size):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv1d(charmap_size, dim, 1)
        self.conv2 = nn.Conv1d(dim, dim, 5, padding=2)
        self.conv3 = nn.Conv1d(dim, dim, 5, padding=2)
        self.conv4 = nn.Conv1d(dim, dim, 5, padding=2)
        self.conv5 = nn.Conv1d(dim, dim, 5, padding=2)
        self.fc = nn.Linear(seq_len * dim, 1)
        self.seq_len = seq_len
        self.dim = dim

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.conv5(x)
        x = x.view(-1, self.seq_len * self.dim)
        x = self.fc(x)
        return x


def generate_samples(generator, num_samples=100):
    noise = torch.randn(num_samples, 128).to(device)
    generated_samples = generator(noise)
    _, sampled_indices = torch.max(generated_samples, dim=2)

    decoded_samples = []
    for i in range(len(sampled_indices)):
        decoded = []
        for j in range(len(sampled_indices[i])):
            decoded.append(inv_charmap[sampled_indices[i][j].item()])
        decoded_samples.append(tuple(decoded))

    return decoded_samples

def inf_train_gen(lines, batch_size, seq_len):
    while True:
        np.random.shuffle(lines)
        for i in range(0, len(lines) - batch_size + 1, batch_size):
            yield np.array(
                [[charmap[c] for c in l] for l in lines[i:i + batch_size]],
                dtype='int32'
            )


# Hyperparameters
BATCH_SIZE = 128
ITERS = 200000
SEQ_LEN = 52
DIM = 512
CRITIC_ITERS = 5
LAMBDA = 10
MAX_N_EXAMPLES = 14000
cur_path = os.getcwd()
DATA_DIR = os.path.join(cur_path, 'seq')

# Load dataset
lines, charmap, inv_charmap = load_dataset(
    max_length=SEQ_LEN,
    max_n_examples=MAX_N_EXAMPLES,
    data_dir=DATA_DIR
)

# Define PyTorch models
generator = Generator(n_samples=128, seq_len=SEQ_LEN, dim=DIM, charmap_size=len(charmap)).to(device)
discriminator = Discriminator(seq_len=SEQ_LEN, dim=DIM, charmap_size=len(charmap)).to(device)

# Define optimizer and loss function
optimizer_g = optim.Adam(generator.parameters(), lr=1e-4, betas=(0.5, 0.9))
optimizer_d = optim.Adam(discriminator.parameters(), lr=1e-4, betas=(0.5, 0.9))

true_char_ngram_lms = [NgramLanguageModel(i + 1, lines[10 * BATCH_SIZE:], tokenize=False) for i in range(4)]
validation_char_ngram_lms = [NgramLanguageModel(i + 1, lines[:10 * BATCH_SIZE], tokenize=False) for i in range(4)]
for i in range(4):
    print("validation set JSD for n={}: {}".format(i + 1, true_char_ngram_lms[i].js_with(validation_char_ngram_lms[i])))
true_char_ngram_lms = [NgramLanguageModel(i + 1, lines, tokenize=False) for i in range(4)]

gen = inf_train_gen(lines, BATCH_SIZE, SEQ_LEN)
train_num = 0
epoch_num = 0

# Training loop
gen_loss = []
dis_loss = []
for iteration in range(ITERS):
    start_time = time.time()

    if iteration > 0:
        # Train generator
        optimizer_g.zero_grad()
        fake_inputs = generator(torch.randn(BATCH_SIZE, 128).to(device))
        disc_fake = discriminator(fake_inputs)
        gen_cost = -torch.mean(disc_fake)
        gen_cost.backward()
        optimizer_g.step()

    for i in range(CRITIC_ITERS):
        # Train discriminator
        optimizer_d.zero_grad()
        real_inputs = torch.LongTensor(next(gen)).to(device)
        train_num += BATCH_SIZE
        real_inputs_discrete = F.one_hot(real_inputs, num_classes=len(charmap)).float()
        disc_real = discriminator(real_inputs_discrete)
        fake_inputs = generator(torch.randn(BATCH_SIZE, 128).to(device)).detach()
        disc_fake = discriminator(fake_inputs)
        alpha = torch.rand(BATCH_SIZE, 1, 1).to(device)
        interpolates = real_inputs_discrete + (alpha * (fake_inputs - real_inputs_discrete))
        interpolates.requires_grad_(True)
        gradients = torch.autograd.grad(outputs=discriminator(interpolates),
                                        inputs=interpolates,
                                        grad_outputs=torch.ones(disc_real.size()).to(device),
                                        create_graph=True,
                                        retain_graph=True,
                                        only_inputs=True)[0]
        slopes = torch.sqrt(torch.sum(gradients ** 2, dim=[1, 2]))
        gradient_penalty = torch.mean((slopes - 1.) ** 2)
        disc_cost = torch.mean(disc_fake) - torch.mean(disc_real)
        disc_cost += LAMBDA * gradient_penalty
        disc_cost.backward()
        optimizer_d.step()

    # Print and save results
    if iteration % 1000 == 999:
        print('Time: {:.2f} s, Iteration: {}, Gen Cost: {:.4f}, Disc Cost: {:.4f}'.format(time.time() - start_time,
                                                                                          iteration, gen_cost.item(),
                                                                                          disc_cost.item()))

    # if iteration % 10000 == 9999:
    if train_num >= MAX_N_EXAMPLES:
        gen_loss.append(gen_cost.item())
        dis_loss.append(disc_cost.item())
        train_num = 0
        epoch_num += 1
        if epoch_num > 150:

            with open("gen_loss.json", 'w') as file:
                json.dump(gen_loss, file)
            with open("dis_loss.json", 'w') as file:
                json.dump(dis_loss, file)
            break
        if epoch_num % 25 == 24:
        # Save models
            torch.save(generator.state_dict(), f'./model/generator_model_{epoch_num}.pth')
            torch.save(discriminator.state_dict(), f'./model/discriminator_model_{epoch_num}.pth')

            # Generate samples
            samples = []
            for i in range(10):
                samples.extend(generate_samples(generator))
            print("epoch{}:".format(epoch_num))
            # Evaluate JS divergence
            for i in range(4):
                lm = NgramLanguageModel(i + 1, samples, tokenize=False)
                print('JS{}: {:.4f}'.format(i + 1, lm.js_with(true_char_ngram_lms[i])))

            # Save generated samples
            with open(f'res/seq_epoch_{epoch_num}.txt', 'w') as f:
                for s in samples:
                    s = "".join(s)
                    f.write(s + "\n")

# Continue with additional iterations or analysis as needed
