import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

cur_path = os.getcwd()
DATA_DIR = os.path.join(cur_path, 'seq')

BATCH_SIZE = 32
ITERS = 200000
SEQ_LEN = 50
DIM = 512
CRITIC_ITERS = 5
LAMBDA = 10
MAX_N_EXAMPLES = 14098

lines, charmap, inv_charmap = language_helpers.load_dataset(
    max_length=SEQ_LEN,
    max_n_examples=MAX_N_EXAMPLES,
    data_dir=DATA_DIR
)

def make_noise(shape):
    return torch.randn(shape)

def ResBlock(inputs):
    output = nn.ReLU()(inputs)
    output = nn.Conv1d(DIM, DIM, 5, padding=2)(output)
    output = nn.ReLU()(output)
    output = nn.Conv1d(DIM, DIM, 5, padding=2)(output)
    return inputs + (0.3 * output)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.linear = nn.Linear(128, SEQ_LEN * DIM)
        self.conv1 = nn.Conv1d(DIM, DIM, 1)
        self.res_blocks = nn.ModuleList([ResBlock() for _ in range(5)])
        self.conv_out = nn.Conv1d(DIM, len(charmap), 1)

    def forward(self, n_samples):
        output = make_noise((n_samples, 128))
        output = self.linear(output)
        output = output.view(-1, DIM, SEQ_LEN)
        for res_block in self.res_blocks:
            output = res_block(output)
        output = self.conv_out(output)
        output = output.transpose(1, 2)
        output = nn.functional.softmax(output, dim=2)
        return output

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv1d(len(charmap), DIM, 1)
        self.res_blocks = nn.ModuleList([ResBlock() for _ in range(5)])
        self.linear = nn.Linear(SEQ_LEN * DIM, 1)

    def forward(self, inputs):
        output = inputs.transpose(1, 2)
        output = self.conv1(output)
        for res_block in self.res_blocks:
            output = res_block(output)
        output = output.view(-1, SEQ_LEN * DIM)
        output = self.linear(output)
        return output

generator = Generator()
discriminator = Discriminator()

real_inputs_discrete = torch.placeholder(torch.int32, shape=[BATCH_SIZE, SEQ_LEN])
real_inputs = nn.functional.one_hot(real_inputs_discrete, len(charmap))
fake_inputs = generator(BATCH_SIZE)
fake_inputs_discrete = torch.argmax(fake_inputs, dim=2)

disc_real = discriminator(real_inputs)
disc_fake = discriminator(fake_inputs)

disc_cost = torch.mean(disc_fake) - torch.mean(disc_real)
gen_cost = -torch.mean(disc_fake)

alpha = torch.rand(BATCH_SIZE, 1, 1)
differences = fake_inputs - real_inputs
interpolates = real_inputs + (alpha * differences)
interpolates.requires_grad_()
gradients = torch.autograd.grad(discriminator(interpolates), interpolates,
                                grad_outputs=torch.ones_like(discriminator(interpolates)),
                                create_graph=True, retain_graph=True)[0]
slopes = torch.sqrt(torch.sum(gradients ** 2, dim=[1, 2]))
gradient_penalty = torch.mean((slopes - 1.) ** 2)
disc_cost += LAMBDA * gradient_penalty

gen_optimizer = optim.Adam(generator.parameters(), lr=1e-4, betas=(0.5, 0.9))
disc_optimizer = optim.Adam(discriminator.parameters(), lr=1e-4, betas=(0.5, 0.9))

def inf_train_gen():
    while True:
        np.random.shuffle(lines)
        for i in range(0, len(lines) - BATCH_SIZE + 1, BATCH_SIZE):
            yield np.array(
                [[charmap[c] for c in l] for l in lines[i:i + BATCH_SIZE]],
                dtype='int32'
            )

gen = inf_train_gen()

for iteration in range(ITERS):
    start_time = time.time()

    if iteration > 0:
        gen_optimizer.zero_grad()
        gen_cost.backward()
        gen_optimizer.step()

    for i in range(CRITIC_ITERS):
        _data = torch.from_numpy(next(gen))
        disc_optimizer.zero_grad()
        real_inputs_discrete = _data
        real_inputs = nn.functional.one_hot


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class Generator(nn.Module):
    def __init__(self, input_size, output_size):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, output_size),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, input_size):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# Set random seed for reproducibility
torch.manual_seed(42)

# Define the input and output dimensions
input_size = 100
output_size = 784

# Create the generator and discriminator models
generator = Generator(input_size, output_size)
discriminator = Discriminator(output_size)

# Define the loss function and optimizer for each model
criterion = nn.BCELoss()
gen_optimizer = optim.Adam(generator.parameters(), lr=0.0002)
disc_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002)

# Training loop
num_epochs = 100
batch_size = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for epoch in range(num_epochs):
    for batch_idx in range(len(data_loader)):
        # Train the discriminator
        for _ in range(disc_iterations):
            discriminator.zero_grad()

            real_images = batch.to(device)
            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1).to(device)

            # Generate fake images
            noise = torch.randn(batch_size, input_size).to(device)
            fake_images = generator(noise)

            # Train the discriminator on real and fake images
            real_outputs = discriminator(real_images.view(batch_size, -1))
            fake_outputs = discriminator(fake_images.detach())
            disc_loss = criterion(real_outputs, real_labels) + criterion(fake_outputs, fake_labels)
            disc_loss.backward()
            disc_optimizer.step()

        # Train the generator
        generator.zero_grad()

        # Generate fake images
        noise = torch.randn(batch_size, input_size).to(device)
        fake_images = generator(noise)

        # Train the generator to fool the discriminator
        fake_outputs = discriminator(fake_images)
        gen_loss = criterion(fake_outputs, real_labels)
        gen_loss.backward()
        gen_optimizer.step()

    # Print the loss after each epoch
    print(f"Epoch [{epoch + 1}/{num_epochs}], Generator Loss: {gen_loss.item():.4f}, Discriminator Loss: {disc_loss.item():.4f}")