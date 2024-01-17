class PREDICT(nn.Module):

    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(165, 300, kernel_size=(6, 1), padding=(3, 0)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1))
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(300, 600, kernel_size=(5, 1), padding=(3, 0)),
            nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(600, 600, kernel_size=(6, 1), padding=(3, 0)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)),
            nn.Flatten()
        )
        self.fc = nn.Sequential(
            nn.Linear(1200, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.fc(x)
        x = x.squeeze(-1)
        # x = x.squeeze(-1)
        return x
