import torch
import torch.nn as nn
import torch.nn.functional as F


class ABMIL(nn.Module):
    def __init__(self, num_classes=4):
        super(ABMIL, self).__init__()
        self.num_classes = num_classes
        self.M = 500
        self.L = 128

        self.feature_extractor_part1 = nn.Sequential(
            nn.Conv2d(3, 20, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(20, 50, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )

        self.feature_extractor_part2 = nn.Sequential(
            nn.Linear(50 * 5 * 5, self.M),
            nn.ReLU()
        )

        self.attention = nn.Sequential(
            nn.Linear(self.M, self.L),
            nn.Tanh(),
            nn.Linear(self.L, 1)
        )

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(self.M, self.num_classes)
        )

    def forward(self, X):
        B, N, C, H, W = X.shape
        X = X.contiguous().view(B * N, C, H, W)  # (B x N) x C x H x W

        H = self.feature_extractor_part1(X)
        H = H.contiguous().view(B * N, -1)
        H = self.feature_extractor_part2(H)  # (B x N) x M

        A = self.attention(H)  # (B x N) x 1
        A = A.contiguous().view(B, 1, N)  # B x 1 x N
        A = F.softmax(A, dim=2)  # softmax over N

        H = H.contiguous().view(B, N, self.M)  # B x N x M
        Z = torch.bmm(A, H)  # B x 1 x M
        Z = Z.squeeze(1)  # B x M

        output = self.classifier(Z)  # B x K
        output_dict = {
            'y_hat': torch.argmax(output, dim=1),  # B
            'attention': A.squeeze(1),  # B x N
            'feature': Z  # B x M
        }
        return output, output_dict


class AddMIL(nn.Module):
    def __init__(self, num_classes=4, activation='sigmoid'):
        super(AddMIL, self).__init__()
        self.num_classes = num_classes
        self.M = 500
        self.L = 128
        if activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'tanh':
            self.activation = nn.Tanh()

        self.feature_extractor_part1 = nn.Sequential(
            nn.Conv2d(3, 20, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(20, 50, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )

        self.feature_extractor_part2 = nn.Sequential(
            nn.Linear(50 * 5 * 5, self.M),
            nn.ReLU()
        )

        self.attention = nn.Sequential(
            nn.Linear(self.M, self.L),
            nn.Tanh(),
            nn.Linear(self.L, 1)
        )

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(self.M, self.num_classes),
            self.activation
        )

    def forward(self, X):
        B, N, C, H, W = X.shape
        X = X.contiguous().view(B * N, C, H, W)  # (B x N) x C x H x W

        H = self.feature_extractor_part1(X)
        H = H.contiguous().view(B * N, -1)
        H = self.feature_extractor_part2(H)  # (B x N) x M

        A = self.attention(H)  # (B x N) x 1
        A = A.contiguous().view(B, N, 1)  # B x N x 1
        A = F.softmax(A, dim=1)  # softmax over N

        H = H.contiguous().view(B, N, self.M)  # B x N x M
        Z = torch.mul(A, H)  # B x N x M
        Z = Z.contiguous().view(B * N, self.M)  # (B x N) x M

        P = self.classifier(Z)  # (B x N) x K
        P = P.contiguous().view(B, N, self.num_classes)  # B x N x K

        output = torch.sum(P, dim=1)  # B x K
        output_dict = {
            'y_hat': torch.argmax(output, dim=1),  # B
            'attention': A.squeeze(2),  # B x N
            'contribution': P,  # B x N x K
            'feature': Z.contiguous().view(B, N, self.M),  # B x N x M
        }
        return output, output_dict


if __name__ == '__main__':
    torch.manual_seed(0)
    batch_size = 1
    bag_size = 64

    X = torch.rand(batch_size, bag_size, 3, 32, 32)
    print(f'X\t: {X.shape}')

    for model in [ABMIL, AddMIL]:
        model = model()
        print(f'\n{model.__class__.__name__}')
        output, output_dict = model(X)
        print(f'output\t: {output.shape}')
        for k, v in output_dict.items():
            print(f'{k}\t: {v.shape}')