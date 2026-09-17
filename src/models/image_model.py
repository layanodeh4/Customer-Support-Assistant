from pathlib import Path

import torch

from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


DATA_DIR = Path("data/fashion")

MODEL_PATH = Path(
    "artifacts/fashion_mnist_cnn.pt"
)


class FashionCNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(

            nn.Conv2d(
                1,
                16,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(
                16,
                32,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Flatten(),

            nn.Linear(
                32 * 7 * 7,
                10
            ),
        )

    def forward(self, x):

        return self.network(x)


def main():

    transform = transforms.ToTensor()

    train_dataset = datasets.FashionMNIST(
        DATA_DIR,
        train=True,
        download=True,
        transform=transform,
    )

    test_dataset = datasets.FashionMNIST(
        DATA_DIR,
        train=False,
        download=True,
        transform=transform,
    )

    # For a student project, use a subset.
    train_dataset = torch.utils.data.Subset(
        train_dataset,
        range(10000)
    )

    test_dataset = torch.utils.data.Subset(
        test_dataset,
        range(2000)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
    )

    model = FashionCNN()

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
    )

    # -----------------------------
    # Training
    # -----------------------------

    model.train()

    for epoch in range(2):

        total_loss = 0

        for images, labels in train_loader:

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        print(
            f"Epoch {epoch + 1} "
            f"Loss: {total_loss / len(train_loader):.4f}"
        )

    # -----------------------------
    # Evaluation
    # -----------------------------

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in test_loader:

            outputs = model(images)

            predictions = outputs.argmax(
                dim=1
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

    accuracy = correct / total

    print(
        f"Image model accuracy: {accuracy:.3f}"
    )

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    torch.save(
        model.state_dict(),
        MODEL_PATH
    )

    print(
        f"Model saved to {MODEL_PATH}"
    )


if __name__ == "__main__":
    main()