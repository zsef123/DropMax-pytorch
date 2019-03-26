import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms

from network import CNN, DropMaxCNN
from DropMax import DropMax

if __name__ == "__main__":
    n_classes = 10
    device = torch.device("cuda")

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    trains = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True, transform=transform),
        batch_size=32, shuffle=True, num_workers=4
    )

    tests = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, transform=transform),
        batch_size=32, shuffle=False, num_workers=4
    )

    print("Train : ", len(trains.dataset))
    print("Test : ", len(tests.dataset))

    dropmax_cnn = DropMaxCNN(n_classes=n_classes).to(device)
    dropmax_loss = DropMax(device)
    adam = optim.Adam(dropmax_cnn.parameters(), lr=0.0005, betas=(0.5, 0.999), weight_decay=1e-4)

    cnn = CNN(n_classes=n_classes).to(device)
    ce_loss = nn.CrossEntropyLoss()
    adam2 = optim.Adam(cnn.parameters(), lr=0.0005, betas=(0.5, 0.999), weight_decay=1e-4)

    for epoch in range(30):
        dropmax_cnn.train()
        print("Epoch : ", epoch)
        for i, (img, target) in enumerate(trains):
            img = img.to(device)
            one_hot = torch.zeros(img.shape[0], n_classes)
            one_hot.scatter_(1, target.unsqueeze(dim=1), 1)
            one_hot = one_hot.to(device)

            o, p, r, q = dropmax_cnn(img)
            loss = dropmax_loss(o, p, r, q, one_hot)

            adam.zero_grad()
            loss.backward()
            adam.step()

            o2 = cnn(img)
            loss2 = ce_loss(o2, target.to(device))

            adam2.zero_grad()
            loss2.backward()
            adam2.step()

            if i % 30000 == 0:
                print("DM loss  : ", loss.item())
                print("CE loss : ", loss2.item())
                break


        correct, correct2 = 0, 0
        for i, (img, target) in enumerate(tests):
            img = img.to(device)
            one_hot = torch.zeros(img.shape[0], n_classes)
            one_hot.scatter_(1, target.unsqueeze(dim=1), 1)
            one_hot = one_hot.to(device)
            o, p, r, q = dropmax_cnn(img)
            correct += dropmax_loss.get_acc(p, o, target)

            o2 = cnn(img)
            _, idx = o2.max(dim=1)
            correct2 += (idx.cpu() == target).sum().item()

        print("DM acc : ", correct  / len(tests.dataset))
        print("CE acc : ", correct2 / len(tests.dataset))
        print("--------")

