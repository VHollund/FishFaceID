import torchvision.datasets as dset

import torchvision.transforms as transforms
from torchvision.transforms import transforms

import torch
from SiameseNetwork.main import (
    Config, SiameseNetwork, SiameseNetworkDataset, DataLoader,
    ContrastiveLoss
)


# Loads images from data folder
folder_dataset = dset.ImageFolder(root=Config.training_dir)

# Transformation to apply to images
# Resize to config image size
# Convert PIL image to tensor
transformation = transforms.Compose([
        transforms.Resize((Config.image_size, Config.image_size)),
        transforms.ToTensor()
    ])

siamese_dataset = SiameseNetworkDataset(
    imageFolderDataset=folder_dataset,
    transform=transformation,
    should_invert=False)


train_dataloader = DataLoader(siamese_dataset,
                        shuffle=True,
                        num_workers=8,
                        batch_size=Config.train_batch_size)

net = SiameseNetwork()
criterion = ContrastiveLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.0005)

counter = []
loss_history = []
iteration_number = 0

for epoch in range(Config.train_number_epochs):
    for i, data in enumerate(train_dataloader, 0):
        img0, img1, label = data
        optimizer.zero_grad()
        output1, output2 = net(img0, img1)
        loss_contrastive = criterion(output1, output2, label)
        loss_contrastive.backward()
        optimizer.step()
        if i %10 == 0 :
            print("Epoch number {}\n Current loss {}\n".format(epoch,loss_contrastive.item()))
            iteration_number +=10
            counter.append(iteration_number)
            loss_history.append(loss_contrastive.item())

# Save model to file
torch.save(net.state_dict(), "modelHR.pt")
