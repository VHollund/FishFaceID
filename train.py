import torchvision.datasets as dset

import torchvision.transforms as transforms
from torchvision.transforms import transforms
from torch.autograd import Variable
import torchvision
import torch.nn.functional as F

import torch
from SiameseNetwork.main import Config, SiameseNetwork, SiameseNetworkDataset, DataLoader, ContrastiveLoss, imshow


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
# TODO: Explain this use
criterion = ContrastiveLoss()
# TODO: Figure out Adam optimizer and parameters
optimizer = torch.optim.Adam(net.parameters(),lr = 0.0005 )

counter = []
loss_history = [] 
iteration_number= 0

#TODO: Save model to disk

#for epoch in range(Config.train_number_epochs):
for epoch in range(20):
    for i, data in enumerate(train_dataloader, 0):
        img0, img1, label = data
        # TODO: Figure out what this does
        optimizer.zero_grad()
        # TODO: Figure out exactly what this does
        output1, output2 = net(img0, img1)
        # TODO: Figure out what these do
        loss_contrastive = criterion(output1, output2, label)
        loss_contrastive.backward()
        # TODO: Figure this one out too
        optimizer.step()
        if i %10 == 0 :
            print("Epoch number {}\n Current loss {}\n".format(epoch,loss_contrastive.item()))
            iteration_number +=10
            counter.append(iteration_number)
            loss_history.append(loss_contrastive.item())


torch.save(net.state_dict(), "model.pt")

test_dataloader = DataLoader(siamese_dataset,num_workers=6,batch_size=1,shuffle=True)
dataiter = iter(test_dataloader)
x0,_,_ = next(dataiter)
print(x0.size())
print(siamese_dataset.__getitem__(0)[0].size())

for i in range(10):
    _,x1,label2 = next(dataiter)
    concatenated = torch.cat((x0,x1),0)
    
    output1,output2 = net(Variable(x0),Variable(x1))
    euclidean_distance = F.pairwise_distance(output1, output2)
    imshow(torchvision.utils.make_grid(concatenated),'Dissimilarity: {:.2f}'.format(euclidean_distance.item()))
