import torchvision.datasets as dset

import torchvision.transforms as transforms
from torchvision.transforms import transforms
from torch.autograd import Variable
import torchvision

import torch
from SiameseNetwork import main
import torch.nn.functional as F

# Load model from file and set net to eval mode
model = main.SiameseNetwork()
model.load_state_dict(torch.load("modelHR.pt"))
model.eval()

folder_dataset = dset.ImageFolder(root=main.Config.testing_dir)

siamese_dataset = main.SiameseNetworkDataset(
    imageFolderDataset=folder_dataset,
    transform=main.transformation(),
    should_invert=False)

test_dataloader = main.DataLoader(siamese_dataset,num_workers=6,batch_size=1,shuffle=True)
dataiter = iter(test_dataloader)

for i in range(10):
    x0,x1,label = next(dataiter)
    concatenated = torch.cat((x0,x1),0)
    print("unequal" if label[0][0].item() else "equal")
    
    output1,output2 = model(Variable(x0),Variable(x1))
    euclidean_distance = F.pairwise_distance(output1, output2)
    main.imshow(torchvision.utils.make_grid(concatenated),'Dissimilarity: {:.2f}'.format(euclidean_distance.item()))
