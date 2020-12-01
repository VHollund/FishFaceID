import torchvision.datasets as dset

import torchvision.transforms as transforms
from torchvision.transforms import transforms
from torch.autograd import Variable
import torchvision

import torch
from SiameseNetwork import main
import torch.nn.functional as F

def checkStats(dataiter):
    FN = 0
    TN = 0
    FP = 0
    TP = 0
    TH = 0.8
    equalSum = 0
    unEqualSum = 0
    equal, unequal=0,0
    for i in range(100):
        x0, x1, label = next(dataiter)
        is_equall = not label[0][0].item()
        output1, output2 = model(Variable(x0), Variable(x1))
        euclidean_distance = F.pairwise_distance(output1, output2)

        if euclidean_distance==0:
            continue
        print(is_equall)
        if euclidean_distance < TH and is_equall:
            TP += 1
        elif euclidean_distance >= TH and not is_equall:
            TN += 1
        elif euclidean_distance >= TH and is_equall:
            FN += 1
        elif euclidean_distance <= TH and not is_equall:
            FP += 1
        if is_equall:
            equal+=1
            equalSum += euclidean_distance
        else:
            unequal+=1
            unEqualSum += euclidean_distance
    print(f"correct avg / incorrect avg:\n\t\t{equalSum/equal} / {unEqualSum/unequal}")
    print("Accuracy: ", (TP + TN) / (TP + TN + FP + FN))
    print("Recall: ", TP / (TP + FN))
    print("Precision: ", TP / (TP + FP))
    print("F1: ", (2 * TP) / (2 * TP + FP + FN))

    return [equalSum/equal, unEqualSum/unequal, (TP + TN) / (TP + TN + FP + FN),TP / (TP + FN),TP / (TP + FP),(2 * TP) / (2 * TP + FP + FN)]

model = main.SiameseNetwork()
model.load_state_dict(torch.load("model.pt"))
model.eval()

folder_dataset = dset.ImageFolder(root=main.Config.training_dir)

siamese_dataset = main.SiameseNetworkDataset(
    imageFolderDataset=folder_dataset,
    transform=main.transformation(),
    should_invert=False)

test_dataloader = main.DataLoader(siamese_dataset,num_workers=6,batch_size=1,shuffle=True)
dataiter = iter(test_dataloader)
#x0,_,label1 = next(dataiter)

for i in range(10):
    x0,x1,label = next(dataiter)
    concatenated = torch.cat((x0,x1),0)
    print("unequal" if label[0][0].item() else "equal")
    
    output1,output2 = model(Variable(x0),Variable(x1))
    euclidean_distance = F.pairwise_distance(output1, output2)
    main.imshow(torchvision.utils.make_grid(concatenated),'Dissimilarity: {:.2f}'.format(euclidean_distance.item()))
