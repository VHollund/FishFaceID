import torchvision.datasets as dset

import torchvision.transforms as transforms
from torchvision.transforms import transforms
from torch.autograd import Variable
import torchvision

import torch
from SiameseNetwork import main
import torch.nn.functional as F


def checkStats(dataiter):
    FN, TN, FP, TP = 0, 0, 0, 0
    TH = 1.5724
    equalSum, unEqualSum, equal, unequal = 0, 0, 0, 0
    equalMaxMin, NotequalMaxMin=[], []
    results=[]
    for i in range(100):
        x0, x1, label = next(dataiter)
        is_equall = not label[0][0].item()
        output1, output2 = model(Variable(x0), Variable(x1))
        euclidean_distance = F.pairwise_distance(output1, output2)
        results.append((euclidean_distance,is_equall))
        if is_equall:
            equalMaxMin.append(euclidean_distance)
            equal += 1
            equalSum += euclidean_distance
        else:
            NotequalMaxMin.append(euclidean_distance)
            unequal += 1
            unEqualSum += euclidean_distance
    equalMaxMin = [min(equalMaxMin), max(equalMaxMin)]
    NotequalMaxMin = [min(NotequalMaxMin), max(NotequalMaxMin)]
    TH = (equalMaxMin[1] + NotequalMaxMin[0]) / 2
    for x in results:
        euclidean_distance, is_equall=x
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
            equalMaxMin.append(euclidean_distance)
            equal += 1
            equalSum += euclidean_distance
        else:
            NotequalMaxMin.append(euclidean_distance)
            unequal += 1
            unEqualSum += euclidean_distance
    equalMaxMin=[min(equalMaxMin), max(equalMaxMin)]
    NotequalMaxMin=[min(NotequalMaxMin), max(NotequalMaxMin)]
    TH=equalMaxMin[1]+NotequalMaxMin[0]/2
    print(f"correct avg / incorrect avg:\n\t\t{equalSum/equal} / {unEqualSum/unequal}")
    print("Threshold: ", (equalMaxMin[1]+NotequalMaxMin[0])/2)
    print("Accuracy: ", (TP + TN) / (TP + TN + FP + FN))
    print("Recall: ", TP / (TP + FN))
    print("Precision: ", TP / (TP + FP))
    print("F1: ", (2 * TP) / (2 * TP + FP + FN))
    print(f"equal Min:{min(equalMaxMin)}\nEqual Max:{max(equalMaxMin)}\nNot equal min{min(NotequalMaxMin)}\nNot equal max{max(NotequalMaxMin)}")
    return [equalSum/equal, unEqualSum/unequal, equalMaxMin, NotequalMaxMin, (TP + TN) / (TP + TN + FP + FN),TP / (TP + FN),TP / (TP + FP),(2 * TP) / (2 * TP + FP + FN),]



model = main.SiameseNetwork()
model.load_state_dict(torch.load("BodyR100.pt"))
model.eval()

folder_dataset = dset.ImageFolder(root=main.Config.testing_dir)

siamese_dataset = main.SiameseNetworkDataset(
    imageFolderDataset=folder_dataset,
    transform=main.transformation(),
    should_invert=False)

test_dataloader = main.DataLoader(siamese_dataset,num_workers=6,batch_size=1,shuffle=True)
dataiter = iter(test_dataloader)
#x0,_,label1 = next(dataiter)
checkStats(dataiter)

"""
for i in range(10):
    x0,x1,label = next(dataiter)
    concatenated = torch.cat((x0,x1),0)
    print("unequal" if label[0][0].item() else "equal")
    
    output1,output2 = model(Variable(x0),Variable(x1))
    euclidean_distance = F.pairwise_distance(output1, output2)
    main.imshow(torchvision.utils.make_grid(concatenated),'Dissimilarity: {:.2f}'.format(euclidean_distance.item()))
"""