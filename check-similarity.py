import sys
from SiameseNetwork import main
import torch
from PIL import Image
import torchvision

import torch.nn.functional as F

from torch.autograd import Variable

if len(sys.argv) != 3:
    print("Incorrect number of arguments, expected two image files")
    sys.exit()

file1 = sys.argv[1]
file2 = sys.argv[2]

img1 = Image.open(file1).convert("L")
img2 = Image.open(file2).convert("L")

transformation = main.transformation()

img1 = transformation(img1)
img2 = transformation(img2)

img1 = torch.unsqueeze(img1, 0)
img2 = torch.unsqueeze(img2, 0)

model = main.SiameseNetwork()
model.load_state_dict(torch.load("model.pt"))
model.eval()

output1, output2 = model(Variable(img1), Variable(img2))


euclidean_distance = F.pairwise_distance(output1, output2)

concatenated = torch.cat((img1, img2),0)
main.imshow(torchvision.utils.make_grid(concatenated),'Dissimilarity: {:.2f}'.format(euclidean_distance.item()))
