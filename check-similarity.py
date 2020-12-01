import sys
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
from PIL import Image
from SiameseNetwork import main

if len(sys.argv) != 3:
    print("Incorrect number of arguments, expected two image files")
    sys.exit()

file1 = sys.argv[1]
file2 = sys.argv[2]

# Open files passed in as parameters
img1 = Image.open(file1).convert("L")
img2 = Image.open(file2).convert("L")

transformation = main.transformation()

img1 = transformation(img1)
img2 = transformation(img2)

# Needs this to get right tensor dimensions for net
img1 = torch.unsqueeze(img1, 0)
img2 = torch.unsqueeze(img2, 0)

# Load model from file
model = main.SiameseNetwork()
model.load_state_dict(torch.load("modelHR.pt"))
model.eval()

output1, output2 = model(Variable(img1), Variable(img2))
euclidean_distance = F.pairwise_distance(output1, output2)

concatenated = torch.cat((img1, img2), 0)
main.imshow(torchvision.utils.make_grid(concatenated), 'Dissimilarity: {:.2f}'.format(euclidean_distance.item()))
