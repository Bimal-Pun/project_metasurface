from dataloader_main import dataFetcher
from torch.utils.data import DataLoader
import torch
import json
from model_network import GeneratorNet
from simulator import simulatorNet
from torch.autograd import Variable
import itertools
import matplotlib.pyplot as plt
import numpy as np
from statistics import mean
from itertools import chain

from torchvision.utils import save_image
import torch.nn.functional as F

# loading the configurations from the JSON file
json_path = json.load(open("config.json", "r"))

#running the computaion in GPU if not then CPU
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

#loading dataset
data = "Dataset/trainingset_1k.nc"

#fetching, normalizing and transforming
dataloader = DataLoader(dataFetcher(data),batch_size=json_path["batchSize"], shuffle=True)

#loading simulator
simulator = simulatorNet().cuda()
simulator.eval()

#temporary simualtor
simulator_temp = torch.load('simModel/model_s.pt', map_location= torch.device(device))
simulator.load_state_dict(simulator_temp)

# loading the Generator
generator = GeneratorNet(json_path["noiseDim"], json_path["labelDim"]).cuda()
generator.eval()

#temporary generator
generator_temp = torch.load('genModel/final_gen1.pt', map_location= torch.device(device))
generator.load_state_dict(generator_temp)

genTrans = []
realTrans = []

for n_batch, (realImg, labels) in enumerate(dataloader):


    realImg, labels = realImg.to(device, dtype=torch.float), labels.to(device)

    realTrans_one = labels

    # generating noise for the generator
    noise = Variable(torch.randn(labels.size(0), json_path["noiseDim"]).to(device))

    genImage = generator(labels, noise)

    genTrans_one = simulator(genImage)

    genTrans.append(genTrans_one.tolist())
    realTrans.append(realTrans_one.tolist())


genTrans_new = list(itertools.chain(*genTrans))
realTrans_new = list(itertools.chain(*realTrans))

print("gentrans", len(genTrans_new))
print("realtrnas", len(realTrans_new))



a = np.array(genTrans_new)
np.savetxt('transmittance/genTrans_new.txt', a, fmt='%.3f')
a = np.array(realTrans_new)
np.savetxt('transmittance/realTrans_new.txt', a, fmt='%.3f')

data = np.genfromtxt("transmittance/genTrans_new.txt")
data1 = np.genfromtxt("transmittance/realTrans_new.txt")
n = 36

data_gen = list(chain.from_iterable([mean(data[i:i+n])]*n for i in range(0,len(data),n)))
data_real = list(chain.from_iterable([mean(data1[i:i+n])]*n for i in range(0,len(data1),n)))

data_gen1 = [data_gen[x] for x in range(0,828,36)]
data_real1 = [data_real[x] for x in range(0,828,36)]

fig= plt.figure()
plt.scatter(range(len(data)), data, label = 'Generated', marker = 'o')
plt.scatter(range(len(data1)), data1, label = 'Real', marker = '*')
# plt.scatter(range(len(data_gen1)), data_gen1, label = 'Generated', marker = 'o')
# #plt.scatter(range(len(data_real)), data_real, label = 'real')
# plt.scatter(range(len(data_real1)), data_real1, label = 'Real', marker = '*')

fig.suptitle('Real and Generated Transmittance for each image', fontsize = 20)
plt.ylim(0.2, 1)
plt.ylabel('Transmittance')

plt.xlabel('Images')
plt.legend()
fig.savefig('transmittance/ test_new12.jpg', bbox_inches="tight")
plt.show()