import json
import torch
from dataloader_main import dataFetcher
from torch.utils.data import DataLoader
from model_network import GeneratorNet, DiscriminatorNet
from train1 import train
from simulator import simulatorNet
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

# loading the configurations from the JSON file
json_path = json.load(open("config.json", "r"))
#print (json_path)

#loading parameters from the JSON file
noiseDim = json_path["noiseDim"]
labelDim = json_path["labelDim"]
batchSize = json_path["batchSize"]
genLr= json_path["genLr"]
disLr = json_path["disLr"]
genBeta1 = json_path["genBeta1"]
genBeta2 = json_path["genBeta2"]
disBeta1 = json_path["disBeta1"]
disBeta2 = json_path["disBeta2"]
#print(noiseDim, labelDim, epochNo, batchSize, genLr)

#running the computaion in GPU if not then CPU
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

#loading dataset
#data = "Dataset/trainingset_1k.nc"
data = "Dataset/trainingset_5k.nc"

#fetching, normalizing and transforming
dataloader = DataLoader(dataFetcher(data),batch_size=batchSize, shuffle=True)

# importing the Generator and Discriminator Network
generator = GeneratorNet(noiseDim, labelDim).cuda()
discriminator = DiscriminatorNet(labelDim).cuda()

# Optimzers
genOptimizer = torch.optim.Adam(generator.parameters(), lr=genLr, betas=(genBeta1, genBeta2))
disOptimizer = torch.optim.Adam(discriminator.parameters(), lr=disLr,betas=(disBeta1, disBeta2))

#loading simulator
# simulator = simulatorNet().cuda()
# simulator.eval()

#training model and saving the initial and final generator model
# torch.save(generator.state_dict(), 'genModel/init_gen.pt')
dis_loss, gen_loss = train(generator, discriminator,genOptimizer, disOptimizer,dataloader, json_path)
# torch.save(generator.state_dict(), 'genModel/final_gen1.pt')

a = np.array(dis_loss)
np.savetxt('Dis_mi.txt', a, fmt='%.15f')
a = np.array(gen_loss)
np.savetxt('gen_mi.txt', a, fmt='%.15f')

img = mpimg.imread('images/gen/iter'+str(json_path["epochGAN"]-1)+'.png')
imgplot = plt.imshow(img)
plt.show()
