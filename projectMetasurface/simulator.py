import torch.nn as nn
import torch
from dataloader_main import dataFetcher
from torch.utils.data import DataLoader
import json
from tqdm import tqdm
import numpy as np

# running the computaion in GPU if not then CPU
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
class simulatorNet(nn.Module):


    def __init__(self):
        super(simulatorNet, self).__init__()
        self.conv2d = nn.Conv2d(
            in_channels=1,
            out_channels=64,
            kernel_size=5, stride=2,
            padding=2,
            bias=True,
            dilation=1,
            groups=1)

        self.convLayer = nn.Sequential(
            self.conv2d,
            nn.LeakyReLU(0.2)
            )

        self.linearLayer = nn.Sequential(
            #first linear layer
            nn.Linear(64*16*64, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            #second layer
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            #third layer
            nn.Linear(512, 1)
            )

    def forward(self, img):
        # image is passed (either real or fake) with the label values = 2.0

        # adding a small amount of noise to the passed image so that the discriminator doesnt overfit
        net = img + (torch.randn(img.size())*0.1).to(device)
        #net = img + (torch.randn(img.size())*0.1).to(device, dtype=torch.float)

        net = self.convLayer(net)
        # changing 2d into 1 array so that it can be passed to teh linear layer in self liner layer
        net = net.view(net.size(0), -1)
        net = torch.cat([net], -1)
        net = self.linearLayer(net)
        return net

if __name__ == "__main__":
    # loading the configurations from the JSON file
    json_path = json.load(open("config.json", "r"))

    #loading parameters from the JSON file
    batchSize = json_path["batchSize"]

    #loading dataset
    data = "Dataset/trainingset_1k.nc"
    #data = "Dataset/trainingset_5k.nc"

    #fetching, normalizing and transforming
    dataloader = DataLoader(dataFetcher(data),batch_size=batchSize, shuffle=True)

    #setting training and testing dataset
    #for 5k
    #train_set, test_set = torch.utils.data.random_split(dataloader, [65, 12])
    # for 1k
    train_set, test_set = torch.utils.data.random_split(dataloader, [11, 2])

    #running the computaion in GPU if not then CPU
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    training_set = []
    testing_set  = []

    for ind, (x, i) in enumerate(dataloader):
        if ind in train_set.indices:
             training_set.append([x,i])
        else:
             testing_set.append([x,i])

    # print("training set, testing set", len(training_set), len(testing_set))
    # print("shape for training and testing dataset", training_set[0][0].shape, testing_set[0][0].shape)
    # print("Total images in Testing set",sum([testing_set[x][0].shape[0] for x in range(len(testing_set))]))


    # creating a object
    simulator = simulatorNet()
    simulator = simulator.cuda()
    #optimizer
    simOptimizer = torch.optim.Adam(simulator.parameters(), lr=0.00002, betas=(json_path["simBeta1"], json_path["simBeta2"]))

    #loss function
    loss = nn.MSELoss()
    #loss = nn.SmoothL1Loss()

    sim_loss = []
    accuracy = []
    diffValue = []
    #training the simulator
    for itera in tqdm(range(json_path["epochSIM"])):
        correct = 0
        predTrans_temp = []
        sim_loss_temp = []
        diffValue_temp = []
        diffValue_temp1 = []

        #for n_batch, (realImg, labels) in enumerate(training_set):
        for n_batch, (realImg, labels) in enumerate(dataloader):

            realImg, labels = realImg.to(device), labels.to(device)

            # generating transmittance
            predTrans = simulator(realImg)

            # Resetting gradient to zero for optimizer
            simOptimizer.zero_grad()

            # loss, adjusting loss
            s_loss = loss(predTrans, labels)

            # back propagation to adjust the weights of the neural network
            s_loss.backward()
            simOptimizer.step()

            sim_loss_temp.append(s_loss.item())

            #diffValue = torch.abs(predTrans - labels)
            diffValue_temp = torch.abs(predTrans - labels)

            diffValue_temp1.append(diffValue_temp)

            diffValue.append(diffValue_temp1)
            diffValue_temp = torch.abs(predTrans-labels)

            correct += (diffValue_temp < json_path["simThreshold"]).float().sum()


        # for n_batch, (realImg, labels) in enumerate(testing_set):
        #
        #     realImg, labels = realImg.to(device), labels.to(device)
        #
        #     # generating transmittance
        #     predTrans = simulator(realImg)
        #
        #     # loss, adjusting loss
        #     s_loss = loss(predTrans, labels)
        #
        #     sim_loss_temp.append(s_loss.item())
        #
        #     diffValue_temp = torch.abs(predTrans-labels)
        #
        #     correct += (diffValue_temp < json_path["simThreshold"]).float().sum()

            #diffValue.append(diffValue_temp.cpu().detach().numpy())


        sim_loss.append(sum(sim_loss_temp)/len(sim_loss_temp))

        accuracy1 = 100 * correct / dataloader.dataset.transmittance.shape[0]
        #accuracy1 = 100 * correct / sum([testing_set[x][0].shape[0] for x in range(len(testing_set))])

        accuracy.append(accuracy1.item())


        print("Epoch {}/{}, Loss: {:.3f}".format(itera + 1, json_path["epochSIM"], sim_loss[itera]))
        print("Accuracy = {}".format(accuracy1))

    a = np.array(sim_loss)
    np.savetxt('sim_error_1k.txt', a, fmt='%.10f')

    b = np.array(accuracy)
    np.savetxt('accuracy_1k.txt', b, fmt='%.10f')

    #saving the simulator model
    # torch.save(simulator.state_dict(), 'simModel/model_s.pt')



