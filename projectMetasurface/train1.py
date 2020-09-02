from tqdm import tqdm
from torch.autograd import Variable
import torch
import numpy as np
import logging
from torchvision.utils import save_image
import torch.nn.functional as F


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

def generating_images(generator, fig_path, n_row = 4, n_col = 4):
    generator.eval()
    transmittance = torch.linspace(-1, 1, n_row).view(n_row, 1).repeat(1, n_col).view(-1, 1)
    labels = torch.cat([transmittance], -1).to(device)
    noise = Variable(torch.cuda.FloatTensor(labels.size(0), generator.noiseDim).normal_())
    noise.cuda()
    imgs, _ = generator(labels,noise), noise
    imgs = F.pad(imgs, (0, 0, 0, imgs.size(2)-1), mode='reflect')
    save_image(imgs, fig_path, n_row)
    generator.train()

def gradient_penalty(D, real_samples, fake_samples, labels):
    # Random weight term for interpolation between real and fake samples
    alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1)))
    labels = Tensor(labels)
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates, labels)
    fake = Tensor(real_samples.shape[0], 1).fill_(1.0)
    fake.requires_grad = False
    # Get gradient w.r.t. interpolates
    gradients = torch. autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )
    gradients = gradients[0].view(gradients[0].size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def train(generator, discriminator,genOptimizer, disOptimizer, dataloader, json_path):

    gen_loss_history = []
    dis_loss_history = []

    generator.train()
    discriminator.train()
    a = 0
    for fakeImg in tqdm(range(json_path["epochGAN"])):
        d_overallError =[]
        g_overallError = []


        for n_batch, (realImg, labels) in enumerate(dataloader):

            #running in the GPU if available
            realImg, labels = realImg.to(device, dtype=torch.float), labels.to(device)

            # Training Discriminator

            #Resetting gradient to zero
            disOptimizer.zero_grad()

            #generating noise for the generator
            noise = Variable(torch.randn(labels.size(0), json_path["noiseDim"]).to(device))

            #generating fake images
            fakeImg = generator(labels, noise)

            # with torch.no_grad():
            #     transFake = simulator(fakeImg)
            # sim_loss = ((labels - transFake)**2)/json_path["batchSize"]

            #loss, adjusting loss
            disloss_real = discriminator(realImg, labels)

            # Fake images
            disloss_fake =discriminator(fakeImg, labels)

            graPen = gradient_penalty(discriminator, realImg.data, fakeImg.data, labels.data)

            #Loss
            d_loss = -torch.mean(disloss_real) + torch.mean(disloss_fake) + json_path["lambda_for_GP"] * graPen
            # loss, adjusting weights
            d_loss.backward()
            disOptimizer.step()

            d_overallError.append(d_loss.data)

            genOptimizer.zero_grad()

            # Train the generator with each step
            if a % json_path["nCritic"]  == 0:
                # Generating images in batches
                fakeImgs1 = generator(labels, noise)
                valFAke = discriminator(fakeImgs1, labels)
                # generator loss measures the ability to fake the discrminator
                g_loss = -torch.mean(valFAke)

                g_loss.backward()
                genOptimizer.step()

            if a % 1 == 0:
                fig_path = 'images/gen/iter{}.png'.format(a)
                generating_images(generator, fig_path)

            g_overallError += [g_loss.data] * json_path["nCritic"]
        a += 1
        dis_loss_history.append(sum(d_overallError) / len(d_overallError))
        gen_loss_history.append(sum(g_overallError) / len(g_overallError))

    return [w.item() for w in dis_loss_history], [w.item() for w in gen_loss_history]


