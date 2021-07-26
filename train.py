
import argparse
import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image
from PIL import Image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

from datasets import *
from models import *
from utils import *

import torch.nn as nn
import torch.nn.functional as F
import torch

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

def save_sample(batches_done):
    samples, croped_samples,labels, _ = next(iter(test_dataloader))
    samples = Variable(samples.type(FloatTensor))
    croped_samples = Variable(croped_samples.type(FloatTensor))
    # Generate inpainted image            
    z = Variable( torch.FloatTensor( np.random.uniform(-1,1,(len(croped_samples),100)) ).to(device) )
    gen_features, gen_crop = generator(croped_samples,z)  #gen_validity, 
    # Save sample
    sample = torch.cat((gen_crop.data, samples.data), -2)
    save_image(sample, "images/%d.png" % batches_done, nrow=6, normalize=True)

def validate(val_loader, model):
    top1       = AverageMeter()
    top5       = AverageMeter()
    top10      = AverageMeter()
    top50      = AverageMeter()
    
    # switch to evaluate mode
    model.eval()
    
    for i, (_, croped_imgs, labels, _) in enumerate(val_loader):   
        labels = labels-1
        indata = croped_imgs.to(device)
        target = labels.to(device)
        z = Variable( torch.FloatTensor( np.random.uniform(-1,1,(len(indata),100)) ).to(device) )
        with torch.no_grad():
            input_var  = Variable(croped_imgs.type(FloatTensor))
            target_var = Variable(labels.type(LongTensor))

        # compute output
        output,_ ,_= model(input_var, z)
        #loss   = criterion(output, target_var)
                
        # measure accuracy and record loss
        prec1, prec5, pre10, pre50 = accuracy(output.data, target, topk=(1,5,10,50))
        #losses.update(loss.item(), indata.size(0))
        top1.update(prec1.item(),  indata.size(0))
        top5.update(prec5.item(),  indata.size(0))
        top10.update(pre10.item(), indata.size(0))
        top50.update(pre50.item(), indata.size(0))

    print('\nTest set: Prec@1 : {:.4f}, Prec@5 {:.4f}, Prec@10 {:.4f},  Prec@50 {:.4f}\n'
          .format(top1.avg, top5.avg, top10.avg, top50.avg))

    return top1.avg, top5.avg, top10.avg, top50.avg

if __name__ == '__main__':
    os.makedirs("images", exist_ok=True)
    os.makedirs("checkpoint", exist_ok=True)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
    parser.add_argument("--n_epochs", type=int, default=100, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=16, help="size of the batches")
    parser.add_argument("--root_path", type=str, default='E:\data', help="dataset root path")
    parser.add_argument("--dataset_name", type=str, default="prepared_image/img_align_celeba_crop/complete", help="name of the dataset")
    parser.add_argument("--dataset_label", type=str, default="prepared_image/identity_CelebA.txt", help="name of the dataset label")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--num_classes", type=int, default=10177, help="number of the clesses")              
    parser.add_argument("--img_size", type=int, default=128, help="size of each image dimension")
    parser.add_argument("--mask_size", type=int, default=64, help="size of random mask")
    parser.add_argument("--channels", type=int, default=3, help="number of image channels")
    parser.add_argument("--sample_interval", type=int, default=500, help="interval between image sampling")
    parser.add_argument("--checkpoint_interval", type=int, default=5000, help="batch interval between model checkpoints")
    opt = parser.parse_args(args=[])

    img_shape = (opt.channels, opt.img_size, opt.img_size)

    cuda = True if torch.cuda.is_available() else False
    device = torch.device("cuda" if cuda else "cpu")
     
    # Losses
    pixelwise_loss = torch.nn.L1Loss()
    similarity_preserve = torch.nn.L1Loss()
    cross_entropy = torch.nn.CrossEntropyLoss()
    # other loss:
    #g_loss and d_loss

    # Initialize generator and discriminator
    generator = GeneratorRes(channels=opt.channels, img_size = opt.img_size, num_classes = opt.num_classes)
    discriminator = Discriminator(channels=opt.channels)

    # Initialize other function

    # load model checkpoint
    if opt.epoch != 0:
        # Load pretrained models
        generator.load_state_dict(torch.load("./checkpoint/generator_%d.pth" % opt.epoch))
        discriminator.load_state_dict(torch.load("./checkpoint/discriminator_%d.pth" % opt.epoch))
    else:
        # Initialize weights
        generator.apply(weights_init_normal)
        discriminator.apply(weights_init_normal)

    if cuda:
        generator.cuda()
        discriminator.cuda()
        similarity_preserve.cuda()
        cross_entropy.cuda()

    # Configure data loader          
    transforms_ = [
        transforms.Resize((opt.img_size, opt.img_size), Image.BICUBIC),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]

    dataloader = DataLoader(
        ImageDataset(opt.root_path, opt.dataset_label, opt.dataset_name, 'prepared_image/gallery_list.txt',
                     transforms_=transforms_, 
                     img_size=opt.img_size, mask_size=opt.mask_size),
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_cpu,
        )

    test_dataloader = DataLoader(
        ImageDataset(opt.root_path, opt.dataset_label, opt.dataset_name, 'prepared_image/gallery_list.txt',
                     transforms_=transforms_, 
                     img_size=opt.img_size, mask_size=opt.mask_size,mode="test"),
        batch_size=12,
        shuffle=True,
        num_workers=opt.n_cpu,
        )

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor
        
    # ----------
    #  Training
    # ----------
    gamma = 0.75
    lambda_k = 0.001
    k = 0.0
    for epoch in range(opt.epoch, opt.n_epochs):
        for i, (imgs_gt, croped_imgs, labels, imgs) in enumerate(dataloader):

            batches_done = epoch * len(dataloader) + i

            # Configure input   
            gt_imgs = Variable(imgs_gt.type(FloatTensor))
            real_imgs = Variable(imgs.type(FloatTensor))
            croped_imgs = Variable(croped_imgs.type(FloatTensor))
            labels = labels-1
            labels = Variable(labels.type(LongTensor))
            z = Variable( torch.FloatTensor( np.random.uniform(-1,1,(len(gt_imgs),100)) ).to(device) )
            # Avoid possble memory leak
            del imgs_gt, imgs
      
            # -----------------------------------
            #  Train Generator 
            # -----------------------------------
            optimizer_G.zero_grad()
            
            for idx in range(2):  #Generator loop 2       
                # 1              reconstruction error only
                # Measure pixel-wise loss against ground truth
                for idx_pixel in range(3): 
                    gen_features, gen_imgs = generator(croped_imgs,z)  #gen_validity,  
                    g_pixel = pixelwise_loss(gen_imgs, real_imgs)
                    g_pixel.backward()
                    optimizer_G.step() 
            
                # Generate a batch of images
                gen_features, gen_imgs = generator(croped_imgs,z)   #corrupted data
                gt_features,_ = generator(gt_imgs,z)
                # total generator loss   
                # 1              reconstruction error
                g_pixel = pixelwise_loss(gen_imgs, real_imgs)
                           
                # 2.similarity preservation term 
                g_sp = similarity_preserve(gen_features, gt_features)
            
                # 3             adversarial loss  
                # Loss measures generator's ability to fool the discriminator
                g_adv = torch.mean(torch.abs(discriminator(gen_imgs) - gen_imgs))     
        
                g_loss = g_pixel + g_sp + 5e-1*g_adv 
                g_loss.backward()     #retain_graph=True
                optimizer_G.step()
        
            # Avoid possble memory leak
            del gen_features, gt_features, _
        
            # ----------------------------------------------------------------------
            #  Train Discriminator
            # ----------------------------------------------------------------------
            optimizer_D.zero_grad()

            # Measure discriminator's ability to classify real from generated samples
            d_real = discriminator(real_imgs)
            d_fake = discriminator(gen_imgs.detach())
        
            d_loss_real = torch.mean(torch.abs(d_real - real_imgs))
            d_loss_fake = torch.mean(torch.abs(d_fake - gen_imgs.detach()))     
            d_loss = d_loss_real - k * d_loss_fake
            d_loss.backward()
            optimizer_D.step()
        
            # ----------------
            # Update weights
            # ----------------
            diff = torch.mean(gamma * d_loss_real - d_loss_fake)
        
            # Update weight term for fake samples
            k = k + lambda_k * diff.item()
            k = min(max(k, 0), 1)  # Constraint to interval [0, 1]

            # Update convergence metric
            M = (d_loss_real + torch.abs(diff)).data     #[0]

            # --------------
            # Log Progress
            # --------------
            if batches_done % 50 == 0:
                print(
                    "[Epoch %d/%d] [Batch %d/%d] [G loss: %f, pixel: %f, sim pre: %f, adv: %f] [D loss: %f] -- M: %f, k: %f"
                    % (
                        epoch, opt.n_epochs, 
                        i, len(dataloader), 
                        g_loss.item(), g_pixel.item(), g_sp.item(),g_adv.item(), 
                        d_loss.item(),                     
                        M, k
                        ))

            if batches_done % opt.sample_interval == 0:
                save_sample(batches_done)
               
            # save model parameters
            if (batches_done) % opt.checkpoint_interval == 0:
                torch.save(generator.state_dict(), "checkpoint/generator_%d.pth" % epoch)
                torch.save(discriminator.state_dict(), "checkpoint/discriminator_%d.pth" %epoch)
            validate(val_dataloader, generator)


