import numpy as np
import os
import shutil
import torch
from torch.utils.data import DataLoader
import glob
from common import load_data, run_test
from net import Net

if torch.cuda.is_available():     #check for the cuda module
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

save_images = 'restored_images'     

shutil.rmtree(save_images, ignore_errors = True)   #remove the directory if its places there
os.makedirs(save_images)     #create a new directory with save_images

test_files = glob.glob('C:\\Users\\91812\\Desktop\\img3.ARW')     #retrieve file and path name
dataloader_test = DataLoader(load_data(test_files), batch_size=1, shuffle=False, num_workers=0, pin_memory=True)     #loads the data

model = Net()   #calling Net class
print('\n Network parameters : {}\n'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
model = model.to(device)
print('Device on GPU: {}'.format(next(model.parameters()).is_cuda))
checkpoint = torch.load('C:\\Users\\91812\\Desktop\\weights', map_location=device)   #calling the weights
model.load_state_dict(checkpoint['model'])

run_test(model, dataloader_test, save_images)    #run the model
print('Restored images saved in RESTORED_IMAGES directory')