import torch
from tqdm import tqdm
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import ISBI_Loader
from unetpp_model2 import Unetplusplus
from torch.cuda.amp import autocast , GradScaler
from torchvision import transforms
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt


# load the data
data_path = "E:/finalproject/ACDC_dataset/ACDC_MYO/20240313_ACDC_1_2"
acdc_dataset = ISBI_Loader(data_path)
batch_size = 2
if len(acdc_dataset) == 0:
    raise ValueError("the dataset is empty!")
dataloader = DataLoader(acdc_dataset, batch_size = batch_size, shuffle=True)
per_epoch_num = len(acdc_dataset) / batch_size

model = Unetplusplus(num_classes=1, input_channels=1,deep_supervision=False)
print(model)

# set the loss function and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters())
scheduler = lr_scheduler.StepLR(optimizer,step_size=15,gamma=0.1)
train_loss_history =[]
best_loss = float('inf')

# training
num_epochs = 40
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
print(device)
model.to(device)

scaler = GradScaler()
with tqdm(total=num_epochs*per_epoch_num) as pbar:
    for epoch in range(num_epochs):

        model.train()
        epoch_loss = 0.0
        for image,label in dataloader:
        #inputs, labels = inputs.to(device), labels.to(device)
        # plt.figure()
        # plt.subplot(1, 2, 1)
        # plt.title('Image')
        # plt.imshow(image.squeeze(), cmap='gray')
        # plt.subplot(1, 2, 2)
        # plt.title('Label')
        # plt.imshow(label.squeeze(), cmap='gray')
        # plt.show()

           optimizer.zero_grad()
           image = image.to(device=device, dtype=torch.float32)
           label = label.to(device=device, dtype=torch.float32)
           outputs = model(image)
       # print('label:',label.shape,label.type)    #torch.Size([1, 1, 512, 512])   tensor
       # print('outputs:',outputs.shape)   #outputs: torch.Size([1, 1, 512, 512])

           loss = criterion(outputs, label)
           epoch_loss += loss.item()
         # with autocast():
         #     outputs = model(image)
         #    #outputs = outputs.float()
         #     loss = criterion(outputs, label)
         #   train_loss_history.append(loss.detach().cpu().numpy())

           scaler.scale(loss).backward()
           scaler.step(optimizer)
           scaler.update()
         #   loss.requires_grad_(True)
         #   loss.backward()
         #   optimizer.step()
           pbar.update(1)
           if loss < best_loss:
               best_loss = loss
               torch.save(model.state_dict(), 'unetplusplus_model_MYO_1_2.pth')    #set the save path

        scheduler.step()
        epoch_loss /= len(dataloader)
        train_loss_history.append(epoch_loss)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {train_loss_history[-1]},Learning Rate:{scheduler.get_lr()[0]}')

    train_loss_history = np.array(train_loss_history)
    plt.plot(range(1, num_epochs + 1), train_loss_history, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.show()

print("The unetplusplus model training has completed successfully!")
print( "The trained model has been saved!")
