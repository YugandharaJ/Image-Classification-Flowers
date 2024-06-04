#Import necessary libraries and modules
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models 
from torch.utils.data import DataLoader
import os
import sys

#Get command line user inputs
def get_input_args():
    parser = argparse.ArgumentParser(description="Train a neural network on a dataset of images")
    parser.add_argument('--user_data_dir', type=str, default='flowers', help='Dataset directory')
    parser.add_argument('--user_save_dir', type=str, default='checkpoint.pth', help='Directory with path file name and ext to save the trained model checkpoint')
    parser.add_argument('--user_arch', type=str, default='vgg16', choices=['vgg16', 'densenet121'], help='Model architecture')
    parser.add_argument('--user_learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--user_hidden_units', type=int, default=512, help='Number of hidden units')
    parser.add_argument('--user_epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--user_gpu', action='store_true', help='Use GPU for training')
    return parser.parse_args()

#Transforms: 
def load_transforms(user_data_dir): 
    train_dir = user_data_dir + '/train'
    valid_dir = user_data_dir + '/valid'
    # TODO: Define your transforms for the training, validation, and testing sets
    data_transforms = transforms.Compose([transforms.Resize(255), 
                                        transforms.CenterCrop(224), 
                                        transforms.ToTensor()]) 

    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                        transforms.RandomResizedCrop(224),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(), 
                                        transforms.Normalize([0.485, 0.456, 0.406], 
                                                                [0.229, 0.224, 0.225])]) 

    valid_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(), 
                                        transforms.Normalize([0.485, 0.456, 0.406], 
                                                                [0.229, 0.224, 0.225])])

    # TODO: Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms) 

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    trainloader = DataLoader(train_data, batch_size=64, shuffle=True)
    validloader = DataLoader(valid_data, batch_size=64, shuffle = True) 

    return trainloader, validloader, train_data.class_to_idx 

#Build model: 
def build_model(user_arch, user_hidden_units, class_to_idx): 
    if user_arch == 'vgg16': 
        model = models.vgg16(pretrained=True)
        input_units = model.classifier[0].in_features
    elif user_arch == 'densenet121':
        model = models.densenet121(pretrained=True)
        input_units = model.classifier.in_features

    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False 

    #Modify classifier to train on new data 
    classifier = nn.Sequential(nn.Linear(input_units, user_hidden_units, bias=True), 
                           nn.ReLU(inplace=True), 
                           nn.Dropout(p=0.5, inplace=False), 
                           nn.Linear(user_hidden_units, 102, bias=True), 
                           nn.LogSoftmax(dim=1))

    model.classifier = classifier 
    model.class_to_idx = class_to_idx 

    return model 

def train_model(model, trainloader, validloader, criterion, optimizer, device, user_epochs): 
    model.to(device)

    epochs = user_epochs #user_epochs # Change
    steps = 0
    print_every = 5 #How often validation is performed and accuracy is calculated


    train_losses, valid_losses = [], []


    for e in range(epochs):
        running_loss = 0
        for ii, (images, labels) in enumerate(trainloader): 

            steps += 1

            # Move input and label tensors to the GPU
            images, labels = images.to(device), labels.to(device)
            print('steps',steps,'Images:', len(images),'labels', len(labels))
            optimizer.zero_grad()
            
            log_ps = model.forward(images)
            loss = criterion(log_ps, labels)
            loss.backward()
            optimizer.step()
            
            
            running_loss += loss.item() 
            # train_loss += loss.item() 
            if steps % print_every == 0:
                valid_loss = 0
                accuracy = 0
                model.eval()
                model.to(device)
                with torch.no_grad():
                    for images, labels in validloader:
                        images, labels = images.to(device), labels.to(device)
                        
                        logps = model.forward(images)
                        batch_loss = criterion(logps, labels)
                        
                        valid_loss += batch_loss.item()
                        
                        # Calculate accuracy
                        ps = torch.exp(logps)

                        equality = (labels.data == ps.max(dim=1)[1])
                        accuracy += equality.type(torch.FloatTensor).mean()

                        
                # At completion of epoch
                train_losses.append(running_loss/print_every)
                valid_losses.append(valid_loss/len(validloader))

                print(f"Epoch {e+1}/{epochs}.. "
                    f"Train loss: {running_loss/print_every:.3f}.. "
                    f"Valid loss: {valid_loss/len(validloader):.3f}.. "
                    f"Test accuracy(fractions): {accuracy/len(validloader):.3f}")
                

                running_loss = 0

                model.train()        

def save_checkpoint(model, user_arch, user_save_dir, optimizer, user_epochs, user_hidden_units):
    checkpoint = {
        'state_dict': model.state_dict(),
        'class_to_idx': model.class_to_idx,
        'optimizer_state_dict': optimizer.state_dict(),
        'classifier': model.classifier,
        'epochs': user_epochs,
        'arch': user_arch,
        'hidden_units': user_hidden_units,
    }
    torch.save(checkpoint, user_save_dir)

def main():
    args = get_input_args()

    trainloader, validloader, class_to_idx = load_transforms(args.user_data_dir)
    device = torch.device("cuda" if args.user_gpu and torch.cuda.is_available() else "cpu")

    model = build_model(args.user_arch, args.user_hidden_units, class_to_idx)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.user_learning_rate)

    train_model(model, trainloader, validloader, criterion, optimizer, device, args.user_epochs)
    save_checkpoint(model, args.user_arch, args.user_save_dir, optimizer, args.user_epochs, args.user_hidden_units)

if __name__ == "__main__":
    main()

