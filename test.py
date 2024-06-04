#Import necessary libraries and modules
import argparse
import torch
from torchvision import datasets, transforms, models 
from torch import optim
from PIL import Image 
import numpy as np
import json


#Get command line user inputs
def get_input_args():
    parser = argparse.ArgumentParser(description="Predict top classes using a pretrained neural network on a dataset of images")
    parser.add_argument('--user_image_path', type=str, default='flowers/test/6/image_07181.jpg', help='Image')
    parser.add_argument('--user_load_chkpt', type=str, default='checkpoint.pth', help='Path file to load the trained model checkpoint')
    parser.add_argument('--user_dispnames', action='store_true', help='Display class names')
    parser.add_argument('--user_k', type=int, default=5, help='K for top K classes and their probabilities')
    parser.add_argument('--user_jsonfl', type=str, default='cat_to_name.json', help='Display class names')
    parser.add_argument('--user_gpu', action='store_true', help='Use GPU for training')
    return parser.parse_args()


# TODO: Write a function that loads a checkpoint and rebuilds the model
def load_checkpoint(user_load_chkpt): 
    # Load the checkpoint
    checkpoint = torch.load(user_load_chkpt)

    # Rebuild the model
    model = models.vgg16(pretrained=True) 
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    # Load the class-to-index mapping
    model.class_to_idx = checkpoint['class_to_idx']
    

    # # Optionally, set the model to evaluation mode
    # model.eval()
    return model 

def process_image(user_image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    with Image.open(user_image_path) as im: 

        short_side = 256
        im.thumbnail((short_side, short_side), Image.Resampling.LANCZOS)
        width, height = im.size         
#         print('min',min(im.size), 'max',max(im.size)) 
        new_width = 224
        new_height = 224
        
        left = (width - new_width)/2
        right = (width + new_width)/2
        top = (height - new_height)/2
        bottom = (height + new_height)/2      
        
        im = im.crop((left, top, right, bottom))
        
        np_im = np.array(im)

        # Ensure the image is of type float
        np_im = np_im.astype(np.float32)/255 
        
        
        means = np.array([0.485, 0.456, 0.406])
        std_dev = np.array([0.229, 0.224, 0.225])
        
        nrm_im = (np_im-means)/std_dev

        trnsp_im = nrm_im.transpose((2,0,1))
        
        trnsp_im = torch.from_numpy(trnsp_im)
        
        return trnsp_im
    
def predict(user_image_path, device, model, user_k):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file

    

    # # Move the model to the appropriate device
    model.to(device)

    model.eval()
    

    img2 = user_image_path.float()
    print(img2.dtype)


    #Add a batch dimension and move to the same device as the model
    image = img2.unsqueeze(dim=0)#0).to(device)

    with torch.no_grad(): 
        output = model.forward(image) 
    
    ps = torch.exp(output) 

    top_p, top_class = ps.topk(user_k) 

    top_most_p, top_most_class = ps.topk(1)

    return top_most_p, top_most_class, top_p, top_class

# Function to get class name from index
def get_class_name(index, model, cat_to_name): 
    idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    class_label = idx_to_class[index]
    return cat_to_name[class_label]

def main(): 
    #Get input arguments from user or default
    args = get_input_args() 
    device = torch.device("cuda" if args.user_gpu and torch.cuda.is_available() else "cpu")

    #Process image
    img = process_image(args.user_image_path)#user_image

    #Load checkpoint 
    model_test = load_checkpoint(args.user_load_chkpt) #user_chkpt

    #Top most class and probability and top k classes and probabilities 
    top_prob, top_class, probs, classes = predict(img, device, model_test, args.user_k) #user_k

    print('The most likely image class is ', top_class, ' with probability ', top_prob) 
    probs_lst = np.array(probs)
    top_prob_idx = np.where(probs_lst == np.max(probs_lst))
    print('check:',np.max(probs_lst))
    # top_p_idx = probs_lst.index(top_prob)

    print('The top ', args.user_k, 'most likely image classes are ', classes, ' with probabilities ', probs) 

    if args.user_dispnames: 
        
        with open(args.user_jsonfl, 'r') as f:
            cat_to_name = json.load(f) 
        lst_classes = classes.tolist()
        class_names = [get_class_name(key1, model_test, cat_to_name) for key in lst_classes for key1 in key]
        print(class_names, type(class_names))
        print('The most likely image class name is ', class_names[int(top_prob_idx[0])], ' with probability ', top_prob) 
        print('The top ', args.user_k, 'most likely image classes are ', class_names, ' with probabilities ', probs) 

if __name__ == "__main__":
    main()