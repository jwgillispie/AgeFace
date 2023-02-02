import torch 
import io
from PIL import Image
from torchvision.transforms import transforms

def get_model():

    model_path = r"C:\Users\jorda\OneDrive\Desktop\python\deep learning\ml apps\ml apps\model\resnet34_Jan-16-2023.pt"
    model = torch.load(model_path, map_location=torch.device('cpu'))
    return model

def get_tensor(image_bytes):
    my_transforms=transforms.Compose([transforms.Grayscale(num_output_channels=3), 
                                     transforms.Resize(255),
                                     transforms.CenterCrop(244),
		                             transforms.ToTensor(),
		                             transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    #image bytes from user 
    image = Image.open(io.BytesIO(image_bytes)) 
    
    return my_transforms(image).unsqueeze(0)

    
