import torchvision 
from PIL import Image
from facenet_pytorch import  InceptionResnetV1
import torch

class FaceRepresention:
    def __init__(self) :
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
    
    def represent(self,face):

        transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((160,160)),  
                torchvision.transforms.ToTensor()
                
            ])
        face = Image.fromarray(face)
        face = transform(face).to(self.device)
        embdding = self.model(face.unsqueeze(0)).detach()


        return embdding