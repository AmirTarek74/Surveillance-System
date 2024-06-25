import torchvision 
from PIL import Image
class FaceRepresention:

    def represent(self,model,device,face):

        transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((160,160)),  
                torchvision.transforms.ToTensor()
                
            ])
        face = Image.fromarray(face)
        face = transform(face).to(device)
        embdding = model(face.unsqueeze(0)).detach()


        return embdding