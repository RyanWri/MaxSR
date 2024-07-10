import torch
from torchvision import transforms


# Define image postprocessing function
def postprocess_image(tensor):
    tensor = tensor.squeeze(0)  # Remove batch dimension
    tensor = tensor * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    tensor = tensor + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    tensor = torch.clamp(tensor, 0, 1)
    image = transforms.ToPILImage()(tensor)
    return image
