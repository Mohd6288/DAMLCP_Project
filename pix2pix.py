import torch
import numpy as np
from torchvision.transforms import v2

device = "cuda" if torch.cuda.is_available() else "cpu"
PIX2PIX_PATH = "../../canvas/models/pix2pix_face2comics.iter_18000_scripted.pt"
G = torch.jit.load(PIX2PIX_PATH, map_location=device)

def generate_pix2pix(image):
    image = torch.permute(torch.tensor(image.copy()), (2, 0, 1))
    image = v2.ToImage()(image)
    if image.shape[0] == 1:
        image = image.repeat(3, 1, 1)
    elif image.shape[0] == 4:
        image = image[:3, :, :]
    image = v2.Resize((256, 256), antialias=True)(image)
    image = v2.ToDtype(torch.float32, scale=True)(image)
    image = image.to(device)[None, ...]
    with torch.no_grad():
        outputs = G(image).detach().cpu()
    output = outputs[0].permute(1, 2, 0) * 0.5 + 0.5
    return (output.numpy() * 255).astype(np.uint8)
