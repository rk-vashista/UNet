import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
from unet import UNet


def single_image_inference(image_pth, model_pth, device):
    model = UNet(in_channels=3, num_classes=1).to(device)
    model.load_state_dict(torch.load(
        model_pth, map_location=torch.device(device)))

    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()])

    img = Image.open(image_pth).convert('RGB')
    img = transform(img).float().to(device)
    img = img.unsqueeze(0)

    pred_mask = model(img)

    img = img.squeeze(0).cpu().detach()
    img = img.permute(1, 2, 0)

    pred_mask = pred_mask.squeeze(0).cpu().detach()
    pred_mask = pred_mask.permute(1, 2, 0)
    pred_mask[pred_mask < 0] = 0
    pred_mask[pred_mask > 0] = 1

    fig = plt.figure()
    for i in range(1, 3):
        fig.add_subplot(1, 2, i)
        if i == 1:
            plt.imshow(img, cmap="gray")
        else:
            plt.imshow(pred_mask, cmap="gray")
    
    plt.savefig('./res/result.png')


if __name__ == "__main__":
    SINGLE_IMG_PATH = "./data/train/image/1.png"
    DATA_PATH = "./data"
    MODEL_PATH = "./models/unet.pth"

    device = "cpu"
    single_image_inference(SINGLE_IMG_PATH, MODEL_PATH, device)