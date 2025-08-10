# inference.py
import os
import torch
from torchvision import utils
from main import UNet, DDPM  # aynı dizinde main.py ise import edilebilir

def infer(ckpt="ddpm_cifar10_fixed.pth", out="gen.png", n_samples=16, cls=None, guidance=2.0):
    device = torch.device("cpu")
    num_classes = 11
    model = UNet(num_classes=num_classes, base_ch=16, time_emb_dim=64, class_emb_dim=64).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    diffusion = DDPM(model, timesteps=100)  # timesteps must match eğitilmiş model
    # prepare labels
    if cls is None:
        y = torch.randint(0, 10, (n_samples,), device=device)
    else:
        y = torch.full((n_samples,), cls, device=device, dtype=torch.long)

    with torch.no_grad():
        samples = diffusion.sample((n_samples, 3, 32, 32), device=device, y=y, guidance_scale=guidance)
        samples = (samples.clamp(-1, 1) + 1) / 2.0
        os.makedirs("inference_samples", exist_ok=True)
        utils.save_image(samples, os.path.join("inference_samples", out), nrow=int(max(1, math.sqrt(n_samples))))
    print(f"Saved generated images to inference_samples/{out}")

if __name__ == "__main__":
    import math
    infer(ckpt="ddpm_cifar10_fixed.pth", out="generated.png", n_samples=16, cls=None, guidance=2.0)
