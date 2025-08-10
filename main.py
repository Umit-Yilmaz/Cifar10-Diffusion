# main.py
import math, random, os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
import tqdm

# Repro
seed = 42
random.seed(seed)
torch.manual_seed(seed)

# Sinusoidal embedding
def sinusoidal_embedding(timesteps, dim):
    timesteps = timesteps.float()
    device = timesteps.device
    half = dim // 2
    inv_freq = torch.exp(-math.log(10000) * torch.arange(half, device=device) / (half - 1))
    args = timesteps[:, None] * inv_freq[None, :]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

# Residual block with GroupNorm (lighter on small batches)
class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, class_emb_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.time_fc = nn.Linear(time_emb_dim, out_ch)
        self.class_fc = nn.Linear(class_emb_dim, out_ch)
        self.res_conv = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb, c_emb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_fc(t_emb)[:, :, None, None]
        h = h + self.class_fc(c_emb)[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.res_conv(x)

# Small U-Net with skip projections
class UNet(nn.Module):
    def __init__(self, num_classes, base_ch=16, time_emb_dim=64, class_emb_dim=64):
        super().__init__()
        # time MLP
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim*4),
            nn.SiLU(),
            nn.Linear(time_emb_dim*4, time_emb_dim)
        )
        self.class_emb = nn.Embedding(num_classes, class_emb_dim)

        self.in_conv = nn.Conv2d(3, base_ch, 3, padding=1)
        self.down1 = ResBlock(base_ch, base_ch*2, time_emb_dim, class_emb_dim)
        self.down2 = ResBlock(base_ch*2, base_ch*4, time_emb_dim, class_emb_dim)
        self.bot = ResBlock(base_ch*4, base_ch*4, time_emb_dim, class_emb_dim)
        self.up1 = ResBlock(base_ch*4, base_ch*2, time_emb_dim, class_emb_dim)
        self.up2 = ResBlock(base_ch*2, base_ch, time_emb_dim, class_emb_dim)

        # skip projection convs to match channels before adding
        self.skip_proj_4_to_2 = nn.Conv2d(base_ch*4, base_ch*2, 1)
        self.skip_proj_2_to_1 = nn.Conv2d(base_ch*2, base_ch, 1)

        self.out_conv = nn.Conv2d(base_ch, 3, 1)
        self.downsample = nn.AvgPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x, t, y):
        # t: (B,) ints
        t_emb = sinusoidal_embedding(t, self.time_mlp[0].in_features)
        t_emb = self.time_mlp(t_emb)
        c_emb = self.class_emb(y)

        x1 = self.in_conv(x)                     # B, C, 32,32
        x2 = self.down1(x1, t_emb, c_emb)        # B, 2C, 32,32
        x3 = self.downsample(x2)                 # B, 2C, 16,16
        x4 = self.down2(x3, t_emb, c_emb)        # B, 4C, 16,16
        x5 = self.downsample(x4)                 # B, 4C, 8,8

        mid = self.bot(x5, t_emb, c_emb)         # B, 4C, 8,8

        u = self.upsample(mid)                   # B, 4C, 16,16
        u = self.up1(u, t_emb, c_emb)            # B, 2C, 16,16
        u = u + self.skip_proj_4_to_2(x4)        # project x4 -> 2C then add

        u = self.upsample(u)                     # B, 2C, 32,32
        u = self.up2(u, t_emb, c_emb)            # B, C, 32,32
        u = u + self.skip_proj_2_to_1(x2)        # project x2 -> C then add

        return self.out_conv(u)                  # B,3,32,32

# DDPM engine (fixed posterior computation)
class DDPM:
    def __init__(self, model, timesteps=100, beta_start=1e-4, beta_end=0.02):
        self.model = model
        self.num_timesteps = timesteps
        self.betas = torch.linspace(beta_start, beta_end, timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), self.alphas_cumprod[:-1]])
        self.posterior_variance = self.betas * (1 - alphas_cumprod_prev) / (1 - self.alphas_cumprod)

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        return self.sqrt_alphas_cumprod[t][:, None, None, None] * x_start + \
               self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None] * noise

    def p_sample(self, x, t, y, guidance_scale=1.0):
        # classifier-free guidance
        cond = self.model(x, t, y)
        null_y = torch.full_like(y, fill_value=(self.model.class_emb.num_embeddings - 1))
        uncond = self.model(x, t, null_y)
        eps = uncond + guidance_scale * (cond - uncond)

        betas_t = self.betas[t][:, None, None, None]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        sqrt_recip_alphas_t = (1.0 / torch.sqrt(self.alphas[t]))[:, None, None, None]

        mean = sqrt_recip_alphas_t * (x - betas_t * eps / sqrt_one_minus_alphas_cumprod_t)
        if (t == 0).all():
            return mean
        else:
            noise = torch.randn_like(x)
            var = self.posterior_variance[t][:, None, None, None]
            return mean + torch.sqrt(var) * noise

    def sample(self, shape, device, y, guidance_scale=1.0):
        img = torch.randn(shape, device=device)
        for i in reversed(range(self.num_timesteps)):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            img = self.p_sample(img, t, y, guidance_scale)
        return img

# Training loop (RAM-friendly defaults)
def train():
    device = torch.device("cpu")
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    dataset = datasets.CIFAR10("./data", train=True, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0, pin_memory=False)

    num_classes = 11  # 10 + null
    model = UNet(num_classes=num_classes, base_ch=16, time_emb_dim=64, class_emb_dim=64).to(device)

    timesteps = 100
    diffusion = DDPM(model, timesteps=timesteps)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    epochs = 500
    os.makedirs("samples", exist_ok=True)
    for epoch in range(epochs):
        pbar = tqdm.tqdm(loader)
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device)
            # classifier-free guidance training: random null labels
            if random.random() < 0.1:
                labels = torch.full_like(labels, num_classes-1)
            t = torch.randint(0, diffusion.num_timesteps, (imgs.size(0),), device=device).long()
            noise = torch.randn_like(imgs)
            x_t = diffusion.q_sample(imgs, t, noise)
            pred_noise = model(x_t, t, labels)
            loss = F.mse_loss(pred_noise, noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_description(f"Epoch {epoch+1} Loss: {loss.item():.4f}")

        # quick sampling to inspect progress
        y_sample = torch.randint(0, 10, (16,), device=device)
        samples = diffusion.sample((16, 3, 32, 32), device=device, y=y_sample, guidance_scale=1.5)
        samples = (samples.clamp(-1, 1) + 1) / 2.0
        utils.save_image(samples, f"samples/sample_epoch_{epoch+1}.png", nrow=4)

    torch.save(model.state_dict(), "ddpm_cifar10_fixed.pth")
    print("Training done. Model saved to ddpm_cifar10_fixed.pth")

if __name__ == "__main__":
    os.makedirs("samples", exist_ok=True)
    train()
