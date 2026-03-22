import sys
import cv2
import numpy as np
import torch
from torch import optim
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm

torch._logging.set_logs(graph_code=True, recompiles=True)  # type: ignore


class SobelFilter(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels * 3
        self.conv = nn.Conv2d(
            self.in_channels,
            self.out_channels,
            kernel_size=3,
            padding=1,
            groups=in_channels,
            bias=False,
            padding_mode="circular",
        )
        self.reset_params()

    def reset_params(self):
        identity = torch.tensor([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
        sobel_x = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
        sobel_y = sobel_x.T
        kernel = torch.stack([identity, sobel_x, sobel_y])[:, None, :, :]
        with torch.no_grad():
            self.conv.weight.copy_(kernel.repeat(self.in_channels, 1, 1, 1))

    def forward(self, x):
        return self.conv(x)


class CANN(nn.Module):
    def __init__(self, update_rate=0.25):
        super(CANN, self).__init__()
        self.update_rate = update_rate

        self.perception = SobelFilter(CHANNELS)
        self.perception.conv.weight.requires_grad = False

        self.seq = nn.Sequential(
            nn.Conv2d(CHANNELS * 3, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(64, CHANNELS, kernel_size=1, bias=False),
        )

        with torch.no_grad():
            self.seq[-1].weight.zero_()  # type: ignore

    def step(self, x, update_rate=None):
        pre_alive_mask = self.get_alive_mask(x)

        y = self.perception(x)
        y = self.seq(y)

        update_mask = self.get_update_mask(y.shape, update_rate)
        # x.add_(y * update_mask)
        x = x + y * update_mask

        post_alive_mask = self.get_alive_mask(x)
        combined_mask = pre_alive_mask * post_alive_mask
        # x.mul_(combined_mask)
        x = x * combined_mask

        return x

    def get_update_mask(self, shape, update_rate=None):
        b, _, h, w = shape
        update_rate = update_rate or self.update_rate
        update_mask = (torch.rand(b, 1, h, w) < update_rate).float()
        return update_mask

    def get_alive_mask(self, x, threshold=0.1):
        x = F.pad(x, (1, 1, 1, 1), mode="circular")
        alpha = x[:, 3:4, :, :]
        mask = F.max_pool2d(alpha, kernel_size=3, stride=1, padding=0) > threshold
        return mask.float()

    def forward(self, x, steps=1, update_rate=None):
        for i in range(steps):
            x = self.step(x, update_rate=update_rate)
        return x


class Pool:
    def __init__(self, target_img, pool_size=1024):
        self.target_img = target_img
        self.pool_size = pool_size
        self.reset()

    def reset(self):
        self.seed = get_seed()
        self.pool = self.seed.repeat(self.pool_size, 1, 1, 1)

    def sample(self, num_samples):
        self.idxs = torch.randperm(self.pool_size)[:num_samples]
        batch = self.pool[self.idxs, ...]

        losses = F.mse_loss(
            batch[:, :4, :, :],
            self.target_img.repeat(BATCH_SIZE, 1, 1, 1),
            reduction="none",
        ).sum(dim=(1, 2, 3))

        replace_idx = int(torch.argmax(losses).item())
        batch[replace_idx] = self.seed[0]

        return batch

    def sample_damaged(self, num_samples, damaged_samples=3):
        self.idxs = torch.randperm(self.pool_size)[:num_samples]
        batch = self.pool[self.idxs, ...]

        losses = F.mse_loss(
            batch[:, :4, :, :],
            self.target_img.repeat(BATCH_SIZE, 1, 1, 1),
            reduction="none",
        ).sum(dim=(1, 2, 3))

        sorted_idxs = torch.argsort(losses)
        replace_idx = sorted_idxs[-1]
        batch[replace_idx] = self.seed[0]
        damaged_idxs = sorted_idxs[:damaged_samples]
        batch[damaged_idxs] = create_hole(batch[damaged_idxs])

        return batch

    def update(self, new_samples):
        new_samples = new_samples.detach()
        # replaces samples in batch returned by sample()
        self.pool[self.idxs] = new_samples


def load_image(path: str):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"u silly goof u did the wrong image path: {path}")
        exit()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    img = img.astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    return img_tensor


def nnout_to_img(x):
    y = torch.clamp(x[0, :4], 0.0, 1.0)
    rgb = y[:3]
    alpha = y[3:4]
    bg_color = 1.0
    blended = rgb * alpha + bg_color * (1 - alpha)
    img = blended.detach().permute(1, 2, 0).numpy()
    return img


def nnout_hidden_to_img(x):
    y = x[0, :4].detach().permute(1, 2, 0)
    y = torch.clamp(y, 0, 1)
    return y.numpy()


def batch_to_img(x):
    y = x[:, :4].detach()
    y = torch.cat(torch.unbind(y, dim=0), dim=2)
    y = torch.clamp(y.permute(1, 2, 0), 0, 1)
    return y.numpy()


def alpha_blend_img(x):
    return x
    bgr = x[:, :, 0:3]
    a = x[:, :, 3]  # / 255.0
    bg = np.full_like(bgr, 255)
    bgr = (a[..., None] * bgr) + ((1 - a[..., None]) * bg)
    return bgr


def get_seed():
    seed = torch.zeros(1, CHANNELS, IMG_HEIGHT, IMG_WIDTH)
    seed[:, :, IMG_HEIGHT // 2, IMG_WIDTH // 2] = 1.0
    return seed


def create_hole(batch):
    b, c, h, w = batch.shape
    grid = torch.meshgrid(
        torch.linspace(-1, 1, h), torch.linspace(-1, 1, w), indexing="ij"
    )
    grid = torch.stack(grid, dim=0).unsqueeze(0).repeat(b, 1, 1, 1)
    center = torch.rand(b, 2, 1, 1) - 0.5
    radius = 0.3 * torch.rand(b, 1, 1, 1) + 0.1
    mask = ((grid - center) * (grid - center)).sum(1, keepdim=True).sqrt() > radius
    return batch * mask.float()


MODEL_NAME = "thegrassisgreen_full"
CHANNELS = 16
IMG_WIDTH = 150
IMG_HEIGHT = 16
IMG_PATH = "images/text/thegrassisgreen_full.png"
EPOCHS = 20000
BATCH_SIZE = 8

LR = 2e-3
LR_GAMMA = 0.9999
BETAS = (0.5, 0.5)

if __name__ == "__main__":
    cann = CANN()
    # cann_compiled = torch.compile(CANN())

    seed = get_seed()

    optimizer = optim.Adam(cann.parameters(), lr=LR, betas=BETAS)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, LR_GAMMA)

    loaded_epoch = 0

    if len(sys.argv) == 2:
        state = torch.load(sys.argv[1])
        MODEL_NAME = state["name"]
        loaded_epoch = state["epoch"] + 1
        cann.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        scheduler.load_state_dict(state["scheduler_state_dict"])
        # IMG_PATH = state["img_path"]
        IMG_WIDTH = state["img_width"]
        IMG_HEIGHT = state["img_height"]
        CHANNELS = state["channels"]
        BATCH_SIZE = state["batch_size"]

    curr_epoch = loaded_epoch

    print("model name:", MODEL_NAME)
    print("shape:", BATCH_SIZE, CHANNELS, IMG_HEIGHT, IMG_WIDTH)

    print(f"using image: {IMG_PATH}")
    img_tensor = load_image(IMG_PATH)
    pool = Pool(target_img=img_tensor, pool_size=1024)
    target = img_tensor.repeat(BATCH_SIZE, 1, 1, 1)

    device = (
        torch.accelerator.current_accelerator().type  # type: ignore
        if torch.accelerator.is_available()
        else "cpu"
    )
    print(f"using device: {device}")
    if device != "cpu":
        print(
            "WARNING: the code doesnt use ur fancy device so ur actually just running on ur dumb cpu rn lmao"
        )

    try:
        for i in tqdm(range(EPOCHS), leave=False):
            curr_epoch = i + loaded_epoch
            optimizer.zero_grad()
            steps = torch.randint(64, 96, (1,)).item()
            model_in = pool.sample_damaged(BATCH_SIZE)
            res = cann(model_in, steps=steps)
            loss = F.mse_loss(res[:, :4], target)
            pool.update(res)
            if i % 100 == 0:
                tqdm.write(f"epoch {curr_epoch} loss: {loss.item()}")
            loss.backward()
            optimizer.step()
            scheduler.step()
    except KeyboardInterrupt:
        print("training canceled")

    state = {
        "name": MODEL_NAME,
        "epoch": curr_epoch,
        "model_state_dict": cann.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "img_path": IMG_PATH,
        "img_width": IMG_WIDTH,
        "img_height": IMG_HEIGHT,
        "channels": CHANNELS,
        "batch_size": BATCH_SIZE,
    }
    save_path = f"models/{MODEL_NAME}-{curr_epoch + 1}.tar"
    torch.save(state, save_path)
    print(f"model saved: {save_path}")
