import sys
import cv2
import numpy as np
import torch
from tqdm import tqdm

from main import (
    CANN,
    get_seed,
    nnout_to_img,
    alpha_blend_img,
    IMG_WIDTH,
    IMG_HEIGHT,
)

MODEL_NAME = "thegrassisgreen_full-20000"
MODEL_PATH = f"models/{MODEL_NAME}.tar"
ANIM_PREFIX = MODEL_NAME

print(f"using model: {MODEL_PATH}")

cann = CANN()
state = torch.load(MODEL_PATH, weights_only=True)
cann.load_state_dict(state["model_state_dict"])

print("rendering animation...")

frames = []
x = get_seed()
for i in tqdm(range(1000), leave=False):
    if i > 250 and i % 500 == 0:
        x[:, :, :, (IMG_WIDTH // 2) + 1 :] = 0
    if i > 250 and (i + 250) % 500 == 0:
        x[:, :, (IMG_HEIGHT // 2) + 1 :, :] = 0
    with torch.no_grad():
        x = cann.step(x)
    frames.append(alpha_blend_img(nnout_to_img(x)))

print("saving animation...")
for i in tqdm(range(len(frames)), leave=False):
    frame = frames[i]
    j = str(i).zfill(len(str(len(frames) - 1)))
    cv2.imwrite(f"anim/{ANIM_PREFIX}-{j}.png", (frame * 255.0).astype(np.uint8))

print("displaying animation...")

quit = False
window = "mycells"
cv2.namedWindow(window, cv2.WINDOW_NORMAL)
while not quit:
    for i in tqdm(range(len(frames)), leave=False):
        frame = frames[i]
        cv2.imshow(window, frame)
        if cv2.waitKey(int(1000 / 60)) & 0xFF == ord("q"):
            quit = True
            break

cv2.destroyAllWindows()
