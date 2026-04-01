import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# 1. Pre-load font once (Global or Class level)
FONT_PATH = "usr/share/fonts/opentype/baby.otf"
FONT_SIZE = 8
font = ImageFont.truetype(FONT_PATH, FONT_SIZE)


def render_text(text, font):
    # bbox (left, top, right, bottom)
    bbox = font.getbbox(text)
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]

    img = Image.new("L", (width, height), color=255)
    draw = ImageDraw.Draw(img)

    # -bbox[0], -bbox[1] offsets any font internal padding
    draw.text((-bbox[0], -bbox[1]), text, fill=0, font=font)

    return np.asarray(img)


img = render_text(
    "There has been considerable progress during the year on a number of aspects related to the cost competitiveness of the UK's oil industry, progress in which the Wood Group has played a leading role in national discussions and, more importantly, in a wide range of initiatives and developments directly with our customers.",
    font,
)

print(img.shape)

cv2.namedWindow("hi", cv2.WINDOW_NORMAL)
# cv2.setWindowProperty("hi", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
# cv2.setWindowProperty("hi", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
cv2.imshow("hi", img)
while True:
    if cv2.waitKey(int(1000 / 60)) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
