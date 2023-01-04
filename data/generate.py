import cv2
import numpy as np
from PIL import Image, ImageFont, ImageDraw


chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<"
chars = [c for c in chars]

font = ImageFont.truetype('./ocrb.ttf', 32)

for char in [chars[-1]]:
    img = np.ones((32, 32, 3))
    img.fill(255)
    img = Image.fromarray(np.uint8(img))
    # cv2.putText(img, text=char, org=(3, 21), fontFace=font, fontScale=1, color=(0, 0, 0), thickness=1,
    #             lineType=cv2.LINE_AA)
    draw = ImageDraw.Draw(img)
    draw.text((3, 0), char, (0, 0, 0), font=font)
    img.resize((32, 32))
    img.save(f'./characters/BRACKET.jpg')
