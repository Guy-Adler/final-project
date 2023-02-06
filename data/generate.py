import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageFont, ImageDraw


def sp_noise(img, prob):
    '''
    Add salt and pepper noise to image
    img: the image
    prob: Probability of the noise
    '''
    output = img.copy()
    if len(output.shape) == 2:
        black = 0
        white = 255
    else:
        colorspace = output.shape[2]
        if colorspace == 3:  # RGB
            black = np.array([0, 0, 0], dtype='uint8')
            white = np.array([255, 255, 255], dtype='uint8')
        else:  # RGBA
            black = np.array([0, 0, 0, 255], dtype='uint8')
            white = np.array([255, 255, 255, 255], dtype='uint8')
    probs = np.random.random(output.shape[:2])
    output[probs < (prob / 2)] = black
    output[probs > 1 - (prob / 2)] = white
    return output


if __name__ == '__main__':

    chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<"  # Valid characters
    chars = [c for c in chars]
    # Initialize a dataframe:
    df = pd.DataFrame(columns=['label'])

    font = ImageFont.truetype('./ocrb.ttf', 32)  # Load the font
    IMAGE_SIZE = (32, 32)
    W, H = IMAGE_SIZE
    for char in chars:
        print(f"Generating images for character {char}")
        # Create a completely white image:
        img = np.ones((32, 32))
        img.fill(255)
        # Use PIL instead of opencv to be able to use a custom font:
        img = Image.fromarray(np.uint8(img))
        draw = ImageDraw.Draw(img)
        # Calculate the bounding box of the character, in order to center it:
        _, _, w, h = draw.textbbox((0, 0), char, font=font)
        # Draw the character on the image
        draw.text(((W-w)/2, (H-h)/2), char, 0, font=font)
        # Convert the image to numpy, for using opencv
        image = np.asarray(img)  # noqa this is the way to do it according to the docs

        # Generate 4 images and add to the dataframe: one pure, one with distortion, one with noise and one with both.
        # Save the pure image to a dataset:
        df = pd.concat([df, pd.DataFrame(image.reshape((1, -1)))], ignore_index=True)

        # Distort the image:
        distorted_image = np.zeros((W, H))
        A = W / 10.0
        w = 0.5 / H
        shift = lambda x: A * np.sin(2.0 * np.pi * x * w)
        for i in range(W):
            distorted_image[:, i] = np.roll(image[:, i], int(shift(i)))

        # To create enough examples, and since the noise is random, create multiple images of it:
        for _ in range(1500):
            # Add noise:
            noise_image = sp_noise(image, 0.05)
            df = pd.concat([df, pd.DataFrame(noise_image.reshape((1, -1)))], ignore_index=True)
            # Distorted with noise:
            distorted_noise_image = sp_noise(distorted_image, 0.05)
            df = pd.concat([df, pd.DataFrame(distorted_noise_image.reshape((1, -1)))], ignore_index=True)

        # Add the label (since we just added the row, the label will be NaN. This means we can just replace all NaNs)
        df['label'].fillna(char, inplace=True)

    # Export the dataset
    df.to_csv('characters.csv', index=False)
