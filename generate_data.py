from PIL import Image, ImageFont, ImageDraw
import random
import numpy as np
import os
from tqdm import trange
from src.config import IMG_SIZE, CHARACTERS, FONT_SIZE, AMOUNT, TRAIN_SPLIT, BASE_PATH

FONT = ImageFont.truetype(os.path.join(BASE_PATH, 'ocr_b.ttf'), FONT_SIZE)


# region Filters
def random_width_stretch(image: Image) -> Image:
    # Randomly stretch the image
    new_width = random.randint(IMG_SIZE // 2, IMG_SIZE * 2)
    img = image.resize((new_width, IMG_SIZE))

    if new_width > IMG_SIZE:
        # If the new width is bigger than the original, crop so that the original size is in the center.
        left = (new_width - IMG_SIZE) // 2
        top = 0
        right = (new_width + IMG_SIZE) // 2
        bottom = IMG_SIZE
        return img.crop((left, top, right, bottom))
    else:
        # Otherwise, pad the image evenly, so it will get to the original size.
        new_img = Image.new("L", (IMG_SIZE, IMG_SIZE), 255)
        new_img.paste(img, ((IMG_SIZE - new_width) // 2, 0))
        return new_img


def random_height_stretch(image: Image) -> Image:
    # Randomly stretch the image
    new_height = random.randint(IMG_SIZE // 2, IMG_SIZE * 2)
    img = image.resize((IMG_SIZE, new_height))

    if new_height > IMG_SIZE:
        # If the new height is bigger than the original, crop so that the original size is in the center.
        left = 0
        top = (new_height - IMG_SIZE) // 2
        right = IMG_SIZE
        bottom = (new_height + IMG_SIZE) // 2
        return img.crop((left, top, right, bottom))
    else:
        # Otherwise, pad the image evenly, so it will get to the original size.
        new_img = Image.new("L", (IMG_SIZE, IMG_SIZE), 255)
        new_img.paste(img, (0, (IMG_SIZE - new_height) // 2))
        return new_img


def random_perspective_transform(image: Image) -> Image:
    """
    Taken (with minor changes) from torchvision's RandomPerspective transform.
    """
    distortion_scale = 0.25

    def get_perspective_coefficients(
            start_points: tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]],
            end_points: tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]]
    ) -> list[float]:
        a_matrix = np.zeros((2 * len(start_points), 8))

        for i, (p1, p2) in enumerate(zip(end_points, start_points)):
            a_matrix[2 * i, :] = np.array([p1[0], p1[1], 1, 0, 0, 0, -p2[0] * p1[0], -p2[0] * p1[1]])
            a_matrix[2 * i + 1, :] = np.array([0, 0, 0, p1[0], p1[1], 1, -p2[1] * p1[0], -p2[1] * p1[1]])

        b_matrix = np.array(start_points).reshape(8)

        res = np.linalg.solve(a_matrix, b_matrix)
        return np.array(res).reshape(8).tolist()

    half_height = IMG_SIZE // 2
    half_width = IMG_SIZE // 2
    top_left = (
        int(random.randint(0, int(distortion_scale * half_width) + 1)),
        int(random.randint(0, int(distortion_scale * half_height) + 1)),
    )
    top_right = (
        int(random.randint(IMG_SIZE - int(distortion_scale * half_width) - 1, IMG_SIZE)),
        int(random.randint(0, int(distortion_scale * half_height) + 1)),
    )
    bottom_right = (
        int(random.randint(IMG_SIZE - int(distortion_scale * half_width) - 1, IMG_SIZE)),
        int(random.randint(IMG_SIZE - int(distortion_scale * half_height) - 1, IMG_SIZE)),
    )
    bottom_left = (
        int(random.randint(0, int(distortion_scale * half_width) + 1)),
        int(random.randint(IMG_SIZE - int(distortion_scale * half_height) - 1, IMG_SIZE)),
    )
    startpoints = ((0, 0), (IMG_SIZE - 1, 0), (IMG_SIZE - 1, IMG_SIZE - 1), (0, IMG_SIZE - 1))
    endpoints = (top_left, top_right, bottom_right, bottom_left)

    return image.transform(image.size,
                           Image.PERSPECTIVE,
                           get_perspective_coefficients(startpoints, endpoints),
                           Image.BICUBIC,
                           fillcolor=255)


def random_salt_pepper_noise(image: Image) -> Image:
    sp_probability = 0.05  # The probability that noise will be added to each pixel, either black or white
    result = np.asarray(image).copy()
    probs = np.random.random(result.shape[:2])  # Create an array in the shape of the image with random floats
    result[probs < (sp_probability / 2)] = 0
    result[probs > 1 - (sp_probability / 2)] = 255
    return Image.fromarray(result)


# endregion


def main():
    for char in CHARACTERS:
        # Create a new blank image
        img = Image.new("L", (IMG_SIZE, IMG_SIZE), 255)

        # Draw the character onto the image
        draw = ImageDraw.Draw(img)
        # Calculate the bounding box of the character, in order to center it:
        _, _, w, h = draw.textbbox((0, 0), char, font=FONT)
        # Draw the character on the image
        draw.text(((IMG_SIZE - w) / 2, (IMG_SIZE - h) / 2), char, 0, font=FONT)
        # Crop the image to the character, +2px padding on each size
        y_nonzero, x_nonzero = np.nonzero(np.invert(img))
        tmp = img.crop((
            max(np.min(x_nonzero) - 2, 0), max(np.min(y_nonzero) - 2, 0),
            min(np.max(x_nonzero) + 2, img.size[0]), min(np.max(y_nonzero) + 2, img.size[1])
        ))

        # Resize the image to the original image size
        img = tmp.resize((IMG_SIZE, IMG_SIZE))

        # Set up the folders for the train/test split
        train_dir = os.path.join(BASE_PATH, 'train', char if char != "<" else "arrow", '')
        test_dir = os.path.join(BASE_PATH, 'test', char if char != "<" else "arrow", '')
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)
        img.save(os.path.join(train_dir, '0.png'))

        for i in trange(1, AMOUNT, desc=f"Generating images for {char}", initial=1, total=AMOUNT, unit="images"):
            image = img.copy()  # Create a copy of the original character image, to augment.

            # Randomly augment the image: 70% chance a certain filter will be added
            activation_probability = 0.7
            image = random_salt_pepper_noise(image) if random.random() < activation_probability else image
            image = random_perspective_transform(image) if random.random() < activation_probability else image
            image = random_width_stretch(image) if random.random() < activation_probability else image
            image = random_height_stretch(image) if random.random() < activation_probability else image

            # Save the image
            output_dir = train_dir if i < TRAIN_SPLIT * AMOUNT else test_dir
            image.save(os.path.join(output_dir, f'{i}.png'))


if __name__ == '__main__':
    main()
