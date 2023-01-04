from image_processing import get_image_by_path, get_mrz, get_mrz_lines, get_chars, get_images_by_path
from utils import show_image


def main():
    images = get_images_by_path('./images/')
    for i, image in enumerate(images):
        mrz = get_mrz(image)
        if len(mrz) != 0:
            lines = get_mrz_lines(mrz)
            if len(lines) == 2:
                get_chars(lines[0], True)
                get_chars(lines[1], True)


if __name__ == '__main__':
    main()
