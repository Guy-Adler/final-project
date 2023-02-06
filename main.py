from segmentation import MRZSegmentor, LineSegmentor, CharacterSegmentor
from utils import get_images_by_path


def main():
    images = get_images_by_path('images')
    mrz_segmentor = MRZSegmentor(True)
    line_segmentor = LineSegmentor(True)
    char_segmentor = CharacterSegmentor(True)
    for i, image in enumerate(images):
        mrz = mrz_segmentor.segment(image)
        if mrz is None:
            continue
        lines = line_segmentor.segment(mrz)
        if len(lines) != 2:
            # print(i + 1)
            continue
        for line in lines:
            chars = char_segmentor.segment(line)


if __name__ == '__main__':
    main()
