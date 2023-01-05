from segmentation import MRZSegmentor, LineSegmentor, CharacterSegmentor
from utils import get_images_by_path


def main():
    images = get_images_by_path('./images/')
    mrz_segmentor = MRZSegmentor()
    line_segmentor = LineSegmentor()
    char_segmentor = CharacterSegmentor(True)
    for image in images:
        mrz = mrz_segmentor.segment(image)
        if mrz is not None:
            lines = line_segmentor.segment(mrz)
            if len(lines) > 0:
                for line in lines:
                    char_segmentor.segment(line)


if __name__ == '__main__':
    main()
