from .LineSegmentor import LineSegmentor
from .MRZSegmentor import MRZSegmentor
from .CharacterSegmentor import CharacterSegmentor
from utils import Array


def segment_characters_of_mrz(image: Array) -> tuple[tuple[Array, ...], tuple[Array, ...]] | None:
    """
    Pipeline to segment image to characters.
    :param image: The image to get characters of
    :return: A tuple, containing two tuples representing the lines with a length of length 44.
    """
    # Initialize the segmentors
    mrz_segmentor = MRZSegmentor()
    line_segmentor = LineSegmentor()
    char_segmentor = CharacterSegmentor()

    mrz = mrz_segmentor.segment(image)
    if mrz is not None:  # If an MRZ was found
        lines = line_segmentor.segment(mrz)
        if len(lines) == 2:  # Make sure exactly two lines were found
            chars_l1 = char_segmentor.segment(lines[0])
            chars_l2 = char_segmentor.segment(lines[1])
            if len(chars_l1) == 44 == len(chars_l2):
                # Each line needs to have exactly 44 characters for the segmentation to be considered successful.
                return (
                    tuple(chars_l1),
                    tuple(chars_l2)
                )

    return None
