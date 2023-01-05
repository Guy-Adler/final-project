import cv2
from .Segmentor import Segmentor, Array, BoundingBox, sort_contours
from utils import resize_image


class CharacterSegmentor(Segmentor):
    def __init__(self, draw_contours: bool = False):
        super().__init__(draw_contours)
        self.rect_kernel: Array = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    def segment(self, image):
        # Resize the image
        resized_image = resize_image(image, height=50)

        # Call the general segmentor
        return super().segment(resized_image)

    def sort_contours(self):
        # Sort the contours by area:
        contours = sort_contours(self.contours, 'area')
        # Get the 44 largest ones (there are exactly 44 characters in an MRZ line)
        contours = contours[:44]
        # Sort the contours from leftmost to rightmost
        self.contours = sort_contours(contours, 'ltr')
