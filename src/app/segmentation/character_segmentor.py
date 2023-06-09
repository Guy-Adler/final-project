import cv2
from .segmentor import Segmentor, Array, sort_contours
from ..utils import resize_image, show_images
from src.config import CHARACTERS_AMOUNT


class CharacterSegmentor(Segmentor):
    """
    A segmentor for getting the characters from a picture of a line
    """

    def __init__(self, draw_contours: bool = False, draw_debug: bool = False):
        super().__init__(draw_contours, draw_debug)

    def segment(self, image) -> list[Array]:
        """
        The segmentation for characters.
        :param image: The image to segment
        :return: a list of all characters
        """
        # Resize the image to width of 50px
        resize_height = 50
        resize_ratio = resize_height / image.shape[0]
        resized_image = resize_image(image, height=resize_height)

        # Call the base segmentor
        rois = super().segment(resized_image)

        # Resize the ROIs back to the original image size
        original_rois: list[Array] = []  # to keep the resized ROIs

        for roi in rois:
            # Get the bounding *line* of the ROI for the original image size:
            x_left, x_right, y_top, y_bottom = self.resize_roi(roi[1], resize_ratio)

            # Make sure only rois with a size after the refitting are included:
            # Because of the resize, sometimes it is possible for the ROI to be
            # less than 1px in one of the axis (which is not possible for an image).
            # To make sure only real images (=have 2 dimensions) are selected:
            if (x_right - x_left) * (y_bottom - y_top) > 0:
                # Pad the ROI with 2px on each side (while making sure to stay within the original image borders):
                padding_amount = 2

                x_left = max(x_left - padding_amount, 0)
                y_top = max(y_top - padding_amount, 0)
                x_right = min(x_right + padding_amount, image.shape[1])
                y_bottom = min(y_bottom + padding_amount, image.shape[0])

                # Crop the original image to the ROI, and save it to the array
                original_rois.append(image[y_top:y_bottom, x_left:x_right].copy())

        return original_rois

    def _transformations(self):
        """
        The character segmentor requires different transformations than the base segmentor.
        """
        transformed: Array = cv2.cvtColor(self._image, cv2.COLOR_BGR2GRAY)  # convert the image to grayscale

        transformed = cv2.GaussianBlur(transformed, (3, 3), 0)  # blur the image using a 3x3 Gaussian blur

        # apply Otsu's thresholding method
        self._thresh = cv2.threshold(transformed, 255 // 2, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        if self._draw_debug:
            with show_images():
                cv2.imshow("threshold", self._thresh)

        # cv2.findContours expects black background and white foreground. Therefore, the threshold mask needs
        # to be inverted (unlike in the base segmentor, where the blackhat does the inversion)
        self._thresh = 255 - self._thresh

        if self._draw_debug:
            with show_images():
                cv2.imshow("threshold after inversion", self._thresh)

    def _sort_contours(self):
        """
        Sort the contours by area,
        get the 44 largest contours (amount of characters in a line),
        and sort them from left to right (the reading order).
        """
        contours = sort_contours(self._contours, 'area')  # Sort the contours by area:
        # Get the 44 largest ones (there are exactly 44 characters in an MRZ line)
        contours = contours[:CHARACTERS_AMOUNT]
        self._contours = sort_contours(contours, 'ltr')  # Sort the contours from leftmost to rightmost
