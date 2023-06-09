import numpy as np
import cv2
# Types
from typing import Literal
from ..utils import Array, BoundingBox, show_images


class Segmentor:
    """
    An abstract segmentor class.
    """

    def __init__(self, draw_contours: bool = False, draw_debug: bool = False):
        self._draw_contours: bool = draw_contours  # Whether an image with the found contours should be shown.
        self._draw_debug: bool = draw_debug  # Whether images of the segmentation process should be shown.
        # The kernel for the morphological transformations
        self._rect_kernel: Array = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 2))
        self._image: Array | None = None  # Will save the original image
        self._thresh: Array | None = None  # Will save the thresholded binary image
        self._contours: Array | None = None  # Will save the found contours

    def segment(self, image: Array) -> list[tuple[Array, BoundingBox]]:
        """
        The main segmentation pipeline.
        transformations -> finding contours -> sorting contours -> selecting contours
        :param image: The image to segment
        :return: a list of all contours, in the form (image, bounding_box)
        """
        self._image = image  # Keep the original image on the instance

        self._transformations()  # Apply transformations

        self._post_transformations()  # Apply optional transformations, that each inheriting segmentor should override.

        if self._draw_debug:
            # Draw the thresholded image after all the transformations.
            with show_images():
                cv2.imshow("thresh after all transformations", self._thresh)

        self._find_contours()
        self._sort_contours()

        return self._select_contours()

    def _select_contours(self):
        """
        Select all contours matching the criteria described in is_roi
        """
        results: list[tuple[Array, BoundingBox]] = []  # An array to keep the selected contours

        for contour in self._contours:
            bounding_rect: BoundingBox = cv2.boundingRect(contour)  # Get the bounding rectangle of the contour
            if self._is_roi(bounding_rect):
                (x, y, w, h) = bounding_rect  # Extract the components of the bounding box for the ROI extraction

                # extract the ROI (region of interest) from the image and draw a bounding box
                # surrounding its location on the original image
                roi = self._image[y:y + h, x:x + w].copy()  # https://stackoverflow.com/a/9085008
                results.append((roi, (x, y, w, h)))

                if self._draw_contours:
                    # Draw the surrounding box of the contour on the original image
                    cv2.rectangle(self._image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        if self._draw_contours:
            with show_images():
                cv2.imshow("Segmented Image", self._image)

        return results

    def _transformations(self):
        """"
        Apply transformation to the image to get a binary image with clear separation, for the contour finding process
        to work well.
        """
        transformed: Array  # Will keep the transformed image throughout the transformation process
        transformed = cv2.cvtColor(self._image, cv2.COLOR_BGR2GRAY)  # convert the image to grayscale
        transformed = cv2.GaussianBlur(transformed, (3, 3), 0)  # blur the image using a 3x3 Gaussian blur

        # apply the blackhat morphological operator to find dark regions on a light background
        transformed: Array = cv2.morphologyEx(transformed, cv2.MORPH_BLACKHAT, self._rect_kernel)
        if self._draw_debug:
            with show_images():
                cv2.imshow("blackhat", transformed)

        # compute the Scharr gradient of the blackhat image
        transformed: Array = cv2.Laplacian(transformed, ddepth=cv2.CV_32F)
        transformed = np.absolute(transformed)
        (minVal, maxVal) = (np.min(transformed), np.max(transformed))

        # scale the result into the range [0, 255]
        transformed = (255 * ((transformed - minVal) / (maxVal - minVal))).astype("uint8")

        if self._draw_debug:
            with show_images():
                cv2.imshow("Scharr", transformed)

        # apply a closing operation using the rectangular kernel to close gaps in between letters
        transformed = cv2.morphologyEx(transformed, cv2.MORPH_CLOSE, self._rect_kernel)
        if self._draw_debug:
            with show_images():
                cv2.imshow("Scharr after closing", transformed)

        # apply Otsu's thresholding method
        self._thresh = cv2.threshold(transformed, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        if self._draw_debug:
            with show_images():
                cv2.imshow("threshold", self._thresh)

    def _post_transformations(self):
        """
        Add more transformations to do after the basic ones.
        Can be overriden by inheriting classes if they require more transformations.
        """
        pass

    def _find_contours(self):
        """
        find the contours in the thresholded image
        :return:
        """
        self._contours, _ = cv2.findContours(self._thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    def _sort_contours(self):
        """
        Sort the contours (will be overriden by inheriting classes).

        By default, do not sort at all.
        """
        self._contours = sort_contours(self._contours)

    def _is_roi(self, bounding_box: BoundingBox) -> bool:
        """
        Determine whether the contour should be treated as a ROI.
        Will be overriden by inheriting classes.

        By default, all contours are ROI.
        :param bounding_box: The bounding box of the ROI
        :return: True if the contour should be a ROI, False otherwise.
        """
        return True

    @staticmethod
    def resize_roi(bounding_box: BoundingBox, resize_ratio: float) -> BoundingBox:
        """
        Resize a ROI bounding box to the size of the image before it was passed to the segmentor.
        :param bounding_box: The bounding box of the ROI
        :param resize_ratio: The resize ratio of the segmented image
        :return: the bounding **lines** of the image: (x_left, x_right, y_top, y_bottom)
        """
        (x1, y1, w1, h1) = bounding_box

        # resize the ROI back to the original image size:
        x_left = int(x1 // resize_ratio)
        x_right = int((x1 + w1) // resize_ratio)
        y_top = int(y1 // resize_ratio)
        y_bottom = int((y1 + h1) // resize_ratio)

        return x_left, x_right, y_top, y_bottom


def sort_contours(
        contours: tuple[Array, ...],
        direction: Literal["ttb", "btt", "ltr", "rtl", "area"] | None = None
) -> tuple[Array, ...]:
    """
    Multiple ways of sorting a tuple of contours.
    Contours can be sorted in the following ways:

    - "ttb" (top to bottom)
    - "btt" (bottom to top)
    - "ltr" (left to right)
    - "rtl" (right to left)
    - "area"

    Taken and modified from https://pyimagesearch.com/2015/04/20/sorting-contours-using-python-and-opencv/
    I shortened the explanation, it is much more indepth on the website.

    :param contours: The tuple to sort
    :param direction: The key for the sorting. Can be one of the following:
    :return: The sorted tuple
    """
    if direction == 'area':
        return tuple(sorted(contours, key=cv2.contourArea, reverse=True))
    if direction in ['ttb', 'btt', 'ltr', 'rtl']:
        # Create a list containing all the bounding rectangles of the contours
        bounding_boxes: list[BoundingBox] = [cv2.boundingRect(contour) for contour in contours]
        # The box looks like (x, y, w, h).

        # To sort on the y-axis, we need it to be 1. for the x-axis, it needs to be 0.
        axis = 1 if direction in ['btt', 'ttb'] else 0

        # The x-axis goes left-to-right, the y-axis goes top-to-bottom. This means the default
        # order will be either ltr to ttb. We need to reverse it if we want rtl or btt
        reverse = True if direction in ['btt', 'rtl'] else False

        # Sort the contours:
        sorted_contours: tuple[Array, ...]  # The result tuple

        sorted_contours, _ = zip(  # Convert the object back to the original shape:
            # (
            #   (contour_1, contour_2, ..., contour_n),
            #   (bounding_box_1, bounding_box_2, ..., bounding_box_n)
            # )
            *sorted(  # sort the object
                zip(  # Combine the contours and bounding boxes to an object shaped like:
                    # (
                    #     (contour_1, bounding_box_1),
                    #     (contour_2, bounding_box_2),
                    #     ...,
                    #     (contour_n, bounding_box_n)
                    # )
                    contours, bounding_boxes
                ),
                key=lambda contour: contour[1][axis],  # Sort by the axis on the relevant bounding box (second element),
                reverse=reverse  # Reverse the sort, if needed
            )
        )

        # return the list of sorted contours and bounding boxes
        return sorted_contours

    # No sorting method selected; return the contours
    return contours
