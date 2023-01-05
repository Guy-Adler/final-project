import numpy as np
import cv2
# Types
from numpy import uint8
import numpy.typing as npt
from typing import Literal
from utils import Array, BoundingBox, show_images


class Segmentor:
    def __init__(self, draw_contours: bool = False):
        self.draw_contours: bool = draw_contours
        self.rect_kernel: Array = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
        self.image: Array | None = None
        self.thresh: Array | None = None
        self.contours: Array | None = None

    def segment(self, image: Array) -> list[Array]:
        self.image = image
        self.transformations()

        self.post_transformations()

        self.find_contours()
        self.sort_contours()

        results: list[Array] = []

        for contour in self.contours:
            bounding_rect: BoundingBox = cv2.boundingRect(contour)
            if self.is_roi(bounding_rect):
                (x, y, w, h) = self.change_roi_rect(bounding_rect)
                # extract the ROI from the image and draw a bounding box
                # surrounding its location on the original image
                roi = image[y:y + h, x:x + w].copy()
                results.append(roi)
                if self.draw_contours:
                    # Draw the surrounding box of the contour on the original image
                    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        if self.draw_contours:
            with show_images():
                cv2.imshow("Image", image)

        return results

    def transformations(self):
        # convert it to grayscale
        gray: Array = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        # smooth the image using a 3x3 Gaussian
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        # apply the blackhat morphological operator to find dark regions on a light background
        blackhat: Array = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, self.rect_kernel)

        # compute the Scharr gradient of the blackhat image and scale the
        gradX: Array = cv2.Laplacian(blackhat, ddepth=cv2.CV_32F)
        gradX = np.absolute(gradX)
        (minVal, maxVal) = (np.min(gradX), np.max(gradX))
        # result into the range [0, 255]
        gradX = (255 * ((gradX - minVal) / (maxVal - minVal))).astype("uint8")

        # apply a closing operation using the rectangular kernel to close gaps in between letters
        gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, self.rect_kernel)

        # apply Otsu's thresholding method
        self.thresh = cv2.threshold(gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    def post_transformations(self):
        """
        Add more transformations to do after the basic ones here
        """
        pass

    def remove_border_pixels(self):
        # during thresholding, it's possible that border pixels were
        # included in the thresholding, so let's set 5% of the left and
        # right borders to zero
        width = self.image.shape[1]
        p = int(width * 0.05)
        self.thresh[:, 0:p] = 0
        self.thresh[:, width - p:] = 0

    def find_contours(self):
        # find contours in the thresholded image
        # Assumes using opencv 4
        self.contours, _ = cv2.findContours(self.thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    def sort_contours(self):
        # Sort the contours based on the desired method. By default, do not sort at all.
        self.contours = sort_contours(self.contours)

    def is_roi(self, bounding_box: BoundingBox) -> bool:  # noqa
        # Decide whether the contour should be treated as a ROI. By default, all contours are ROI.
        return True

    def change_roi_rect(  # noqa This method needs to be able to be overridden
            self,
            rect: BoundingBox
    ) -> BoundingBox:
        # change the bounding rect of the ROI. By default, leave it as-is.
        return rect


def sort_contours(
        contours: tuple[Array, ...],
        direction: Literal["ttb", "btt", "ltr", "rtl", "area"] | None = None
) -> tuple[Array, ...]:
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
        sorted_contours: tuple[Array, ...]

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
                key=lambda b: b[1][axis],  # Sort by the axis on the relevant bounding box (second element),
                reverse=reverse  # Reverse the sort, if needed
            )
        )

        # return the list of sorted contours and bounding boxes
        return sorted_contours

    # No sorting method selected; return the contours
    return contours
