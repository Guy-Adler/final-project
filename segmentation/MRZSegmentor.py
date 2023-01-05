import cv2
from .LineSegmentor import LineSegmentor, Array, BoundingBox, sort_contours
from utils import resize_image


class MRZSegmentor(LineSegmentor):
    def __init__(self, draw_contours: bool = False):
        super().__init__(draw_contours)
        self.square_kernel: Array = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
        self.found_roi: bool = False

    def segment(self, image) -> Array | None:
        # reset the initial variables:
        self.found_roi = False

        # Resize the image
        resized_image = resize_image(image, height=600)

        # Call the general segmentor
        rois = super().segment(resized_image)

        # rois is a list, with at most 1 element.
        if len(rois) == 0:
            # Mo ROI was found
            return None

        return rois[0]

    def post_transformations(self):
        # perform another closing operation, this time using the square kernel to close gaps between lines of the MRZ
        self.thresh = cv2.morphologyEx(self.thresh, cv2.MORPH_CLOSE, self.square_kernel)
        # perform a series of erosions to break apart connected components
        self.thresh = cv2.erode(self.thresh, None, iterations=4)
        super().post_transformations()

    def sort_contours(self):
        self.contours = sort_contours(self.contours, 'area')

    def is_roi(self, bounding_rect: BoundingBox):
        # Limit to only 1 ROI per image
        if self.found_roi:
            return False

        is_roi = super().is_roi(bounding_rect)
        if is_roi:
            self.found_roi = True
            return True

        return False
