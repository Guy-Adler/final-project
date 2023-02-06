import cv2
from .Segmentor import Segmentor, Array, BoundingBox, sort_contours


class LineSegmentor(Segmentor):
    def __init__(self, draw_contours: bool = False):
        super().__init__(draw_contours)

    def post_transformations(self):
        self.remove_border_pixels()

    def sort_contours(self):
        # Sort the contours by area:
        contours = sort_contours(self.contours, 'area')
        # Get the 2 largest ones (there are exactly 2 lines in an MRZ)
        contours = contours[:2]
        # Sort the contours from topmost to bottommost
        self.contours = sort_contours(contours, 'ttb')

    def is_roi(self, bounding_rect: BoundingBox):
        # Calculate the aspect ratio and coverage ratio of the ROI
        *_, w, h = bounding_rect
        aspect_ratio = w / float(h)
        coverage_ratio = w / float(self.image.shape[1])

        return aspect_ratio > 5 and coverage_ratio > 0.75

    def change_roi_rect(self, bounding_rect: BoundingBox):
        x, y, w, h = bounding_rect
        # because we applied erosions in the post transformations, we now need to re-grow the bounding box
        p_x = int((x + w) * 0.03)
        p_y = int((y + h) * 0.04)
        x, y = (x - p_x, y - p_y)
        w, h = (w + (p_x * 2), h + (p_y * 2))

        return x, y, w, h
