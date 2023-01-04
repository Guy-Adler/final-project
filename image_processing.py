from imutils import paths
import numpy as np
import imutils
import cv2

from utils import show_image


def sort_contours_top_to_bottom(cnts):
    # construct the list of bounding boxes
    bounding_boxes = [cv2.boundingRect(c) for c in cnts]
    # sort the list from the bottom to the top
    (sorted_cnts, bounding_boxes) = zip(*sorted(zip(cnts, bounding_boxes),
                                                key=lambda b: b[1][1]))
    # return the list of sorted contours and bounding boxes
    return sorted_cnts


def sort_contours_left_to_right(cnts):
    # construct the list of bounding boxes
    bounding_boxes = [cv2.boundingRect(c) for c in cnts]
    # sort the list from the bottom to the top
    (sorted_cnts, bounding_boxes) = zip(*sorted(zip(cnts, bounding_boxes),
                                                key=lambda b: b[1][0]))
    # return the list of sorted contours and bounding boxes
    return sorted_cnts


def find_roi(img, resize: bool = False, apply_sq_kernel: bool = False, sort: str = "top_bottom",
             extract_more_than_one: bool = True, draw_contours: bool = False):

    # initialize a rectangular and square structuring kernel
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
    sq_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    # initiate results list
    results = []

    # load the image
    image = img.copy()
    # resize it
    if resize:
        image = imutils.resize(image, height=600)
    # convert it to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # smooth the image using a 3x3 Gaussian
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    # apply the blackhat morphological operator to find dark regions on a light background
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rect_kernel)

    # compute the Scharr gradient of the blackhat image and scale the
    # result into the range [0, 255]
    gradX = cv2.Laplacian(blackhat, ddepth=cv2.CV_32F)
    gradX = np.absolute(gradX)
    (minVal, maxVal) = (np.min(gradX), np.max(gradX))
    gradX = (255 * ((gradX - minVal) / (maxVal - minVal))).astype("uint8")

    # apply a closing operation using the rectangular kernel to close gaps in between letters
    gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rect_kernel)
    # apply Otsu's thresholding method
    thresh = cv2.threshold(gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    if apply_sq_kernel:
        # perform another closing operation, this time using the square kernel to close gaps between lines of the MRZ
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sq_kernel)
        # perform a series of erosions to break apart connected components
        thresh = cv2.erode(thresh, None, iterations=4)

    if draw_contours:
        cv2.imshow("Image", thresh)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # during thresholding, it's possible that border pixels were
    # included in the thresholding, so let's set 5% of the left and
    # right borders to zero
    p = int(image.shape[1] * 0.05)
    thresh[:, 0:p] = 0
    thresh[:, image.shape[1] - p:] = 0

    # find contours in the thresholded image and sort them by their
    # size
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    if sort == 'area':
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    else:
        cnts = sort_contours_top_to_bottom(cnts)

    # loop over the contours
    for c in cnts:
        # compute the bounding box of the contour and use the contour to
        # compute the aspect ratio and coverage ratio of the bounding box
        # width to the width of the image
        (x, y, w, h) = cv2.boundingRect(c)
        ar = w / float(h)
        crWidth = w / float(gray.shape[1])
        # check to see if the aspect ratio and coverage width are within
        # acceptable criteria
        if ar > 5 and crWidth > 0.75:
            # pad the bounding box since we applied erosions and now need
            # to re-grow it
            pX = int((x + w) * 0.03)
            pY = int((y + h) * 0.04)
            (x, y) = (x - pX, y - pY)
            (w, h) = (w + (pX * 2), h + (pY * 2))
            # extract the ROI from the image and draw a bounding box
            # surrounding the MRZ
            roi = image[y:y + h, x:x + w].copy()
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            if draw_contours and not extract_more_than_one:
                cv2.imshow("Image", image)
                cv2.imshow("ROI", roi)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            if not extract_more_than_one:
                return roi
            results.append(roi)

    if draw_contours:
        cv2.imshow("Image", image)
        for i, roi in enumerate(results):
            cv2.imshow(f"ROI {i}", roi)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return results


def get_mrz_lines(mrz, draw_contours: bool = False):
    return find_roi(mrz, draw_contours=draw_contours)


def get_mrz(img, draw_contours: bool = False):
    return find_roi(img, resize=True, apply_sq_kernel=True, sort="area", extract_more_than_one=False,
                    draw_contours=draw_contours)


def get_images_by_path(path):
    return [cv2.imread(imagePath) for imagePath in paths.list_images(path)]


def get_image_by_path(path):
    return cv2.imread(path)


def get_chars(row, draw_contours: bool = False):
    # initialize a rectangular and square structuring kernel
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    # initiate results list
    results = []

    # load the image
    image = row.copy()
    # resize
    image = imutils.resize(image, height=50)
    # convert it to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # smooth the image using a 3x3 Gaussian
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    # apply the blackhat morphological operator to find dark regions on a light background
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rect_kernel)

    # compute the Scharr gradient of the blackhat image and scale the
    # result into the range [0, 255]
    gradX = cv2.Laplacian(blackhat, ddepth=cv2.CV_32F)
    gradX = np.absolute(gradX)
    (minVal, maxVal) = (np.min(gradX), np.max(gradX))
    gradX = (255 * ((gradX - minVal) / (maxVal - minVal))).astype("uint8")
    # apply a closing operation using the rectangular kernel to close gaps in between letters
    gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rect_kernel)
    # apply Otsu's thresholding method
    thresh = cv2.threshold(gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    # during thresholding, it's possible that border pixels were
    # included in the thresholding, so let's set 5% of the left and
    # right borders to zero
    # p = int(image.shape[1] * 0.05)
    # thresh[:, 0:p] = 0
    # thresh[:, image.shape[1] - p:] = 0

    # find contours in the thresholded image and sort them by their
    # size
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    # Sort the contours by area:
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    # Get the 44 largest ones (there are exactly 44 characters in a line)
    cnts = cnts[:44]
    # Sort the contours by position (left to right)
    cnts = sort_contours_left_to_right(cnts)

    # show_image(thresh)
    # return
    # loop over the contours
    for c in cnts:
        # compute the bounding box of the contour and use the contour to
        # compute the aspect ratio and coverage ratio of the bounding box
        # width to the width of the image
        (x, y, w, h) = cv2.boundingRect(c)
        ar = w / float(h)
        area = w * h

        roi = image[y:y + h, x:x + w].copy()
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        results.append(roi)

    if draw_contours:
        cv2.imshow("Characters", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return results
