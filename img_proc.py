#!/usr/bin/python3

import operator
import numpy as np
import cv2 as cv


def process_img(path):
    img = cv.imread(path, flags=cv.IMREAD_COLOR)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (5,5), 0)
    thresh = cv.adaptiveThreshold(blur, 255, 1, 1, 11, 2)
    return img, thresh


def get_corners(contour):
    bottom_right, _ = max(enumerate([pt[0][0] + pt[0][1] for pt in contour]), key=operator.itemgetter(1))
    top_right, _ = max(enumerate([pt[0][0] - pt[0][1] for pt in contour]), key=operator.itemgetter(1))
    top_left, _ = min(enumerate([pt[0][0] + pt[0][1] for pt in contour]), key=operator.itemgetter(1))
    bottom_left, _ = min(enumerate([pt[0][0] - pt[0][1] for pt in contour]), key=operator.itemgetter(1))

    return contour[top_left][0], contour[top_right][0], contour[bottom_right][0], contour[bottom_left][0]



def crop_contour(img, thresh):
    contours, _ = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv.contourArea, reverse=True)
    top_left, top_right, bottom_right, bottom_left = get_corners(contours[0])
    src = np.array([top_left, top_right, bottom_right, bottom_left], dtype="float32")
    side = max([
        np.linalg.norm(bottom_right - top_right),
        np.linalg.norm(top_left - bottom_left),
        np.linalg.norm(bottom_right - bottom_left),
        np.linalg.norm(top_left - top_right)
    ])

    dst = np.array([[0, 0], [side - 1, 0], [side - 1, side - 1], [0, side - 1]], dtype="float32")
    m = cv.getPerspectiveTransform(src, dst)
    cropped = cv.warpPerspective(img, m, (int(side), int(side)))

    return cropped


def main():
    path = "./doc/sample.jpg"
    img, thresh = process_img(path)
    cropped = crop_contour(img, thresh)

    cv.imshow("image", cropped)
    k = cv.waitKey(0) & 0xFF
    if k == 27:
        cv.destroyAllWindows()



if __name__ == "__main__":
    main()