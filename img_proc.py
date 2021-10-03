#!/usr/bin/python3

import cv2 as cv

def process_img(path):
    img = cv.imread(path, flags=cv.IMREAD_COLOR)
    # img = cv.GaussianBlur(img.copy(), (9, 9), 0)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    process = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 199, 5)
    process = cv.bitwise_not(process, process)

    cv.imshow("image", process)
    k = cv.waitKey(0) & 0xFF
    if k == 27:
        cv.destroyAllWindows()


if __name__ == "__main__":
    path = "./doc/sample.jpg"
    process_img(path)