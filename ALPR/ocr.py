'''
https://www.geeksforgeeks.org/text-detection-and-extraction-using-opencv-and-ocr/ original inspiration
https://pyimagesearch.com/2021/02/22/opencv-connected-component-labeling-and-analysis/ for component analysis
https://stackoverflow.com/questions/72381645/python-tesseract-license-plate-recognition for easyocr
'''
import cv2
import numpy as np
import os
from datetime import datetime


def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def otsuthreshold(img):
    return cv2.threshold(img, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)


def erode(img, iter):
    return cv2.erode(img, None, iterations=iter)


def dilate(img, iter):
    return cv2.dilate(img, None, iterations=iter)


def examineComponents(TotalComponents, labels, stats, centroids, img, thresh):
    mask = np.zeros(thresh.shape, dtype="uint8")
    character_dimensions = (0.35 * img.shape[0], 0.60 * img.shape[0], 0.05 * img.shape[1], 0.15 * img.shape[1])
    min_height, max_height, min_width, max_width = character_dimensions
    characters = []
    gif = []
    for i in range(0, TotalComponents):

        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]

        if h > min_height and h < max_height and w > min_width and w < max_width:
            roi = img[y:y + h, x:x + w]
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 1)
            characters.append(roi)
            componentMask = (labels == i).astype("uint8") * 255
            mask = cv2.bitwise_or(mask, componentMask)

    return mask


def checkResultFile(path):
    if os.path.isfile(path):
        return True
    else:
        # create the file
        file = open(path, "w")
        file.write("")
        file.close()
        return True


def writeResults(result, path):
    if checkResultFile(path):
        for detection in result:
            if detection[2] > 0.45:
                file = open(path, "a")
                file.write(detection[1])
        try:
            file.write(' ' + datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
            file.write('\n')
            file.close()
        except:
            print('No license number characters were identified')


def resize(img, scale):
    width = int(img.shape[1] * scale / 100)
    height = int(img.shape[0] * scale / 100)
    dim = (width, height)
    # resize image
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    return resized


def cleandir():
    try:
        if len(os.listdir('ALPR\Results\TrackedPlates')) > 0:
            for file in os.listdir('ALPR\Results\TrackedPlates'):
                os.remove('ALPR\Results\TrackedPlates\\' + file)
            os.remove('ALPR\Results\Recognized.txt')
        os.removedirs('ALPR\Results\TrackedPlates')
        os.makedirs('ALPR\Results\TrackedPlates')
    except:
        os.makedirs('ALPR\Results\TrackedPlates')


def main(img, c=0):
    image = img
    image = resize(image, 200)
    impath = f'ALPR\Results\TrackedPlates\Tracked{c}.jpg'
    respath = 'ALPR\Results\Recognized.txt'
    # Preprocessing the image starts

    # Convert the image to gray scale
    gray = grayscale(image)

    # Performing OTSU threshold
    ret, otsuthresh = otsuthreshold(gray)

    thresh = erode(otsuthresh, 1)
    thresh = dilate(thresh, 1)

    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh)

    mask = examineComponents(ret, labels, stats, centroids, image, thresh)

    cv2.imwrite(impath, mask)

    print('done')
    return mask
