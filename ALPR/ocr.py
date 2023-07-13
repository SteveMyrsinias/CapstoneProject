'''
https://www.geeksforgeeks.org/text-detection-and-extraction-using-opencv-and-ocr/ original inspiration
https://pyimagesearch.com/2021/02/22/opencv-connected-component-labeling-and-analysis/ for component analysis
https://stackoverflow.com/questions/72381645/python-tesseract-license-plate-recognition for easyocr
'''
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime

 

def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def otsuthreshold(img):
    return cv2.threshold(img, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)


def erode(img,iter):
    return cv2.erode(img, None, iterations=iter)


def dilate(img,iter):
    return cv2.dilate(img, None, iterations=iter)
 

def examineComponents(TotalComponents,labels,stats,centroids,img,thresh):
    mask = np.zeros(thresh.shape, dtype="uint8")
    # gif = []
    for i in range(0, TotalComponents):
        # if this is the first component then we examine the
        # *background* (typically we would just ignore this
        # component in our loop)
        # if i == 0:
        #     text = (f"examining component {i+1}/{TotalComponents} (background)")
        # # otherwise, we are examining an actual connected component
        # elif stats[i, cv2.CC_STAT_AREA]/img.size*100 < 0.18:
        #     text = (f"examining component {i +1}/{TotalComponents} (small component)")
        # else:
        #     text = (f"examining component {i +1}/{TotalComponents}")
        # # print a status message update for the current connected
        # # component
        # print(f"[INFO] {text}")
        # extract the connected component statistics and centroid for
        # the current label
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        (cX, cY) = centroids[i]


        keepWidth = w/img.shape[1]*100 > 1.6 and w/img.shape[1]*100 < 11
        keepHeight = h/img.shape[0]*100 > 20 and h/img.shape[0]*100 < 50
        keepArea = area/img.size*100 > 0.09 and area/img.size*100 < 1.00

        # clone our original image (so we can draw on it) and then draw
        # a bounding box surrounding the connected component along with
        # a circle corresponding to the centroid
        output = img.copy()
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv2.circle(output, (int(cX), int(cY)), 4, (0, 0, 255), -1)

        componentMask = (labels == i).astype("uint8") * 255
        # show our output image and connected component mask
        # components = np.vstack((output,cv2.cvtColor(componentMask, cv2.COLOR_GRAY2RGB)))
        # gif.append(components)
        #cv2.imshow("Output", output)
        #cv2.imshow("Connected Component", components)
        #cv2.waitKey(0)

        if all((keepWidth, keepHeight, keepArea)):
            # construct a mask for the current connected component and
            # then take the bitwise OR with the mask 
            # print(f"[INFO] keeping connected component '{i+1}'")
            componentMask = (labels == i).astype("uint8") * 255
            mask = cv2.bitwise_or(mask, componentMask)
        # else:
        #     print('Component Rejected. Reason:')
        #     if not keepWidth:
        #         print(f'Width: {w/img.shape[1]*100}')
        #     if not keepHeight:
        #         print(f'Height: {h/img.shape[0]*100}')
        #     if not keepArea:
        #         print(f'Area: {area/img.size*100}')


    # #Create gif example
    # import imageio
    # filename = 'test.gif'
    # mask1 = np.vstack((img,cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)))
    # for i in range(6):
    #     gif.append(mask1)
    # imageio.mimsave('video.gif', gif, fps=3)
    

    return mask    


def checkResultFile(path):
    if os.path.isfile(path):
        return True
    else:
        #create the file
        file = open(path, "w")
        file.write("")
        file.close()
        return True


def writeResults(result,path):
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
        
     

def ocr(path,reader):
    #IMAGE_PATH = 'Project\Results\Tracked.jpg'
    results = reader.readtext(path)
    return results

def resize(img,scale):
    width = int(img.shape[1] * scale / 100)
    height = int(img.shape[0] * scale / 100)
    dim = (width, height)
    # resize image
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    return resized

def findLicenses(reader):
    for i in range(len(os.listdir('ALPR\Results\TrackedPlates'))):
        result = ocr(f'ALPR\Results\TrackedPlates\Tracked{i}.jpg',reader)
        writeResults(result,'ALPR\Results\Recognized.txt')

def cleandir():
    try:
        if len(os.listdir('ALPR\Results\TrackedPlates')) > 0:
            for file in os.listdir('ALPR\Results\TrackedPlates'):
                os.remove('ALPR\Results\TrackedPlates\\'+file)
            os.remove('ALPR\Results\Recognized.txt')
        os.removedirs('ALPR\Results\TrackedPlates')
        os.makedirs('ALPR\Results\TrackedPlates')
    except:
        os.makedirs('ALPR\Results\TrackedPlates')


def main(img,c=0):
#for picture in os.listdir('Project\LicensePlates'):
    #image = cv2.imread(f"Project\LicensePlates\{picture}")
    # Read image from which text needs to be extracted
    # image = cv2.imread("Project\LicensePlates\griekenland75.jpg")
    image = img
    image = resize(image,200)
    impath = f'ALPR\Results\TrackedPlates\Tracked{c}.jpg'
    respath = 'ALPR\Results\Recognized.txt'
    # Preprocessing the image starts

    # Convert the image to gray scale
    gray = grayscale(image)
    
    # Performing OTSU threshold
    ret, otsuthresh = otsuthreshold(gray)

    # cv2.imshow('gray',gray)
    # cv2.imshow('OTSU Thresholded',otsuthresh)
    # cv2.waitKey(0)
    thresh = erode(otsuthresh,1)
    thresh = dilate(thresh,1)

    #cv2.imshow('thresh',thresh)
    #cv2.waitKey(0)
    #thresh = cv2.bitwise_not(thresh)
    # cv2.imshow('thresh',thresh)
    # cv2.waitKey(0)

    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh)

    mask = examineComponents(ret,labels,stats,centroids,image,thresh)
    
    # cv2.imshow("Image", img)
    # cv2.imshow("Characters", mask)
    cv2.imwrite(impath,mask)
    #cv2.waitKey(0)

    # results = ocr(impath)
    # writeResults(results,respath)
    print('done')
    return np.vstack((image,cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)))



if __name__ == '__main__':
    print(datetime.now())
    main()
    print(datetime.now())