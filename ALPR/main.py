from ALPR import tracking2
from ALPR import ocr
from ALPR import freq
import easyocr


def main():
    # #reset the results folder
    # ocr.cleandir()
    # reader = easyocr.Reader(['en'])
    # #create an instance of custom plate recognition tracking that uses custom trained yolov5 model to identify the location of the license plate on each frame
    # track = tracking.plateRecognition(model_name='ALPR/best.pt', filename="ALPR/IMG_8716.mp4")
    # track()
    # #run the processed pictures through the ocr reader
    # ocr.findLicenses(reader)
    # plate = freq.find_frequency()
    # print(plate)
    # f = open("ALPR/plate.txt", "w")
    # f.write(plate)
    # f.close()
    tracking2.test()

if __name__ == '__main__':
    main()

