# organize imports
import cv2
import imutils
import numpy as np

# global variables
bg = None


#---------------------------------------------
# To segment the region of hand in the image
#---------------------------------------------
def segment(image, threshold=9):
    global bg
    # find the absolute difference between background and current frame
    diff = cv2.absdiff(bg.astype("uint8"), image)
    cv2.imshow("diff", diff)
    # threshold the diff image so that we get the foreground
    thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)[1]

    # get the contours in the thresholded image
    (cnts, _) = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # return None, if no contours detected
    if len(cnts) == 0:
        return
    else:
        # based on contour area, get the maximum contour which is the hand
        segmented = max(cnts, key=cv2.contourArea)
        return (thresholded, segmented)



#-----------------
# MAIN FUNCTION
#-----------------
if __name__ == "__main__":
    # initialize weight for running average
    aWeight = 0.5
    num_frames = 0

    # get the reference to the picture



    camera = cv2.imread('images/Hand_0000106.jpg')

    image = cv2.resize(camera, (350,700))

        # region of interest (ROI) coordinates
    top, right, bottom, left = 10, 20, 695, 340

    # cv2.imshow("Image", image)


    clone = image.copy()


    (height, width) = image.shape[:2]


        # get the ROI

        # convert the roi to grayscale and blur it
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)



    # background
    bg = gray.copy().astype("float")


    # accumulated= cv2.accumulateWeighted(gray, bg, aWeight)


    # run_avg(gray, aWeight)

        #         # segment the hand region
    hand = segment(blur)

    (thresholded, segmented) = hand
        #
        #             # draw the segmented region and display the frame
    # cv2.drawContours(clone, [segmented + (right, top)], -1, (0, 0, 255))
    cv2.drawContours(clone, [segmented], -1, (0, 0, 255))
    cv2.imshow("Thesholded", thresholded)
        #
        #     # draw the segmented hand
    cv2.rectangle(clone, (left, top), (right, bottom), (0,255,0), 2)
        #
        #     # increment the number of frames

        #
        #     # display the frame with segmented hand
    cv2.imshow("Video Feed", clone)



    fillpoly= cv2.fillPoly(thresholded , pts = [segmented], color=(255,255,255))

    cv2.imshow("FillPolly", fillpoly)
        #
        #     # observe the keypress by the user
    keypress = cv2.waitKey(1) & 0xFF
        #
        #     # if the user pressed "q", then stop looping


# free up memory
cv2.waitKey(0)

cv2.destroyAllWindows()
