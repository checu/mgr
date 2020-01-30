import cv2
import os
import numpy as np
import math



input_folder = "/Users/chekumis/Desktop/PalmarBBGtest19_21_29/"
output_folder = "/Users/chekumis/Desktop/PalmarBBGtest19_21_29/PalmAreas/"

os.mkdir(output_folder)


#import data
image_list = []
with open(input_folder + "PalmCoordinates.txt") as f:
    for line in f:
        inner_list = [elt.strip() for elt in line.split(',')]
        image_list.append(inner_list)

f.close()

print(image_list)

for image in image_list:

    image_valid = True

    print(image[0])

# get key points values

    momentCoord = (int(image[1][1:]),int(image[2][:-1]))

    circleCenter = (int(image[3][1:]),int(image[4][:-1]))



    if image[5] != 'None' and image[6] != 'None':

        try:

            startPoint = (int(image[5][1:]),int(image[6][:-1]))

            endPoint = (int(image[7][1:]),int(image[8][:-1]))

        except:

            # print(image[5][1:],image[6][:-1],image[7][1:],image[8][:-1])

            image_valid = True

            print("Invalid Coordinates")


    else:

        image_valid = False

        print("Invalid Coordinates")

# extract area

    if image_valid:

        img = cv2.imread(input_folder + image[0])

        # cv2.line(img, startPoint, endPoint, (0, 250, 0), 2)

        # cv2.line(img, circleCenter, momentCoord, (0, 250, 0), 2)

        # cv2.imshow("lines", img)

        distHorizontal = math.sqrt((momentCoord[0]-circleCenter[0])**2 + (momentCoord[1]-circleCenter[1])**2)

        distVertical = math.sqrt((startPoint[0]-endPoint[0])**2 + (startPoint[1]-endPoint[1])**2)

        if circleCenter[0]!= momentCoord[0]:

            a = ((circleCenter[1]-momentCoord[1])/(circleCenter[0]-momentCoord[0]))


            b1 = startPoint[1] - a*startPoint[0]

            b2 = endPoint[1] - a*endPoint[0]

            # calculate alpha
            angleRadians = math.atan(a)

            angleDegrees = math.degrees(angleRadians)

            cx = int(distVertical * math.cos(angleRadians))

            cy = int(distVertical * math.sin(angleRadians))

            dx = int(distVertical * math.cos(angleRadians))

            dy = int(distVertical * math.sin(angleRadians))

            if a > 0:
                C = (startPoint[0] - cx, startPoint[1] - abs(cy))

                D = (endPoint[0] - dx, endPoint[1] - abs(dy))

            else:

                C = (startPoint[0] + cx, startPoint[1] - abs(cy))

                D = (endPoint[0] + dx, endPoint[1] - abs(dy))


            points = np.array([startPoint,C,D,endPoint])

        else:

            points = np.array([startPoint, (startPoint[0], startPoint[1] - int(distVertical)), (endPoint[0], endPoint[1] - int(distVertical)), endPoint])


        cv2.polylines(img,[points],True,(0,255,255))
        # cv2.fillPoly(img,[points],(255,255,255))

        # cv2.line(img, startPoint, C, (0, 250, 0), 2)
        # cv2.line(img, endPoint, D, (0, 250, 0), 2)
        # cv2.line(img, C, D, (0, 250, 0), 2)
        # wspolczynnik korelacji dla szukanych wspolrzednych

        mask = np.zeros(img.shape, dtype=np.uint8)

        roi_corners = np.array(points, dtype=np.int32)

        cv2.fillPoly(mask, [points], (255,255,255))

        masked_image = cv2.bitwise_and(img, mask)


        # cv2.circle(img,C, 2, (255, 0, 255), -1)
        # cv2.circle(img,D, 2, (255,255, 255), -1)


        # cv2.imshow("Cross", masked_image)

        cv2.imwrite(output_folder + image[0],masked_image)


        # cv2.waitKey(0)& 0xFF== ord("q")
        # cv2.destroyAllWindows()

    else:
        pass