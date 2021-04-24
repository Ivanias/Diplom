import cv2
import numpy as np
import utlis
import sys

net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
car=[]
classes = []
with open("coco.names.txt","r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

image=cv2.imread("Car.png")
height, width, channels=image.shape
blob= cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0,0,0), True, crop=False)
net.setInput(blob)
outs= net.forward(output_layers)
class_ids =[]
confidences =[]
boxes=[]

for out in outs:
    for detection in out:
        scores=detection[5:]
        class_id=np.argmax(scores)
        confidence=scores[class_id]
        if confidence > 0.5:
            center_x=int(detection[0]*width)
            center_y = int(detection[1] * height)
            #Координаты прямоугольника
            w= int(detection[2]* width)
            h= int(detection[3]* height)
            x= int(center_x-w/2)
            y= int(center_y-h/2)

            boxes.append([x,y,w,h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

indexes=cv2.dnn.NMSBoxes(boxes, confidences, 0.5,0.4)
font=cv2.FONT_HERSHEY_PLAIN
#Отображение прямоугольников
for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        print("Координаты центра объекта на исходном изображении",(x+x+w)/2,(y+y+h)/2)
        car=[(int((x+x+w)/2),int((y+y+h)/2))]
        #print (car[0])
        cv2.circle(image, car[0], 3, (0, 255, 0), -1)
        cv2.putText(image, label, (x, y + 30), font, 2, (0, 255, 0), 3)

cv2.imshow("Results", image)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
########################################################################
webCamFeed = True
pathImage = "Car.png"
#cap = cv2.VideoCapture(1)
#cap.set(10, 160)
heightImg = 480
widthImg = 640
kheight = 720/heightImg
kwidth = 1280/widthImg
fon=cv2.imread("fon.png")
plan=cv2.imread('plan.png')
########################################################################

#utlis.initializeTrackbars()
count = 0

while True:

    imgBlank=np.zeros((heightImg,widthImg,3),np.uint8)

    img = cv2.imread(pathImage)
    img = cv2.resize(img, (widthImg, heightImg))  # RESIZE IMAGE
    imgBlank = np.zeros((heightImg, widthImg, 3), np.uint8)  # CREATE A BLANK IMAGE FOR TESTING DEBUGING IF REQUIRED
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # CONVERT IMAGE TO GRAY SCALE
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)  # ADD GAUSSIAN BLUR
    thres = utlis.valTrackbars()  # GET TRACK BAR VALUES FOR THRESHOLDS
    imgThreshold = cv2.Canny(imgBlur, thres[0], thres[1])  # APPLY CANNY BLUR
    kernel = np.ones((5, 5))
    imgDial = cv2.dilate(imgThreshold, kernel, iterations=2)  # APPLY DILATION
    imgThreshold = cv2.erode(imgDial, kernel, iterations=1)  # APPLY EROSION

    ## FIND ALL COUNTOURS
    imgContours = img.copy()  # COPY IMAGE FOR DISPLAY PURPOSES
    imgBigContour = img.copy()  # COPY IMAGE FOR DISPLAY PURPOSES
    contours, hierarchy = cv2.findContours(imgThreshold, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)  # FIND ALL CONTOURS
    cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 10)  # DRAW ALL DETECTED CONTOURS

    # FIND THE BIGGEST COUNTOUR
    biggest, maxArea = utlis.biggestContour(contours)  # FIND THE BIGGEST CONTOUR
    #print(biggest)
    if biggest.size != 0:
        biggest = utlis.reorder(biggest)
        cv2.drawContours(imgBigContour, biggest, -1, (0, 255, 0), 20)  # DRAW THE BIGGEST CONTOUR
        imgBigContour = utlis.drawRectangle(imgBigContour, biggest, 2)
        #cv2.drawContours(fon, contours, -1, (0, 0, 0), 1)
        #cv2.circle(fon, (int((x+x+w)/2/kwidth),int((y+y+h)/2/kheight)), 3, (0, 255, 0), -1)
        cv2.rectangle(fon, (int(x/kwidth), int(y/kheight)), (int((x + w)/kwidth), int((y + h)/kheight)), (0, 255, 0), 2)
        #cv2.imshow("fon", fon)
        pts1 = np.float32(biggest)  # PREPARE POINTS FOR WARP
        pts2 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])  # PREPARE POINTS FOR WARP
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg))
        fon1=cv2.warpPerspective(fon, matrix, (widthImg, heightImg))
        #cv2.imshow("fon1", fon1)

        imgGray1 = cv2.cvtColor(fon1, cv2.COLOR_BGR2GRAY)  # CONVERT IMAGE TO GRAY SCALE
        imgBlur1 = cv2.GaussianBlur(imgGray1, (5, 5), 1)  # ADD GAUSSIAN BLUR
        thres1 = utlis.valTrackbars()  # GET TRACK BAR VALUES FOR THRESHOLDS
        imgThreshold1 = cv2.Canny(imgBlur1, thres1[0], thres1[1])  # APPLY CANNY BLUR
        kernel1 = np.ones((5, 5))
        imgDial1 = cv2.dilate(imgThreshold1, kernel1, iterations=2)  # APPLY DILATION
        imgThreshold1 = cv2.erode(imgDial1, kernel1, iterations=1)  # APPLY EROSION

        ## FIND ALL COUNTOURS
        imgContours1 = fon1.copy()  # COPY IMAGE FOR DISPLAY PURPOSES
        imgBigContour1 = fon1.copy()  # COPY IMAGE FOR DISPLAY PURPOSES
        contours1, hierarchy1 = cv2.findContours(imgThreshold1, cv2.RETR_EXTERNAL,
                                                 cv2.CHAIN_APPROX_SIMPLE)  # FIND ALL CONTOURS
        #cv2.drawContours(fon1, contours1, -1, (255, 0, 0), 10)  # DRAW ALL DETECTED CONTOURS

        # FIND THE BIGGEST COUNTOUR
        biggest1, maxArea1 = utlis.biggestContour(contours1)  # FIND THE BIGGEST CONTOUR
        #print(biggest1)
        if biggest1.size != 0:
            biggest1 = utlis.reorder(biggest1)
            cv2.drawContours(imgBigContour1, biggest1, -1, (0, 255, 255), 20)  # DRAW THE BIGGEST CONTOUR
            imgBigContour1 = utlis.drawRectangle(imgBigContour1, biggest1, 2)
            cv2.drawContours(fon1, contours1, -1, (255, 0, 0), 1)
            x11 = int(str(biggest1[0])[2:6])
            #print('x11',x11)
            x12 = int(str(biggest1[1])[2:6])
            #print('x12', x12)
            x21 = int(str(biggest1[2])[2:6])
            #print('x21', x21)
            x22 = int(str(biggest1[3])[2:6])
            #print('x22', x22)
            y11 = int(str(biggest1[0])[6:-2])
            #print('y11', y11)
            y12 = int(str(biggest1[1])[6:-2])
            #print('y12', y12)
            y21 = int(str(biggest1[2])[6:-2])
            #print('y21', y21)
            y22 = int(str(biggest1[3])[6:-2])
            #print('y22', y22)
            #x2 = str(points[2])
            #x2 = int(x2[8:x2.find(",")])
            cv2.circle(fon1,(int(((x11+x12)/2+(x21+x22)/2)/2),int(((y11+y21)/2+(y12+y22)/2)/2)), 3,(0, 0, 0), -1)
            cv2.circle(plan,
                       (int(((x11 + x12) / 2 + (x21 + x22) / 2) / 2*kwidth), int(((y11 + y21) / 2 + (y12 + y22) / 2) / 2*kheight)), 3,
                       (0, 0, 0), -1)
            print('Координаты центра объекта на повёрнутом изображении',(((x11 + x12) / 2 + (x21 + x22) / 2) / 2*kwidth),(((y11 + y21) / 2 + (y12 + y22) / 2) / 2*kheight))
            #cv2.circle(fon1, (int(((96+377)/2+(116+397)/2)/2),int(((181+345)/2+(180+344)/2)/2)), 3, (0, 0, 0), -1)
            # cv2.circle(fon1, (int((x+x+w)/2/kwidth),int((y+y+h)/2/kheight)), 3, (0, 255, 0), -1)
            # cv2.rectangle(fon, (int(x / kwidth), int(y / kheight)), (int((x + w) / kwidth), int((y + h) / kheight)),(0, 255, 0), 2)
            #cv2.imshow("fonpoint", fon1)
            cv2.imshow("Plan", plan)






        # REMOVE 20 PIXELS FORM EACH SIDE
        #imgWarpColored = imgWarpColored[20:imgWarpColored.shape[0] - 20, 20:imgWarpColored.shape[1] - 20]
        #imgWarpColored = cv2.resize(imgWarpColored, (widthImg, heightImg))
        #fon1 = fon1[1:fon1.shape[0] - 1, 1:fon1.shape[1] - 1]
        #fon1 = cv2.resize(fon1, (widthImg, heightImg))
        #cv2.imshow("fon2", fon1)

        # APPLY ADAPTIVE THRESHOLD
        imgWarpGray = cv2.cvtColor(imgWarpColored, cv2.COLOR_BGR2GRAY)
        imgAdaptiveThre = cv2.adaptiveThreshold(imgWarpGray, 255, 1, 1, 7, 2)
        imgAdaptiveThre = cv2.bitwise_not(imgAdaptiveThre)
        imgAdaptiveThre = cv2.medianBlur(imgAdaptiveThre, 3)

        # Image Array for Display
        imageArray = ([img, imgGray, imgThreshold, imgContours],
                      [imgBigContour, imgWarpColored, imgWarpGray, imgAdaptiveThre])
        #imageArray = ([img, imgGray, imgThreshold, imgContours],
        #              [imgBigContour, imgWarpColored, imgBlank, imgBlank])

    else:
        imageArray = ([img, imgGray, imgThreshold, imgContours],
                      [imgBigContour, imgBlank, imgBlank, imgBlank])

        imgGray1 = cv2.cvtColor(fon1, cv2.COLOR_BGR2GRAY)  # CONVERT IMAGE TO GRAY SCALE
        imgBlur1 = cv2.GaussianBlur(imgGray1, (5, 5), 1)  # ADD GAUSSIAN BLUR
        thres1 = utlis.valTrackbars()  # GET TRACK BAR VALUES FOR THRESHOLDS
        imgThreshold1 = cv2.Canny(imgBlur1, thres1[0], thres1[1])  # APPLY CANNY BLUR
        kernel1 = np.ones((5, 5))
        imgDial1 = cv2.dilate(imgThreshold1, kernel1, iterations=2)  # APPLY DILATION
        imgThreshold1 = cv2.erode(imgDial1, kernel1, iterations=1)  # APPLY EROSION

        ## FIND ALL COUNTOURS
        imgContours1 = fon1.copy()  # COPY IMAGE FOR DISPLAY PURPOSES
        imgBigContour1 = fon1.copy()  # COPY IMAGE FOR DISPLAY PURPOSES
        contours1, hierarchy1 = cv2.findContours(imgThreshold1, cv2.RETR_EXTERNAL,
                                                 cv2.CHAIN_APPROX_SIMPLE)  # FIND ALL CONTOURS
        #cv2.drawContours(fon1, contours1, -1, (255, 0, 0), 10)  # DRAW ALL DETECTED CONTOURS

        # FIND THE BIGGEST COUNTOUR
        biggest1, maxArea1 = utlis.biggestContour(contours1)  # FIND THE BIGGEST CONTOUR
        print(biggest1)
        if biggest1.size != 0:
            biggest1 = utlis.reorder(biggest1)
            #cv2.drawContours(imgBigContour1, biggest1, -1, (0, 255, 0), 20)  # DRAW THE BIGGEST CONTOUR
            imgBigContour1 = utlis.drawRectangle(imgBigContour1, biggest1, 2)
            #cv2.drawContours(fon1, contours1, -1, (255, 0, 0), 1)
            # cv2.circle(fon1, (int((x+x+w)/2/kwidth),int((y+y+h)/2/kheight)), 3, (0, 255, 0), -1)
            # cv2.rectangle(fon, (int(x / kwidth), int(y / kheight)), (int((x + w) / kwidth), int((y + h) / kheight)),(0, 255, 0), 2)
            cv2.imshow("fonpoint", fon1)


    # LABELS FOR DISPLAY
    #lables = [["Original", "Gray", "Threshold", "Contours"],["Biggest Contour", "Warp Prespective", "Warp Gray", "Adaptive Threshold"]]

    #stackedImage = utlis.stackImages(imageArray, 0.75, lables)
    #cv2.imshow("Result", stackedImage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()








"""
    # SAVE IMAGE WHEN 's' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('s'):
        cv2.imwrite("Image/myImage" + str(count) + ".jpg", imgWarpColored)
        cv2.rectangle(stackedImage, ((int(stackedImage.shape[1] / 2) - 230), int(stackedImage.shape[0] / 2) + 50),
                      (1100, 350), (0, 255, 0), cv2.FILLED)
        cv2.putText(stackedImage, "Scan Saved", (int(stackedImage.shape[1] / 2) - 200, int(stackedImage.shape[0] / 2)),
                    cv2.FONT_HERSHEY_DUPLEX, 3, (0, 0, 255), 5, cv2.LINE_AA)
        cv2.imshow('Result', stackedImage)
        cv2.waitKey(300)
        count += 1
"""
