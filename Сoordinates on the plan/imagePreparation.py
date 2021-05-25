import cv2
import numpy as np
import sys
import configparser

def truckDetection (image,fon,x,y,w,h,kheight,kwidth):
    # Нахождение объекта на исходном изображении
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    classes = []
    with open("coco.names.txt", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    height, width, channels = image.shape
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.3:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                # Координаты прямоугольника
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.3)
    font = cv2.FONT_HERSHEY_PLAIN
    # Отображение прямоугольников
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(image, (int((x + x + w) / 2), int((y + y + h) / 2)), 3, (0, 255, 0), -1)
    #cv2.imshow("image", image)
    cv2.rectangle(fon, (int(x/kwidth), int((y+h)/kheight)), (int((x + w)/kwidth), int((y + h)/kheight)), (0, 255, 0), 2) # Рисование на белом листе прямоугольника, ограничивающий объект
    #cv2.imshow("fon", fon)
    return image,fon,x,y,w,h

def imagePreparation (img):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
    imgThreshold = cv2.Canny(imgBlur, 50, 400)
    return imgThreshold

def imagePreparation1 (img):
    #cv2.imshow("test1", img)
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
    imgThreshold = cv2.Canny(imgBlur, 50, 200)
    #cv2.imshow("test2", imgThreshold)
    return imgThreshold

def domatrix(fon,matrix,widthImg,heightImg):
    newfon = cv2.warpPerspective(fon, matrix, (widthImg, heightImg))
    #cv2.imshow("Warp white list", newfon)
    return newfon

def coordinatesOnThePlan(biggest1,newfon,finalPlan):
    x11 = int(str(biggest1[0])[2:5])
    x12 = int(str(biggest1[1])[2:5])
    x21 = int(str(biggest1[2])[2:5])
    x22 = int(str(biggest1[3])[2:5])
    y11 = int(str(biggest1[0])[5:-2])
    y12 = int(str(biggest1[1])[5:-2])
    y21 = int(str(biggest1[2])[5:-2])
    y22 = int(str(biggest1[3])[5:-2])
    klength=780/640
    kwidth=250/480
    cv2.circle(newfon, (int(((x11 + x12) / 2 + (x21 + x22) / 2) / 2), int(((y11 + y21) / 2 + (y12 + y22) / 2) / 2)), 3,
               (0, 0, 0), -1)
    cv2.circle(finalPlan, (int((((x11 + x12) / 2 + (x21 + x22) / 2) / 2)*klength+304), int(((((y11 + y21) / 2 + (y12 + y22) / 2) / 2))*kwidth+67)), 3,
               (0, 255, 0), -1)
    print('Координаты центра объекта на складе:', int((((x11 + x12) / 2 + (x21 + x22) / 2) / 2)*klength+304), int(((((y11 + y21) / 2 + (y12 + y22) / 2) / 2))*kwidth+67))
    cv2.imshow("finalPlan", finalPlan)
    return finalPlan


def reorder(myPoints):
    myPoints = myPoints.reshape((4, 2))
    myPointsNew = np.zeros((4, 1, 2), dtype=np.int32)
    add = myPoints.sum(1)

    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] = myPoints[np.argmax(add)]
    diff = np.diff(myPoints, axis=1)
    myPointsNew[1] = myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]
    return myPointsNew

def biggestContour(contours):
    biggest = np.array([])
    max_area = 0
    for i in contours:
        area = cv2.contourArea(i)
        if area > 5000:
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02 * peri, True)
            if area > max_area and len(approx) == 4:
                biggest = approx
                max_area = area
    return biggest, max_area


def drawRectangle(img, biggest, thickness):
    cv2.line(img, (biggest[0][0][0], biggest[0][0][1]), (biggest[1][0][0], biggest[1][0][1]), (0, 255, 255), thickness)
    cv2.line(img, (biggest[0][0][0], biggest[0][0][1]), (biggest[2][0][0], biggest[2][0][1]), (0, 255, 255), thickness)
    cv2.line(img, (biggest[3][0][0], biggest[3][0][1]), (biggest[2][0][0], biggest[2][0][1]), (0, 255, 255), thickness)
    cv2.line(img, (biggest[3][0][0], biggest[3][0][1]), (biggest[1][0][0], biggest[1][0][1]), (0, 255, 255), thickness)
    return img

def getCoordinates (countX,countY1,countY2,contours1):
    for i in range(len(contours1[0])):
        countX = countX + contours1[0][i][0][0]
        if i==0:
            countY1=contours1[0][i][0][1]
            continue
        if contours1[0][i][0][1]/contours1[0][0][0][1]<=1.15:
            if contours1[0][i][0][1]/contours1[0][0][0][1]>= 0.85:
                countY1 = countY1 + contours1[0][i][0][1]
            else:
                countY2=countY2+contours1[0][i][0][1]
                if i != len(contours1[0]) - 1:
                    continue
        else:
            countY2 = countY2 + contours1[0][i][0][1]
        if i == len(contours1[0])-1:
            countX=countX/len(contours1[0])
            if countY1>countY2:
                countYT=countY1
                countY1=countY2
                countY2=countYT
            countY1=countY1/(len(contours1[0])/2)
            countY2=countY2/(len(contours1[0])/2)

    coord1=[countX,countY1]
    coord2=[countX,countY2]
    return coord1,coord2

def getRectangle (coord1,coord2,newfon):
    config = configparser.ConfigParser()  # создаём объекта парсера
    config.read("settings.ini")  # читаем конфиг
    sizeXm=float(config["forklift positioning"]["lengthForklift"])
    sizeYm=float(config["forklift positioning"]["widthForklift"])
    sizePX=((coord2[1]-coord1[1])/sizeXm)*sizeYm
    cv2.rectangle(newfon, (int(coord1[0]), int(coord1[1])), (int(coord2[0]+sizePX), int(coord2[1])), (0, 255, 0), 2) # Рисование на белом листе прямоугольника, ограничивающий объект
    #cv2.imshow("inew", newfon)
    return newfon