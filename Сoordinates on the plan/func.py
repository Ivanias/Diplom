import cv2
import numpy as np
import sys

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
            if confidence > 0.5:
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

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    font = cv2.FONT_HERSHEY_PLAIN
    # Отображение прямоугольников
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(image, (int((x + x + w) / 2), int((y + y + h) / 2)), 3, (0, 255, 0), -1)
    #cv2.imshow("image", image)
    cv2.rectangle(fon, (int(x/kwidth), int(y/kheight)), (int((x + w)/kwidth), int((y + h)/kheight)), (0, 255, 0), 2) # Рисование на белом листе прямоугольника, ограничивающий объект
    #cv2.imshow("fon", fon)
    return image,fon,x,y,w,h

def imagePreparation (img):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
    imgThreshold = cv2.Canny(imgBlur, 200, 200)
    kernel = np.ones((5, 5))
    imgDial = cv2.dilate(imgThreshold, kernel, iterations=2)
    return imgThreshold

def domatrix(fon,matrix,widthImg,heightImg):
    newfon = cv2.warpPerspective(fon, matrix, (widthImg, heightImg))
    #cv2.imshow("Warp white list", newfon)
    return newfon

def coordinatesOnThePlan(biggest1,newfon,plan):
    x11 = int(str(biggest1[0])[2:6])
    x12 = int(str(biggest1[1])[2:6])
    x21 = int(str(biggest1[2])[2:6])
    x22 = int(str(biggest1[3])[2:6])
    y11 = int(str(biggest1[0])[6:-2])
    y12 = int(str(biggest1[1])[6:-2])
    y21 = int(str(biggest1[2])[6:-2])
    y22 = int(str(biggest1[3])[6:-2])
    cv2.circle(newfon, (int(((x11 + x12) / 2 + (x21 + x22) / 2) / 2), int(((y11 + y21) / 2 + (y12 + y22) / 2) / 2)), 3,
               (0, 0, 0), -1)
    cv2.circle(plan, (int(((x11 + x12) / 2 + (x21 + x22) / 2) / 2), int(((y11 + y21) / 2 + (y12 + y22) / 2) / 2)), 3,
               (0, 0, 0), -1)
    print('Координаты центра объекта на повёрнутом изображении', (((x11 + x12) / 2 + (x21 + x22) / 2) / 2),
          (((y11 + y21) / 2 + (y12 + y22) / 2) / 2))
    cv2.imshow("Plan", plan)
    return plan


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