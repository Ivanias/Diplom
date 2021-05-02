import cv2
import numpy as np
import func
import sys

# Нахождение объекта на исходном изображении
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
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
        cv2.circle(image, (int((x+x+w)/2),int((y+y+h)/2)), 3, (0, 255, 0), -1)

# Поворачивание исходного изображения и нахождения на нём объекта
webCamFeed = True
pathImage = "Car.png"
heightImg = 480
widthImg = 640
kheight = 720/heightImg
kwidth = 1280/widthImg
fon=cv2.imread("fon.png") # Белый лист
plan=cv2.imread('plan.png') # План склада
count = 0

#while True:

imgBlank=np.zeros((heightImg,widthImg,3),np.uint8)

# Обработка изображения для того чтобы нахождения контуров стало возможным
img = cv2.imread(pathImage)
img = cv2.resize(img, (widthImg, heightImg))
imgBlank = np.zeros((heightImg, widthImg, 3), np.uint8)
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
imgThreshold = cv2.Canny(imgBlur, 200, 200)
kernel = np.ones((5, 5))
imgDial = cv2.dilate(imgThreshold, kernel, iterations=2)
imgThreshold = cv2.erode(imgDial, kernel, iterations=1)

#Нахождение всех контуров
imgContours = img.copy()
imgBigContour = img.copy()
contours, hierarchy = cv2.findContours(imgThreshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 10)
#cv2.imshow("imgContours", imgContours)

# Нахождение самого большого контура и поворот изображения
biggest, maxArea = func.biggestContour(contours)
if biggest.size != 0:
    biggest = func.reorder(biggest)
    cv2.drawContours(imgBigContour, biggest, -1, (0, 255, 0), 20)
    imgBigContour = func.drawRectangle(imgBigContour, biggest, 2)
    cv2.rectangle(fon, (int(x/kwidth), int(y/kheight)), (int((x + w)/kwidth), int((y + h)/kheight)), (0, 255, 0), 2) # Рисование на белом листе прямоугольника, ограничивающий объект
    #cv2.imshow("fon", fon)
    pts1 = np.float32(biggest)
    pts2 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg)) # Поворот исходного изображения
    #cv2.imshow("Warp image", imgWarpColored)
    newfon=cv2.warpPerspective(fon, matrix, (widthImg, heightImg))
    #cv2.imshow("Warp white list", newfon)

    # Обработка повёрнутого изображения для того чтобы нахождения контуров стало возможным
    imgGray1 = cv2.cvtColor(newfon, cv2.COLOR_BGR2GRAY)
    imgBlur1 = cv2.GaussianBlur(imgGray1, (5, 5), 1)
    imgThreshold1 = cv2.Canny(imgBlur1, 200, 200)
    kernel1 = np.ones((5, 5))
    imgDial1 = cv2.dilate(imgThreshold1, kernel1, iterations=2)
    imgThreshold1 = cv2.erode(imgDial1, kernel1, iterations=1)

    # Нахождение всех контуров повёрнутого изображения
    imgContours1 = newfon.copy()
    imgBigContour1 = newfon.copy()
    contours1, hierarchy1 = cv2.findContours(imgThreshold1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Нахождение самого большого контура повёрнутого изображения и нахождения центра объекта
    biggest1, maxArea1 = func.biggestContour(contours1)
    if biggest1.size != 0:
        biggest1 = func.reorder(biggest1)
        cv2.drawContours(imgBigContour1, biggest1, -1, (0, 255, 255), 20)  # DRAW THE BIGGEST CONTOUR
        imgBigContour1 = func.drawRectangle(imgBigContour1, biggest1, 2)
        #cv2.imshow("imgBigContour1", imgBigContour1)
        cv2.drawContours(newfon, contours1, -1, (255, 0, 0), 1)
        x11 = int(str(biggest1[0])[2:6])
        x12 = int(str(biggest1[1])[2:6])
        x21 = int(str(biggest1[2])[2:6])
        x22 = int(str(biggest1[3])[2:6])
        y11 = int(str(biggest1[0])[6:-2])
        y12 = int(str(biggest1[1])[6:-2])
        y21 = int(str(biggest1[2])[6:-2])
        y22 = int(str(biggest1[3])[6:-2])
        cv2.circle(newfon,(int(((x11+x12)/2+(x21+x22)/2)/2),int(((y11+y21)/2+(y12+y22)/2)/2)), 3,(0, 0, 0), -1)
        cv2.circle(plan,(int(((x11+x12)/2+(x21+x22)/2)/2),int(((y11+y21)/2+(y12+y22)/2)/2)), 3,(0, 0, 0), -1)
        print('Координаты центра объекта на повёрнутом изображении',(((x11 + x12) / 2 + (x21 + x22) / 2) / 2),(((y11 + y21) / 2 + (y12 + y22) / 2) / 2))
        cv2.imshow("Plan", plan)

else:
    print("Ошибка: нет замкнутых контуров")

cv2.waitKey(0)
cv2.destroyAllWindows()
