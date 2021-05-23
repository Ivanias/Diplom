import cv2
import numpy as np
import imagePreparation
import sys

x,y,w,h=0,0,0,0
heightImg = 480
widthImg = 640
kheight = 761/heightImg
kwidth = 1564/widthImg
image = cv2.imread("newpog.png")
fon=cv2.imread("fon.png") # Белый лист
image,fon,x,y,w,h=imagePreparation.truckDetection(image,fon,x,y,w,h,kheight,kwidth)

webCamFeed = True
pathImage = "newzona.png"
plan=cv2.imread('plan.png') # План склада
finalPlan=cv2.imread('finalPlan.png') # План склада
count = 0

imgBlank=np.zeros((heightImg,widthImg,3),np.uint8)

# Обработка изображения для того чтобы нахождения контуров стало возможным
img = cv2.imread(pathImage)
img = cv2.resize(img, (widthImg, heightImg))
imgBlank = np.zeros((heightImg, widthImg, 3), np.uint8)
imgThreshold = imagePreparation.imagePreparation(img)

#Нахождение всех контуров
imgContours = img.copy()
imgBigContour = img.copy()
contours, hierarchy = cv2.findContours(imgThreshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 10)
cv2.imshow("imgContours", imgContours)

# Нахождение самого большого контура и поворот изображения
biggest, maxArea = imagePreparation.biggestContour(contours)
if biggest.size != 0:
    biggest = imagePreparation.reorder(biggest)
    cv2.drawContours(imgBigContour, biggest, -1, (0, 255, 0), 20)
    imgBigContour = imagePreparation.drawRectangle(imgBigContour, biggest, 2)
    pts1 = np.float32(biggest)
    pts2 = np.float32([[widthImg, heightImg], [widthImg, 0],[0, heightImg], [0, 0]])
    #pts2 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg)) # Поворот исходного изображения
    #cv2.imshow("Warp image", imgWarpColored)
    newfon=imagePreparation.domatrix(fon,matrix,widthImg,heightImg)

    # Обработка повёрнутого изображения для того чтобы нахождения контуров стало возможным
    imgThreshold1 = imagePreparation.imagePreparation1(newfon)

    # Нахождение всех контуров повёрнутого изображения
    imgContours1 = newfon.copy()
    imgBigContour1 = newfon.copy()
    contours1, hierarchy1 = cv2.findContours(imgThreshold1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(imgContours1, contours1, -1, (0, 255, 0), 1)
    #cv2.imshow("imgContours1", imgContours1)

    countX=0
    countY1=0
    countY2=0
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

    import configparser
    config = configparser.ConfigParser()  # создаём объекта парсера
    config.read("settings.ini")  # читаем конфиг
    sizeXm=float(config["forklift positioning"]["lengthForklift"])
    sizeYm=float(config["forklift positioning"]["widthForklift"])
    sizePX=((coord2[1]-coord1[1])/sizeXm)*sizeYm

    cv2.rectangle(newfon, (int(coord1[0]), int(coord1[1])), (int(coord2[0]+sizePX), int(coord2[1])), (0, 255, 0), 2) # Рисование на белом листе прямоугольника, ограничивающий объект
    cv2.imshow("inew", newfon)

    imgThreshold2 = imagePreparation.imagePreparation1(newfon)

    # Нахождение всех контуров повёрнутого изображения
    imgContours2 = newfon.copy()
    imgBigContour2 = newfon.copy()
    contours2, hierarchy2 = cv2.findContours(imgThreshold2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(imgContours2, contours2, -1, (255, 255, 0), 1)
    #cv2.imshow("imgContours2", imgContours2)

    # Нахождение самого большого контура повёрнутого изображения и нахождения центра объекта
    biggest2, maxArea2 = imagePreparation.biggestContour(contours2)
    if biggest2.size != 0:
        biggest1 = imagePreparation.reorder(biggest2)
        cv2.drawContours(imgBigContour2, biggest2, -1, (0, 255, 255), 20)  # DRAW THE BIGGEST CONTOUR
        imgBigContour2 = imagePreparation.drawRectangle(imgBigContour2, biggest2, 2)
        cv2.imshow("imgBigContour2", imgBigContour2)
        cv2.drawContours(newfon, contours2, -1, (255, 0, 0), 1)
        plan=imagePreparation.coordinatesOnThePlan(biggest2,newfon,plan,finalPlan)

else:
    print("Ошибка: нет замкнутых контуров")

cv2.waitKey(0)
cv2.destroyAllWindows()