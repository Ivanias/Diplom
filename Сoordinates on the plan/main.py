import cv2
import numpy as np
import imagePreparation
import sys

x,y,w,h=0,0,0,0
heightImg = 480
widthImg = 640
kheight = 761/heightImg
kwidth = 1564/widthImg
image = cv2.imread("overviewcamera.png")
fon=cv2.imread("fon.png") # Белый лист
image,fon,x,y,w,h=imagePreparation.truckDetection(image,fon,x,y,w,h,kheight,kwidth)

webCamFeed = True
pathImage = "workzone.png"
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
#cv2.imshow("imgContours", imgContours)

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
    coord1, coord2=imagePreparation.getCoordinates(countX,countY1,countY2,contours1)

    newfon=imagePreparation.getRectangle(coord1,coord2,newfon)

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
        #cv2.imshow("imgBigContour2", imgBigContour2)
        cv2.drawContours(newfon, contours2, -1, (255, 0, 0), 1)
        plan=imagePreparation.coordinatesOnThePlan(biggest2,newfon,finalPlan)

else:
    print("Ошибка: нет замкнутых контуров")

cv2.waitKey(0)
cv2.destroyAllWindows()