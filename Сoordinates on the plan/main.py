import cv2
import numpy as np
import imagePreparation
import sys

x,y,w,h=0,0,0,0
heightImg = 480
widthImg = 640
kheight = 720/heightImg
kwidth = 1280/widthImg
image = cv2.imread("Forklift.png")
fon=cv2.imread("fon.png") # Белый лист
image,fon,x,y,w,h=imagePreparation.truckDetection(image,fon,x,y,w,h,kheight,kwidth)

# Поворачивание исходного изображения и нахождения на нём объекта
webCamFeed = True
pathImage = "Forklift.png"
plan=cv2.imread('plan.png') # План склада
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
    pts2 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg)) # Поворот исходного изображения
    #cv2.imshow("Warp image", imgWarpColored)
    newfon=imagePreparation.domatrix(fon,matrix,widthImg,heightImg)

    # Обработка повёрнутого изображения для того чтобы нахождения контуров стало возможным
    imgThreshold1 = imagePreparation.imagePreparation(newfon)

    # Нахождение всех контуров повёрнутого изображения
    imgContours1 = newfon.copy()
    imgBigContour1 = newfon.copy()
    contours1, hierarchy1 = cv2.findContours(imgThreshold1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Нахождение самого большого контура повёрнутого изображения и нахождения центра объекта
    biggest1, maxArea1 = imagePreparation.biggestContour(contours1)
    if biggest1.size != 0:
        biggest1 = imagePreparation.reorder(biggest1)
        cv2.drawContours(imgBigContour1, biggest1, -1, (0, 255, 255), 20)  # DRAW THE BIGGEST CONTOUR
        imgBigContour1 = imagePreparation.drawRectangle(imgBigContour1, biggest1, 2)
        #cv2.imshow("imgBigContour1", imgBigContour1)
        cv2.drawContours(newfon, contours1, -1, (255, 0, 0), 1)
        plan=imagePreparation.coordinatesOnThePlan(biggest1,newfon,plan)

else:
    print("Ошибка: нет замкнутых контуров")

cv2.waitKey(0)
cv2.destroyAllWindows()