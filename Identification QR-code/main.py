from __future__ import print_function
import pyzbar.pyzbar as pyzbar
import numpy as np
import cv2

def decode(image):
    # Нахождение QR-кодов
    decodedObjects = pyzbar.decode(image)
    return decodedObjects

# Отображение местоположения QR-кода
def display(image, decodedObjects):
    # Цикл по всем декодированным объектам
    for decodedObject in decodedObjects:
        points = decodedObject.polygon
        # Нахождение координат x1,x2
        x1=str(points[0])
        x1=int(x1[8:x1.find(",")])
        x2=str(points[2])
        x2=int(x2[8:x2.find(",")])
        sizePX =x2-x1
        # Используется для расчёта фокусного расстояния камеры
        #if np.any(image == cv2.imread("QR-codeC270distance1m.png")):
            #distance = 1
            #sizeM = 0.05
            #focus = sizePX * distance / sizeM
            #print(focus)
        # Расчёт расстояния от камеры до QR-кода (в метрах)
        focus = 1140
        sizeM = 0.05
        distance=sizeM*focus/sizePX
        print('Distance :',distance)
        # Проверка является ли QR-код на расстоянии, допустимом для записи с него информации
        if distance<=1:
            print('Data : ', decodedObject.data, '\n')
        # Количество точек
        n = len(points)
        for j in range(0, n):
            cv2.line(image, points[j], points[(j + 1) % n], (255, 0, 0), 3)
    # Отображение результата
    cv2.imshow("Results", image);
    cv2.waitKey(0);

if __name__ == '__main__':
    image = cv2.imread("QR-codeC270distance1m.png")
    decodedObjects = decode(image)
    display(image, decodedObjects)


