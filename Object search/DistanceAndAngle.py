import numpy as np
import cv2
import configparser

config = configparser.ConfigParser()  # создаём объекта парсера
config.read("settings.ini")  # читаем конфиг




def getDistance (x,y,w,h):
    x11 = x
    x12 = x+w
    x21 = x
    x22 = x+w
    y11 = y
    y12 = y
    y21 = y+h
    y22 = y+h
    sizeXpx = ((x12-x11)+(x22-x21))/2
    sizeYpx=((y21-y11)+(y22-y12))/2
    sizePX=sizeXpx+sizeYpx
    sizeXm=float(config["forklift positioning"]["lengthForklift"])
    sizeYm=float(config["forklift positioning"]["widthForklift"])
    sizeM=sizeXm+sizeYm
    #sizeM=float(config["QR-code"]["sizeM"])
    focus=int(config["forklift positioning"]["focus"])
    distance = sizeM * focus / sizePX
    return distance

def getAngle (distance,x,y,w,h):
    # Находим координаты точки центра прямогоульника
    x11 = x
    x12 = x+w
    x21 = x
    x22 = x+w
    y11 = y
    y12 = y
    y21 = y+h
    y22 = y+h
    coordinates=[int(((x11 + x12) / 2 + (x21 + x22) / 2) / 2), int(((y11 + y21) / 2 + (y12 + y22) / 2) / 2)]
    point11 = [x11, y11]
    point12 = [x12, y12]
    point21 = [x21, y21]
    point22 = [x22, y22]
    # Находим координаты точки центра изображения
    center=[int(config["forklift positioning"]["widthImage"])/2,int(config["forklift positioning"]["heightImage"])/2]
    disPoint11Center=np.hypot(abs(coordinates[0]-point11[0]),abs(coordinates[1] -point11[1]))
    disPoint12Center=np.hypot(abs(coordinates[0]-point12[0]),abs(coordinates[1] -point12[1]))
    disPoint21Center=np.hypot(abs(coordinates[0]-point21[0]),abs(coordinates[1] -point21[1]))
    disPoint22Center=np.hypot(abs(coordinates[0]-point22[0]),abs(coordinates[1] -point22[1]))
    ratioPXinM=(np.hypot((float(config["forklift positioning"]["lengthForklift"])),(float(config["forklift positioning"]["widthForklift"])))/2)/((disPoint11Center+disPoint12Center+disPoint21Center+disPoint22Center)/4)
    # Находим расстояние от центра изображения до центра прямоугольника (в пикселях)
    differenceX=abs(center[0]-coordinates[0])
    differenceY = abs(center[1] - coordinates[1])
    deviationPX=np.hypot(differenceX,differenceY)
    # Находим расстояние от центра изображения до центра прямоугольника (в метрах)
    deviationM=deviationPX*ratioPXinM
    # Находим угол
    sinAngle=deviationM/distance
    angle = (np.arcsin(sinAngle) / np.pi) * 180
    return angle



