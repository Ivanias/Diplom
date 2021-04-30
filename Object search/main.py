import cv2
import numpy as np
import func

net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
with open("coco.names.txt","r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

image=cv2.imread("Car1.png")
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
        print("Координаты центра: ",(x+x+w)/2,(y+y+h)/2)
        distance=func.getDistance(x,y,w,h)
        print("Дистанция: ",distance)
        angle=func.getAngle(distance,x,y,w,h)
        print("Угол: ", angle)
        cv2.circle(image, (int((x+x+w)/2),int((y+y+h)/2)), 3, (0, 255, 0), -1)
        #cv2.putText(image, label, (x, y + 30), font, 2, (0, 255, 0), 3)

cv2.imshow("Results", image)
cv2.waitKey(0)
cv2.destroyAllWindows()