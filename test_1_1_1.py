import sys
import sim as vrep
import numpy as np
import torch
import time
import cv2
import time

import imutils
import pandas as pd
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img
from collections import Counter
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import serial
print(serial.__version__)
#import webcolors
#vrep.simxFinish(-1) # just in case, close all opened connections
data_serial = serial.Serial(port='COM10', baudrate=9600)
clientID=vrep.simxStart('127.0.0.1',19999,True,True,5000,5)
print(clientID) # if 1, then we are connected.
if clientID!=-1:
    print ("Connected to remote API server")
else:
    print("Not connected to remote API server")
    sys.exit("Could not connect")
err_code,push_apple = vrep.simxGetObjectHandle(clientID,'push_joint', vrep.simx_opmode_blocking)
err_code,push_apple_1 = vrep.simxGetObjectHandle(clientID,'push_joint_1', vrep.simx_opmode_blocking)


err_code,ps_handle = vrep.simxGetObjectHandle(clientID,"Pro_0", vrep.simx_opmode_blocking)
err_code,ps_handle_1 = vrep.simxGetObjectHandle(clientID,"Pro_1", vrep.simx_opmode_blocking)


res, v1 = vrep.simxGetObjectHandle(clientID, 'v1', vrep.simx_opmode_oneshot_wait)

err, resolution, image = vrep.simxGetVisionSensorImage(clientID, v1, 0, vrep.simx_opmode_streaming)
time.sleep(1)

model = torch.hub.load(r'C:\Users\AUBAI\Desktop\Digital Logic\مشروع_تخرجي_أبي\new_yolo\venv\Lib\site-packages\yolov5','custom', 'C:\\Users\\AUBAI\\Desktop\\Digital Logic\\مشروع_تخرجي_أبي\\new_yolo\\venv\\Lib\\site-packages\\yolov5\\best_200_epoch.pt', source='local')
model.conf = 0.7  # NMS confidence threshold
iou = 0.5  # NMS IoU threshold
#model_2 = load_model('C:\\Users\\AUBAI\\Desktop\\Digital Logic\\مشروع_تخرجي_أبي\\new_yolo\\F_R_apples_vgg16.h5',compile=False)
#model_2.compile()
w=0
F_apples =0
R_apples = 0
detection_fresh=0
detection_rotten =0
#x0,y0,x1,y1,confi,cla = results.xyxy[0][1].numpy()
x_pos_max =[]
y_pos_max =[]
x_pos_min=[]
y_pos_min =[]
con_p =[]
ob_n =[]
color_val=[]
percent = []
percent_1=[]
res = {}
key_max = 0
key_max_1 = 0
cm_apple=[]
cv_apple_hsv =[]
update_HSV_cor_dot_x=[]
update_HSV_cor_dot_y=[]
error_diameter=[]
HSV_Dot =[]
HSV_Color=[]
def getValue_on():
    data_serial.write(b'1')
    esp32_data = data_serial.readline()
    return esp32_data
def getValue_off():
    data_serial.write(b'0')
    esp32_data = data_serial.readline()
    return esp32_data
def pega_centro():

    x1 = int(w / 2)

    y1 = int(h / 2)

    cx = x + x1

    cy = y + y1

def kmean_method () :
    clf = KMeans(n_clusters=2)
    color_labels = clf.fit_predict(img_c)
    center_colors = clf.cluster_centers_
    count = Counter(color_labels)
    ordered_colors = [center_colors[i] for i in count.keys()]
    hex_colors = [rgb_to_hex(ordered_colors[i]) for i in count.keys()]
    labels = list(color_labels)
    for i in range(len(center_colors)):
        j = labels.count(i)
        j = j / (len(labels))
        percent.append(round((j * 100), 1))
        percent_1.append(str(round((j * 100), 1)) + '%')
    for key in hex_colors:
        for value in percent:
            res[key] = value
            percent.remove(value)
            break
    key_max = max(res, key=lambda x: res[x])
    color_val.append(key_max)
    print(key_max)
    print(color_val)

    #key_max_1 = key_max.split('#')
    #print(key_max_1[1])
    # Printing resultant dictionary
    #print("Resultant dictionary is : " + str(res))
    #print(hex_colors)
def rgb_to_hex(rgb_color):
  hex_color = "#"
  for i in rgb_color:
    i = int(i)
    hex_color += ("{:02x}".format(i))
  return hex_color
def test_func_b():
    x_pos_max.append(x2)
    y_pos_max.append(y2)
    x_pos_min.append(x1)
    y_pos_min.append(y1)
    con_p.append(conf*100)
    ob_n.append(object_name)
    cm_apple.append((round(cm, 2)))
    cv_apple_hsv.append(round(cm_hsv, 2))
    HSV_Color.append(color)
    HSV_Dot.append(hue_value)
    error_diameter.append((str(abs(int(((round(cm, 2)-6.5)/6.5)*100))))+'%')
    df1 = pd.DataFrame({"X_max": x_pos_max,
                        "Y_max": y_pos_max,
                        "X_min": x_pos_min,
                        "Y_min": y_pos_min,
                         "conf": con_p,
                        "Diameter_yolo":cm_apple,
                        "Diameter_HSV" :cv_apple_hsv,
                        "color__Kmeans": color_val,
                        "Diameter_error": error_diameter,
                        "color_HSV": HSV_Color,
                        "hue_value": HSV_Dot,
                         "object name": ob_n })
    print(df1, "\n")
    print(color_val)
    df1.to_excel("yolo_count.xlsx")

while (vrep.simxGetConnectionId(clientID) != -1):
    # get image from vision sensor 'v0'
    err, resolution, frame = vrep.simxGetVisionSensorImage(clientID, v1, 0, vrep.simx_opmode_buffer)

    #scaledsize = resolution
    #img = np.array(frame, dtype=np.uint16)
    #img.resize([resolution[0]])
    #mg = imutils.rotate_bound(img, 180)
    img = np.array(frame, dtype=np.uint8)
    img.resize([resolution[0], resolution[1], 3])
    img = imutils.rotate_bound(img, 180)
    #frame = cv2.resize(img , (320, 320))
    frame = cv2.resize(img, (320, 320))

    #cv2.polylines(frame, [np.array(area, np.int32)], True, (0, 255, 0), 2)
    # Set range for red color and
    hsvFrame = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    # define mask
    red_lower = np.array([136, 87, 111], np.uint8)
    red_upper = np.array([180, 255, 255], np.uint8)
    red_mask = cv2.inRange(hsvFrame, red_lower, red_upper)





    # Morphological Transform, Dilation
    # for each color and bitwise_and operator
    # between imageFrame and mask determines
    # to detect only that particular color
    kernal = np.ones((5, 5), "uint8")

    # For red color
    red_mask = cv2.dilate(red_mask, kernal)
    res_red = cv2.bitwise_and(frame, frame,
                              mask=red_mask)



    # Creating contour to track red color
    contours, hierarchy = cv2.findContours(red_mask,
                                           cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)


    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if (area > 300):
            x, y, w, h = cv2.boundingRect(contour)
            imageFrame = cv2.drawContours(frame, contours, -1, (255, 0, 0), 2)
            imageFrame = cv2.rectangle(frame, (x, y),(x + w, y + h),(0, 0, 0), 5)
            cv2.putText(imageFrame, "Red Color", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (255, 255, 255))


    cv2.startWindowThread()
    #cv2.namedWindow("image")

    #cv2.imshow('image', img)
    results = model(frame,size=320)#,size=150
    imageFrame = np.squeeze(results.render())
    frame = cv2.cvtColor(imageFrame, cv2.COLOR_BGR2RGB)
    area = [(150, 0), (160, 0), (160, 290), (150, 290)]
    #area = [(340, 0), (350, 0), (350, 600), (340, 600)]
    cv2.polylines(frame, [np.array(area, np.int32)], True, (0, 0, 0), 2)
    for index, row in results.pandas().xyxy[0].iterrows():
        x1 = int(row['xmin'])
        y1 = int(row['ymin'])
        x2 = int(row['xmax'])
        y2 = int(row['ymax'])
        conf= float(row['confidence'])
        object_name = (row['name'])
        c_x = int(round((x2 + x1)/2))
        c_y = int(round((y2 + y1)/2))
        width_apple_for_yolo = x2 - x1
        width_apple_for_HSV_mask = w
        ratio_px_mm = 130/65
        mm = width_apple_for_yolo/ratio_px_mm
        mm_hsv = width_apple_for_HSV_mask/ratio_px_mm
        cm_hsv = mm_hsv/10
        cm = mm/10
        '''apple_vgg16 = np.copy(imageFrame[y1:y2, x1:x2])
        apple_vgg16 = cv2.resize(apple_vgg16, (300, 300))
        apple_vgg16 = apple_vgg16.astype("float") / 255.0
        apple_vgg16 = img_to_array(apple_vgg16)
        apple_vgg16 = np.expand_dims(apple_vgg16, axis=0)
        # apply gender detection on frame
        conf_2 = model_2.predict(apple_vgg16, batch_size=10)[0]  # model.predict return a 2D matrix, ex: [[9.9993384e-01 7.4850512e-05]]
        # get label with max accuracy
        if conf_2[0] > 0.5:
            label = " VGG16_rotten"
        else:
            label = " VGG16_fresh"
        cv2.putText(frame, label, (x1 + 50, y1 - 25), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)'''
        hsvFrame_1 = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        pixel_center = hsvFrame_1[c_y, c_x]
        hue_value = pixel_center[0]
        cv2.circle(frame, (c_x, c_y), 4, (255, 255, 255), -1)

        color = "Undefined"
        if hue_value < 0:
            color = "no color"
        elif hue_value < 22:
            color = "B_RED"
        elif hue_value < 33:
            color = "YELLOW"
        elif hue_value < 78:
            color = "GREEN"
        elif hue_value < 131:
            color = "BLACK"
        elif hue_value < 170:
            color = "VIOLET"
        else:
            color = "RED"
        cv2.putText(frame, color, (x1 -60, y1 - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 2)
        cv2.putText(frame, "{} CM ".format(round(cm,2)), (x1 + 20, y1-25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        apple_crop = np.copy(imageFrame[y1:y2, x1:x2])
        img_c = cv2.resize(apple_crop, (900, 600), interpolation=cv2.INTER_AREA)
        img_c = img_c.reshape(img_c.shape[0] * img_c.shape[1], 3)
        #results = cv2.pointPolygonTest(np.array(area, np.int32), (x2, y1), False)
        results = cv2.pointPolygonTest(np.array(area, np.int32), (c_x, c_y), False)
        if results >= 0 and object_name == "fresh_apple":
            F_apples += 1
            detection_fresh = "f"
            inputUser = 'on'
            print("number of fresh apples = ",F_apples)
            kmean_method()
            test_func_b()
            #time.sleep(0.2)
        elif results >= 0 and object_name == "rotten_apple":
            R_apples += 1
            detection_rotten = "r"
            inputUser = 'off'
            print(getValue_on())
            print("number of Rotten apples = ",R_apples)
            kmean_method()
            test_func_b()
            #time.sleep(0.2)
    #cv2.imshow('image_2', frame )
    #cv2.imshow('image_3', red_mask)
    #cv2.imshow('image_3', res_red)
        cv2.putText(frame, "color =" + color +"  "+"  hue = "+str(hue_value), (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255),1)
        cv2.putText(frame,"Diameter="+ "{} CM ".format(round(cm, 2)), (15, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.8,(0, 0, 255), 1)
    cv2.putText(frame, "fresh= "+str(F_apples), (15, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255))
    cv2.putText(frame, "rotten= "+str(R_apples), (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255))

    err_code, detectionState, detectedPoint, detectedObjectHandle,detectedSurfaceNormalVector = vrep.simxReadProximitySensor(clientID,ps_handle, vrep.simx_opmode_streaming)
    err_code_1, detectionState_1, detectedPoint_1, detectedObjectHandle_1, detectedSurfaceNormalVector_1 = vrep.simxReadProximitySensor(clientID, ps_handle_1, vrep.simx_opmode_streaming)
    if detectionState == 1 and detection_fresh == "f" and inputUser == 'on':
         #user_input = 'f'
         #print(getValue_on())
         #byte_command = str.encode(user_input)
         #arduino.writelines(user_input)  # send a byte
         #time.sleep(0.5)  # wait 0.5 seconds
         #print("detection fresh apple")
         err_code = vrep.simxSetJointTargetVelocity(clientID, push_apple, 0.3, vrep.simx_opmode_streaming)
         time.sleep(4)
         err_code = vrep.simxSetJointTargetVelocity(clientID, push_apple, -0.3, vrep.simx_opmode_streaming)
         time.sleep(4)
         err_code = vrep.simxSetJointTargetVelocity(clientID, push_apple, 0, vrep.simx_opmode_streaming)
         detection_fresh = 0

    if detectionState_1 ==1 and detection_rotten == "r" and inputUser == 'off' :
        #print(getValue_off())

        err_code = vrep.simxSetJointTargetVelocity(clientID, push_apple_1, 0.3, vrep.simx_opmode_streaming)
        time.sleep(3)
        err_code = vrep.simxSetJointTargetVelocity(clientID, push_apple_1, -0.3, vrep.simx_opmode_streaming)
        time.sleep(3)
        err_code = vrep.simxSetJointTargetVelocity(clientID, push_apple_1, 0, vrep.simx_opmode_streaming)
        detection_rotten = 0

    #elif detectionState == 0 :
        #print("no detection ")

    #cv2.line(frame, (150, 320), (150, 0), (0, 0, 0), 3)
    cv2.imshow('image_1',frame)
    cv2.imshow('imageFrame', imageFrame)
    res_red = cv2.cvtColor(res_red, cv2.COLOR_BGR2RGB)
    cv2.imshow('res_red', res_red)
    cv2.imshow('red_mask', red_mask)

    #frame =cv2.resize(frame , (640, 480))
    #frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)0000

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        cv2.destroyAllwindows()


'''cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH , 320) 
cap.set(cv2.CAP_PROP_FRAME_HEIGHT , 240)
while (cap.isOpened()):
    # read the frame from the video file
    ret,  imageFrame = cap.read()
    # if the frame was captured successfully
    if ret == True:
        cv2.imshow("Multiple Color Detection in Real-TIme", imageFrame)
    key = cv2.waitKey(20)
    # if key q is pressed then break
    if key == 113:
        break

    # Closes video file or capturing device.
cap.release()
# finally destroy/close all open windows
cv2.destroyAllWindows()'''
'''import cv2


cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
w=cap.set(cv2.CAP_PROP_FRAME_WIDTH , 640)
h=cap.set(cv2.CAP_PROP_FRAME_HEIGHT , 320)
c_1 = int(h/ 2)
c_2 = int(w/ 2)
while True:
    success,frame = cap.read()
    cv2.circle(frame, (320,160), 5, (0, 0, 255), 3)
    hsvFrame = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    pixel_center = hsvFrame[c_1, c_2 ]
    H_value = pixel_center[0]
    S_value = pixel_center[1]
    V_value = pixel_center[2]
    print("h value is = ",H_value)
    print("S value is = ", S_value)
    print("V value is = ", V_value)
    # define mask


    red_lower = np.array([50, 30, 90], np.uint8)
    red_upper = np.array([150, 100, 200], np.uint8)
    red_mask = cv2.inRange(hsvFrame, red_lower, red_upper)

    # Morphological Transform, Dilation
    # for each color and bitwise_and operator
    # between imageFrame and mask determines
    # to detect only that particular color
    kernal = np.ones((5, 5), "uint8")

    # For red color
    red_mask = cv2.dilate(red_mask, kernal)
    res_red = cv2.bitwise_and(frame, frame,
                              mask=red_mask)

    # Creating contour to track red color
    contours, hierarchy = cv2.findContours(red_mask,
                                           cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)

    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if (area > 300):
            x, y, w, h = cv2.boundingRect(contour)
            imageFrame = cv2.rectangle(frame, (x, y),
                                       (x + w, y + h),
                                       (0, 0, 0), 5)
            cv2.putText(imageFrame, "Red Color", (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (255, 255, 255))

    cv2.startWindowThread()
    # cv2.namedWindow("image")

    # cv2.imshow('image', img)
    #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cv2.imshow('image_2', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()'''
