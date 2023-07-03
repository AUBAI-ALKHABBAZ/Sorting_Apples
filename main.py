import numpy as np
import torch
import cv2
import pandas as pd
import requests
from tensorflow.keras.preprocessing.image import img_to_array
#from tensorflow.keras.utils import img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img
import time
from tracker import *
import tensorflow as tf
#import torch_directml
import timeit
#import tensorflow.compat.v1 as tf
import os
import os.path
#import tensorflow.compat.v1 as tf
from collections import Counter
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import serial
from tkinter import *
from PIL import Image, ImageTk

data_serial = serial.Serial(port='COM10', baudrate=9600)
'''gpus = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    if tf.test.gpu_device_name():
      print('Default GPU Device:{}'.format(tf.test.gpu_device_name()))
    else:
      print("Please install GPU version of TF")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)
# To read image from disk, we use
# cv2.imread function, in below method,
model_2 = load_model('C:\\Users\\AUBAI\\Desktop\\Digital Logic\\مشروع_تخرجي_أبي\\new_yolo\\red_vs_green_apples.h5',compile=False)
model_2.compile()

img = load_img("C:\\Users\\AUBAI\\Desktop\\green_apples.jfif", target_size=(150, 150))
img_2 = load_img("C:\\Users\\AUBAI\\Desktop\\yellow_apple.jpg", target_size=(150, 150))
path = r"C:\\Users\\AUBAI\\Desktop\\green_apples.jfif"
#print("Num GPUs Available: ", len(tensorflow.config.list_physical_devices('GPU')))

# Using cv2.imread() method
# Using 0 to read image in grayscale mode
img_1 = cv2.imread(path, cv2.COLOR_BGR2RGB)



x = img_to_array(img)
x /= 255
x = np.expand_dims(x, axis=0)

images = np.vstack([x])
classes = model_2.predict(images, batch_size=10)
print(classes[0])
if classes[0] > 0.5:
    print(" is a red")
    # Displaying the image
    cv2.putText(img_1, " RED", (30, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
    cv2.imshow('image', img_1)
else:
    print(" is a green")
    # Displaying the image
    cv2.putText(img_1, " GREEN", (30, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
    cv2.imshow('image', img_1)
cv2.waitKey(0)
cv2.destroyAllWindows()'''
'''gpus = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    if tf.test.gpu_device_name():
      print('Default GPU Device:{}'.format(tf.test.gpu_device_name()))
    else:
      print("Please install GPU version of TF")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)'''
#torch.hub.load('ultralytics/yolov5')

#model = torch.hub.load('ultralytics/yolov5', 'custom', 'C:\\Users\\AUBAI\\Desktop\\Digital Logic\\مشروع_تخرجي_أبي\\new_yolo\\venv\\Lib\\site-packages\\yolov5\\best_200_epoch.pt') #best_fresh_rotten_apple_bananna
model = torch.hub.load(r'C:\Users\AUBAI\Desktop\Digital Logic\مشروع_تخرجي_أبي\new_yolo\venv\Lib\site-packages\yolov5','custom', 'C:\\Users\\AUBAI\\Desktop\\Digital Logic\\مشروع_تخرجي_أبي\\new_yolo\\best_6000_img.pt', source='local')
#model.conf = 0.50  # NMS confidence threshold
model.conf = 0.70 # NMS confidence threshold
iou = 0.5     # NMS IoU threshold
#model_2 = load_model('C:\\Users\\AUBAI\\Desktop\\Digital Logic\\مشروع_تخرجي_أبي\\new_yolo\\red_vs_green_apples.h5',compile=False)
#model_2.compile()
#model_2 = load_model('C:\\Users\\AUBAI\\Desktop\\Digital Logic\\مشروع_تخرجي_أبي\\new_yolo\\F_R_apples_vgg16.h5',compile=False)
#model_2.compile()
#agnostic = False  # NMS class-agnostic
#multi_label = False  # NMS multiple labels per box
#classes = None  # (optional list) filter by class, i.e. = [0, 15, 16] for COCO persons, cats and dogs
#max_det = 1000  # maximum number of detections per image
#amp = False  # Automatic Mixed Precision (AMP) inference
#model.conf = 0.35  # confidence threshold (0-1)
# load model
#model_2 = load_model('C:\\Users\\AUBAI\\Desktop\\Digital Logic\\مشروع_تخرجي_أبي\\new_yolo\\class_6_fresh_rotten.h5',compile=False)
#model_2.compile()
#img = cv2.imread('C:\\Users\\AUBAI\\Desktop\\Digital Logic\\مشروع_تخرجي_أبي\\new_yolo\\venv\\Lib\\site-packages\\yolov5\\data\\images\\appl_test_2.jpeg')
#result = model(img)
#classes = ['red','green']
#print(result)
#classes = ['Fresh Apple','Fresh Banana','Fresh Orange','Rotten Apple','Rotten Banana','Rotten Orange']

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
HSV_Dot =[]
HSV_Color=[]
key_max = 0
key_max_1 = 0
cm_apple=[]
cv_apple_hsv =[]
update_HSV_cor_dot_x=[]
update_HSV_cor_dot_y=[]
error_diameter=[]

def getValue_on():
    data_serial.write(b'1')
    esp32_data = data_serial.readline()
    return esp32_data
def getValue_off():
    data_serial.write(b'0')
    esp32_data = data_serial.readline()
    return esp32_data
def kmean_method () :
    apple_crop = np.copy(imageFrame_1[y1:y2, x1:x2])
    img_c = cv2.resize(apple_crop, (640, 640), interpolation=cv2.INTER_AREA)
    img_c = img_c.reshape(img_c.shape[0] * img_c.shape[1], 3)
    centers_3 = np.array([[240, 30, 30],
                          [255, 30, 30],
                          [210, 30, 30],])

    clf = KMeans(n_clusters=3 ).fit(centers_3)
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
    #key_max = max(res, key=lambda x: res[x])
    key_max = min(res, key=lambda x: res[x])
    color_val.append(key_max)
    print("color list \n",res)
    print(key_max)
    print(color_val)
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
    update_HSV_cor_dot_x.append(c_x)
    update_HSV_cor_dot_y.append(c_y)
    con_p.append(conf*100)
    ob_n.append(object_name)
    cm_apple.append((round(cm, 2)))
    HSV_Color.append(color)
    HSV_Dot.append(hue_value)
    error_diameter.append((str(abs(int(((round(cm, 2)-6.5)/6.5)*100))))+'%')

    #cv_apple_hsv.append(round(cm_hsv, 2))
    df1 = pd.DataFrame({"X_max": x_pos_max,
                        "Y_max": y_pos_max,
                        "X_min": x_pos_min,
                        "Y_min": y_pos_min,
                        "Y_mid_dot": update_HSV_cor_dot_y,
                        "X_mid_dot": update_HSV_cor_dot_x,
                         "conf": con_p,
                        "Diameter_yolo":cm_apple,
                        "Diameter_error": error_diameter,
                        "color_Kmeans": color_val,
                        "color_HSV": HSV_Color,
                        "hue_value": HSV_Dot,

                         "object name_yolo": ob_n })
    print(df1, "\n")
    print(color_val)
    df1.to_excel("yolo_count.xlsx")
def pega_centro(x, y, w, h):

    x1 = int(w / 2)

    y1 = int(h / 2)

    cx = x + x1

    cy = y + y1

    return cx, cy
detec = []
font = cv2.FONT_HERSHEY_SIMPLEX
threshold = 0.85
def getCalssName(classNo):
    if   classNo == 0: return 'Fresh Apple'
    elif classNo == 1: return 'Fresh Banana'
    elif classNo == 2: return 'Fresh Orange'
    elif classNo == 3: return 'Rotten Apple'
    elif classNo == 4: return 'Rotten Banana'
    elif classNo == 5: return 'Rotten Orange'
def grayscale(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    return img
def equalize(img):
    img =cv2.equalizeHist(img)
    return img
def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    img = img/255
    return img
#classes = ['Fresh Apple','Fresh Banana','Fresh Orange','Rotten Apple','Rotten Banana','Rotten Orange']
#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

cap.set(cv2.CAP_PROP_FRAME_WIDTH , 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT , 320)




#tracker = Tracker()
#address = 'http://192.168.1.101:8080/video'
#model.conf = 0.7  # confidence threshold (0-1)
# used to record the time when we processed last frame
prev_frame_time = 0

# used to record the time at which we processed current frame
new_frame_time = 0
#area = [(320,0),(350,0),(350,480),(320,480)]
#c = set()
'''first point(x=320,y=0)
second point (x=350,y=0)
third point(x=350,y=420)
fourth point(x=320 , y=420)'''
prev_move = None

while True:
    ret, imageFrame= cap.read()
    if not ret:
        break

    # continue
    #imageFrame = cv2.resize(imageFrame, (640, 640))


    '''apple_crop = np.copy(imageFrame)
    apple_crop = cv2.resize(apple_crop, (20, 20))
    apple_crop = apple_crop.astype("float") / 255.0
    apple_crop = img_to_array(apple_crop)
    apple_crop = np.expand_dims(apple_crop, axis=0)
    #img = img.reshape(1, 150, 150, 1)
    cv2.putText(imageFrame, "CLASS: ", (20, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(imageFrame, "P: ", (20, 75), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
    # PREDICT IMAGE
    predictions = model_2.predict(apple_crop)
    #classIndex = model_2.predict_classes(apple_crop)
    #classIndex = (model_2.predict(apple_crop) > 0.7).astype("int32")
    classIndex = model_2.predict(apple_crop)
    probabilityValue = np.amax(predictions)
    print(classIndex,probabilityValue)
    if probabilityValue > threshold and classIndex[0][0]>= 0.85 :
                cv2.putText(imageFrame, "fresh apple ", (100, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.putText(imageFrame, str(probabilityValue), (100, 75), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
    elif probabilityValue > threshold and classIndex[0][3] >= 0.9 :
                cv2.putText(imageFrame, "rotten  apple ", (100, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.putText(imageFrame, str(probabilityValue), (100, 75), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
    else :
        cv2.putText(imageFrame, "no predaction  ", (100, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)

    results = model(imageFrame,size=320)#,size=150
    imageFrame = np.squeeze(results.render())
    cv2.imshow("Result", imageFrame)'''

    '''cv2.rectangle(imageFrame, (100, 100), (300, 300), (255, 255, 255), 2)
    roi = imageFrame[100:300, 100:300]
    # extract the region of image within the user rectangle
    face_crop = cv2.resize(roi, (150, 150))
    face_crop = face_crop.astype("float") / 255.0
    face_crop = img_to_array(face_crop)
    face_crop = np.expand_dims(face_crop, axis=0)

    # apply gender detection on face
    conf = model_2.predict(face_crop)[0]  # model.predict return a 2D matrix, ex: [[9.9993384e-01 7.4850512e-05]]

    # get label with max accuracy
    idx = np.argmax(conf)
    label = classes[idx]

    label = "{}: {:.2f}%".format(label, conf[idx] * 100)


    # write label and confidence above face rectangle
    cv2.putText(imageFrame, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 0), 2)'''
    new_frame_time = time.time()
    '''cx = 320
    cy = 160
    hsvFrame = cv2.cvtColor(imageFrame, cv2.COLOR_BGR2HSV)
    pixel_center = hsvFrame[cy, cx]
    hue_value = pixel_center[0]
    color = "Undefined"
    if hue_value < 5:
        color = "no color"
    elif hue_value < 78:
        color = "GREEN"
    else:
        color = "RED"
    pixel_center_bgr = imageFrame[cy, cx]
    b, g, r = int(pixel_center_bgr[0]), int(pixel_center_bgr[1]), int(pixel_center_bgr[2])
    #cv2.rectangle(imageFrame, (cx - 220, 10), (cx + 200, 120), (255, 255, 255), -1)
    #cv2.putText(imageFrame, color, (cx - 200, 100), 0, 3, (b, g, r), 5)
    cv2.circle(imageFrame, (cx, cy), 5, (25, 25, 25), 3)'''
    # fps will be number of frame processed in given time frame
    # since their will be most of time error of 0.001 second
    # we will be subtracting it to get more accurate result
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    # converting the fps into integer
    fps = int(fps)

    # converting the fps to string so that we can display it on frame
    # by using putText function
    fps = str(fps)

    #apple_crop = np.copy(imageFrame)
    #apple_crop = cv2.resize(apple_crop, (150, 150))
    #apple_crop = apple_crop.astype("float") / 255.0
    #apple_crop = img_to_array(apple_crop)
    #apple_crop = np.expand_dims(apple_crop, axis=0)
    # apply gender detection on frame
    #conf_2 = model_2.predict(apple_crop)  # model.predict return a 2D matrix, ex: [[9.9993384e-01 7.4850512e-05]]
   # classes = model_2.predict(apple_crop, batch_size=10)
   # label = np.where(classes[0] > 0.5, 1, 0)
   # print(label)
    # get label with max accuracy
    #result1 = conf_2[0]
    #for i in range(6):

        #if result1[i] == 1.:
            #break;
    #prediction = classes[i]
    #print(prediction)
    #cv2.polylines(imageFrame, [np.array(area, np.int32)], True, (0, 255, 0), 2)
    #results = model(imageFrame,size=320)#,size=150
    #imageFrame = np.squeeze(results.render())
    #print(results.pandas().xyxy[0])
    #cv2.line(imageFrame, (320, 640), (320, 0), (255, 255, 255), 2)
    '''points = []
    for index, row in results.pandas().xyxy[0].iterrows():
        x1 = int(row['xmin'])
        y1 = int(row['ymin'])
        x2 = int(row['xmax'])
        y2 = int(row['ymax'])
        n = (row['name'])

        if 'fresh_apple' in n:
            points.append([x1, y1, x2, y2])
            points.append([x1, y1, x2, y2])
            #cv2.rectangle(imageFrame, (x1, y1), (x2, y2), (255, 0, 255), 2)
            #cv2.putText(imageFrame, str(n), (x1, y1), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
    boxes_id = tracker.update(points)
    #print(boxes_id)
    for box_id in boxes_id:
        x_id , y_id , w_id , h_id , idd = box_id
        cv2.rectangle(imageFrame,(x_id,y_id),(w_id,h_id),(255,0,255),2)
        cv2.putText(imageFrame,str(idd),(x_id,y_id),cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),2)
        results = cv2.pointPolygonTest(np.array(area, np.int32), (w_id, h_id), False)
        if results>= 0 :
            c.add(idd)
    a = len(c)
    cv2.putText(imageFrame, 'fresh apples=' + str(a), (50, 65), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 2)'''
    '''for index, row in results.pandas().xyxy[0].iterrows():
        x1 = int(row['xmin'])
        y1 = int(row['ymin'])
        x2 = int(row['xmax'])
        y2 = int(row['ymax'])
        n = (row['name'])
        center = pega_centro(x1, y1, x2, y2)
        detec.append(center)
        cv2.circle(imageFrame, center, 4, (255, 255, 255), -1)
    for (x, y) in detec:

        if x < (320) and x > (315) and 'fresh_apple' == n and color =="RED":
            F_apples_red +=1
            cv2.line(imageFrame, (320, 640), (320, 0), (100, 50, 255), 2)
            detec.remove((x, y))
            print("No  of fresh  red apples  = : " + str(F_apples_red))
        elif x < (320) and x > (315) and 'fresh_apple' == n and  color =="GREEN":
            F_apples_green +=1
            cv2.line(imageFrame, (320, 640), (320, 0), (100, 50, 255), 2)
            detec.remove((x, y))
            print("No  of fresh  green apples   = : " + str(F_apples_green))
        elif   x < (320) and x > (315) and 'rotten_apple' == n :
            R_apples +=1
            cv2.line(imageFrame, (320, 640), (320, 0), (100, 50, 255), 2)
            detec.remove((x, y))
            print("No  of Rotten apples  = : " + str(R_apples))'''
    '''points = []
    for index, row in results.pandas().xyxy[0].iterrows():
        x1 = int(row['xmin'])
        y1 = int(row['ymin'])
        x2 = int(row['xmax'])
        y2 = int(row['ymax'])
        n = (row['name'])
        points.append([x1, y1, x2, y2])
        apple_crop = np.copy(imageFrame[y1:y2, x1:x2])
        apple_crop = cv2.resize(apple_crop, (150, 150))
        apple_crop = apple_crop.astype("float") / 255.0
        apple_crop = img_to_array(apple_crop)
        apple_crop = np.expand_dims(apple_crop, axis=0)
        # apply gender detection on frame
        conf_2 = model_2.predict(apple_crop)[0]  # model.predict return a 2D matrix, ex: [[9.9993384e-01 7.4850512e-05]]
        # get label with max accuracy

        if conf_2 < 0.7:
            label = "red"
            cv2.putText(imageFrame, label, (x1 - 50, y1 - 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 0), 2)
        else:
            label = "green"
            cv2.putText(imageFrame, label, (x1 - 50, y1 - 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 0), 2)'''
    '''points = []
    for index, row in results.pandas().xyxy[0].iterrows():
        x1 = int(row['xmin'])
        y1 = int(row['ymin'])
        x2 = int(row['xmax'])
        y2 = int(row['ymax'])
        n = (row['name'])

        if 'fresh_apple' in n:
            points.append([x1, y1, x2, y2])
            points.append([x1,y1,x2,y2])
            cv2.rectangle(imageFrame,(x1,y1),(x2,y2),(255,0,255),2)
            cv2.putText(imageFrame,str(n),(x1,y1),cv2.FONT_HERSHEY_PLAIN,2,(255,0,0),2)
        boxes_id = tracker.update(points)
        print(boxes_id)
    points = []
    for index, row in results.pandas().xyxy[0].iterrows():
        x1 = int(row['xmin'])
        y1 = int(row['ymin'])
        x2 = int(row['xmax'])
        y2 = int(row['ymax'])
        n = (row['name'])

        if 'fresh_apple' in n:
            points.append([x1, y1, x2, y2])
            apple_crop = np.copy(imageFrame[y1:y2, x1:x2])
            apple_crop = cv2.resize(apple_crop, (150, 150))
            apple_crop = apple_crop.astype("float") / 255.0
            apple_crop = img_to_array(apple_crop)
            apple_crop = np.expand_dims(apple_crop, axis=0)
            # apply gender detection on frame
            conf_2 = model_2.predict(apple_crop)[0]  # model.predict return a 2D matrix, ex: [[9.9993384e-01 7.4850512e-05]]
            # get label with max accuracy

            if conf_2  < 0.4 :
                label = "red"
            else:
                label = "green"


            cv2.rectangle(imageFrame,(x1,y1),(x2,y2),(255,255,0),2)
            cv2.putText(imageFrame,str(n),(x1,y1),cv2.FONT_HERSHEY_PLAIN,2,(255,255,0),2)
            cv2.putText(imageFrame, label, (x1-50, y1-50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 0), 2)'''

    #cv2.putText(imageFrame, fps, (7, 120), cv2.FONT_HERSHEY_PLAIN, 3, (100, 255, 0), 5)
    cv2.putText(imageFrame, "fps = "+str(fps), (15, 100), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 5)
    #cv2.line(imageFrame, (320, 640), (320, 0), (255, 255, 255), 2)
    results = model(imageFrame, size=640)  # ,size=150
    imageFrame_1 = np.squeeze(results.render())
    area = [(340,0),(350,0),(350,480),(340,480)]
    points = []
    cv2.polylines(imageFrame_1, [np.array(area, np.int32)], True, (255, 0, 0), 2)
    for index, row in results.pandas().xyxy[0].iterrows():
        x1 = int(row['xmin'])
        y1 = int(row['ymin'])
        x2 = int(row['xmax'])
        y2 = int(row['ymax'])
        conf = float(row['confidence'])
        object_name = (row['name'])
        c_x = int(round((x2 + x1)/2))
        c_y = int(round((y2 + y1)/2))

        width_apple_for_yolo = x2 - x1

        ratio_px_mm = 200 / 65
        mm = width_apple_for_yolo / ratio_px_mm
        cm = mm / 10
        '''apple_vgg16 = np.copy(imageFrame[y1:y2, x1:x2])
        apple_vgg16 = cv2.resize(apple_vgg16, (300, 300))
        apple_vgg16 = apple_vgg16.astype("float") / 255.0
        apple_vgg16 = img_to_array(apple_vgg16)
        apple_vgg16 = np.expand_dims(apple_vgg16, axis=0)
        # apply gender detection on frame
        conf_2 = model_2.predict(apple_vgg16, batch_size=10)[0]  # model.predict return a 2D matrix, ex: [[9.9993384e-01 7.4850512e-05]]
        # get label with max accuracy
        if conf_2[0] > 0.5:
            label = " VGG16T_rotten"
        else:
            label = " VGG16T_fresh"
        cv2.putText(imageFrame_1, label, (x1 + 80, y1 - 25), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)'''

        hsvFrame = cv2.cvtColor(imageFrame, cv2.COLOR_BGR2HSV)
        pixel_center = hsvFrame[c_y, c_x]
        hue_value = pixel_center[0]

        #S_value = pixel_center[1]
        #V_value = pixel_center[2]
        #print(hue_value ,'\n')
        #print(S_value, '\n')
        #print(V_value, '\n')
        cv2.circle(imageFrame_1, (c_x, c_y), 4, (255, 255, 255), -1)

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
        cv2.putText(imageFrame_1, color, (x1 -60, y1 - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 255), 2)
        cv2.putText(imageFrame_1, "{} CM ".format(round(cm, 2)), (x1 + 20, y1 - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 255), 2)

        result = cv2.pointPolygonTest(np.array(area, np.int32), (c_x, c_y), False)

        if result >= 0 and object_name == "Fresh Apple":
            F_apples += 1

            detection_fresh = "f"
            print("number of fresh apples = ", F_apples)
            #inputUser = 'on'
            #print(getValue_on())
            kmean_method()
            test_func_b()
            #time.sleep(0.5)
        elif result >= 0 and object_name == "Rotten":

            print(getValue_on())

            R_apples += 1
            detection_rotten = "r"
            print("number of Rotten apples = ", R_apples)

            inputUser = 'on'

            #data_serial.close()
            kmean_method()
            test_func_b()
            #time.sleep(0.9)
        cv2.putText(imageFrame_1, "color =" + color +"  "+"  hue = "+str(hue_value), (200, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255),2)
        cv2.putText(imageFrame_1,"Diameter="+ "{} CM ".format(round(cm, 2)), (200, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8,(0, 0, 255), 2)
    cv2.putText(imageFrame_1, "fresh= "+str(F_apples), (15, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255),2)
    cv2.putText(imageFrame_1, "rotten= "+str(R_apples), (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255),2)


    cv2.imshow("FRAME", imageFrame_1)
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()

cv2.destroyAllWindows()
'''import numpy as np
import cv2
import tkinter as tk
from PIL import Image, ImageTk

#Set up GUI
window = tk.Tk()  #Makes main window
window.wm_title("Digital Microscope")
window.config(background="#FFFFFF")

#Graphics window
imageFrame = tk.Frame(window, width=600, height=500)
imageFrame.grid(row=0, column=0, padx=500, pady=300)

#Capture video frames
lmain = tk.Label(imageFrame)
lmain.grid(row=0, column=0)
cap = cv2.VideoCapture(0)
def show_frame():
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    lmain.after(10, show_frame)



#Slider window (slider controls stage position)
sliderFrame = tk.Frame(window, width=600, height=100)
sliderFrame.grid(row = 600, column=0, padx=10, pady=2)


show_frame()  #Display 2
window.mainloop()'''  #Starts GUI
'''import cv2
import numpy as np

import requests


INFO SECTION
- if you want to monitor raw parameters of ESP32CAM, open the browser and go to http://192.168.x.x/status
- command can be sent through an HTTP get composed in the following way http://192.168.x.x/control?var=VARIABLE_NAME&val=VALUE (check varname and value in status)


# ESP32 URL
#URL = "http://192.168.123.35"
URL = "http://192.168.123.42:8080"
AWB = True

# Face recognition and opencv setup
cap = cv2.VideoCapture(URL + ":81/stream")

def set_resolution(url: str, index: int=1, verbose: bool=False):
    try:
        if verbose:
            resolutions = "10: UXGA(1600x1200)\n9: SXGA(1280x1024)\n8: XGA(1024x768)\n7: SVGA(800x600)\n6: VGA(640x480)\n5: CIF(400x296)\n4: QVGA(320x240)\n3: HQVGA(240x176)\n0: QQVGA(160x120)"
            print("available resolutions\n{}".format(resolutions))

        if index in [10, 9, 8, 7, 6, 5, 4, 3, 0]:
            requests.get(url + "/control?var=framesize&val={}".format(index))
        else:
            print("Wrong index")
    except:
        print("SET_RESOLUTION: something went wrong")

def set_quality(url: str, value: int=1, verbose: bool=False):
    try:
        if value >= 10 and value <=63:
            requests.get(url + "/control?var=quality&val={}".format(value))
    except:
        print("SET_QUALITY: something went wrong")

def set_awb(url: str, awb: int=1):
    try:
        awb = not awb
        requests.get(url + "/control?var=awb&val={}".format(1 if awb else 0))
    except:
        print("SET_QUALITY: something went wrong")
    return awb

if __name__ == '__main__':
    set_resolution(URL, index=8)

    while True:
        if cap.isOpened():
            ret, frame = cap.read()

            if ret:

                cv2.imshow("frame", frame)
                key = cv2.waitKey(1)

            if key == ord('r'):
                idx = int(input("Select resolution index: "))
                set_resolution(URL, index=idx, verbose=True)

            elif key == ord('q'):
                val = int(input("Set quality (10 - 63): "))
                set_quality(URL, value=val)

            elif key == ord('a'):
                AWB = set_awb(URL, AWB)

            elif key == 27:
                break

    cv2.destroyAllWindows()
    cap.release()'''

# Python code for Multiple Color Detection


'''import numpy as np
import cv2
import torch
# Capturing video through webcam
model = torch.hub.load('ultralytics/yolov5', 'custom', 'C:\\Users\\AUBAI\\Desktop\\Digital Logic\\مشروع_تخرجي_أبي\\new_yolo\\venv\\Lib\\site-packages\\yolov5\\best_200_epoch.pt')
model.conf = 0.7  # confidence threshold (0-1)
webcam = cv2.VideoCapture(0)
webcam.set(3,640)
webcam.set(4,640)
def empty(a):
    pass

cv2.namedWindow("HSV")
cv2.resizeWindow("HSV",640,240)
cv2.createTrackbar("HUE Min","HSV",0,179,empty)
cv2.createTrackbar("HUE Max","HSV",179,179,empty)
cv2.createTrackbar("SAT Min","HSV",0,255,empty)
cv2.createTrackbar("SAT Max","HSV",255,255,empty)
cv2.createTrackbar("VALUE Min","HSV",0,255,empty)
cv2.createTrackbar("VALUE Max","HSV",255,255,empty)
# Start a while loop
while (1):

    # Reading the video from the
    # webcam in image frames
    _, imageFrame = webcam.read()
    imageFrame = cv2.resize(imageFrame, (640, 640))
    results = model(imageFrame)
    imageFrame = np.squeeze(results.render())
    # Convert the imageFrame in
    # BGR(RGB color space) to
    # HSV(hue-saturation-value)
    # color space
    hsvFrame = cv2.cvtColor(imageFrame, cv2.COLOR_BGR2HSV)
    h_min = cv2.getTrackbarPos("HUE Min","HSV")
    h_max = cv2.getTrackbarPos("HUE Max", "HSV")
    s_min = cv2.getTrackbarPos("SAT Min", "HSV")
    s_max = cv2.getTrackbarPos("SAT Max", "HSV")
    v_min = cv2.getTrackbarPos("VALUE Min", "HSV")
    v_max = cv2.getTrackbarPos("VALUE Max", "HSV")
    red_lower = np.array([h_min , s_min,v_min], np.uint8)
    red_upper = np.array([h_max, s_max, v_max], np.uint8)
    red_mask = cv2.inRange(hsvFrame, red_lower, red_upper)
    kernal = np.ones((5, 5), "uint8")
    red_mask = cv2.dilate(red_mask, kernal)
    res_red = cv2.bitwise_and(imageFrame, imageFrame,
                              mask=red_mask)

    contours, hierarchy = cv2.findContours(red_mask,
                                           cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)

    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if (area > 300):
            x, y, w, h = cv2.boundingRect(contour)
            imageFrame = cv2.rectangle(imageFrame, (x, y),
                                       (x + w, y + h),
                                       (0, 0, 255), 2)
            cv2.putText(imageFrame, "Red Color", (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (0, 0, 255))
    h_min = cv2.getTrackbarPos("HUE Min","HSV")
    h_max = cv2.getTrackbarPos("HUE Max", "HSV")
    s_min = cv2.getTrackbarPos("SAT Min", "HSV")
    s_max = cv2.getTrackbarPos("SAT Max", "HSV")
    v_min = cv2.getTrackbarPos("VALUE Min", "HSV")
    v_max = cv2.getTrackbarPos("VALUE Max", "HSV")
    print (h_min)
    lower = np.array([h_min,s_min,v_min])
    upper = np.array([h_max, s_max, v_max])
    kernal = np.ones((5, 5), "uint8")
    mask = cv2.inRange(hsvFrame, lower,upper)
    mask = cv2.dilate(mask, kernal)
    result = cv2.bitwise_and(imageFrame,imageFrame,mask= mask)
    mask_2 = cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR)
    hstack = np.hstack([imageFrame,mask_2])
    contours, hierarchy = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if (area > 300):
            x, y, w, h = cv2.boundingRect(contour)
            imageFrame = cv2.rectangle(imageFrame, (x, y),
                                       (x + w, y + h),
                                       (0, 0, 255), 2)
            cv2.putText(imageFrame, "Red Color", (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (0, 0, 255))
    # Program Termination
    #cv2.imshow("Real Img", imageFrame)
    #cv2.imshow('HSV_frame', hsvFrame)
    #cv2.imshow("Mask", mask)
    #cv2.imshow("result", result)
    cv2.imshow("Hor stacking", hstack)
    cv2.imshow("result", result)'''
    #cv2.imshow("Real Img", imageFrame)

    #if cv2.waitKey(10) & 0xFF == ord('q'):
       # webcam.release()
        #cv2.destroyAllWindows()
       # break