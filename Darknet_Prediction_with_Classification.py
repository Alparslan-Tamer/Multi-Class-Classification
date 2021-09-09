import re
import cv2
import glob
import os
import torch
import time
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from korelasyon import korelasyon
from korelasyon_code.match_pairs import match_pairs
from korelasyon_code.models.matching import Matching
from config_korelasyon import MODEL_CONF
from yon_karar_edited import Decider
from tensorflow import keras
import tensorflow as tf

def prediction(my_image, net, match = None):  # sourcery no-metrics
    classes = []
    with open("data/sign.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]

    blob = cv2.dnn.blobFromImage(my_image, 1/255, (416,416), (0,0,0), swapRB=True, crop=False) # bunu (416, 416) yapmadan önce kullandığın modelin cfgsinde width ve height bak.
    net.setInput(blob)
    last_layer = net.getUnconnectedOutLayersNames()
    layer_out = net.forward(last_layer)

    ht, wt, _ = my_image.shape
    boxes = []
    confidences = []
    class_ids = []

    for output in layer_out:
        for detection in output:
            score = detection[5:]
            class_id = np.argmax(score)
            confidence = score[class_id]
            if confidence > .6:
                center_x = int(detection[0] * wt)
                center_y = int(detection[1] * ht)
                w = int(detection[2] * wt)
                h = int(detection[3]* ht)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                if w * h > 2000: # uzaklıga göre seçim yapıyoruz.
                    boxes.append([x, y, w, h])
                    confidences.append((float(confidence)))
                    class_ids.append(class_id)
    #print((class_ids))
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, .5, .4)
    font = cv2.FONT_HERSHEY_PLAIN
    colors = np.random.uniform(0, 255, size=(len(boxes), 3))

    ## Sağ - Sol - ve Dönülmez ayrımı için TF ile yapılmış olan yama #################

    if match:
        for i, value in enumerate(np.array(class_ids)):  
            if value in [1, 4, 3, 8] and confidences[i] < 0.9: # 1 - sağa dön, 4 - sola dön, 3 - saga donme, 8 - sola donme
                crop_img = my_image[boxes[i][1]:boxes[i][1]+boxes[i][3], boxes[i][0]:boxes[i][0]+boxes[i][2]]
                total_shape = crop_img.shape[0] * crop_img.shape[1]
                if (crop_img.shape[0] > 50 and total_shape > 5000) | (crop_img.shape[1] > 50 and  total_shape > 5000):
                    crop_img = cv2.resize(crop_img, (32, 32), interpolation = cv2.INTER_LINEAR)
                    crop_img = cv2.cvtColor(crop_img, cv2.COLOR_RGB2GRAY)
                    start = time.time()
                    pred = match_RL_model.predict(crop_img.reshape(1, 32, 32, 1))
                    pred = np.argmax(pred, axis = 1)[0]
                    end = time.time()
                    print(end - start) # yama işlemi için zaman bakma amaçlı print işlemi
                    if pred == 0:
                        class_ids[i] = 1
                    elif pred == 1:
                        class_ids[i] = 4
                    elif pred == 2:
                        class_ids[i] = 3
                    elif pred == 3:
                        class_ids[i] = 8
    
    ######################################################################

    labels = []
    if indexes is not tuple(): # sahnede nesne bulunmadığı durum için bu if gerekli
        for i in indexes.flatten():
            #print(i)
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            labels.append(label)
            confidence = str(round(confidences[i], 2))
            color = colors[i]
            cv2.rectangle(my_image, (x, y), (x+w, y+h), color, 2)
            cv2.putText(my_image, label + " " + confidence, (x, y-5), font, 1, (0,0,0), 2)

    my_image = cv2.resize(my_image, (1280, 720))
    cv2.imshow("Predicted Image", my_image)

    return labels #np.array(class_ids).take(indexes.flatten())



if __name__ == "__main__":
    tf.config.set_visible_devices([], 'GPU') # GPU kullanma işi predictionda hoş olmadı benim pcde, araçta da saçmalarsa bu kodu açabilirsiniz.
    device = 'cuda' if torch.cuda.is_available() and not False else 'cpu'
    match_RL_model = keras.models.load_model('SagSolModel.h5') # Sağ sol ayrımı için keras modelimizi yüklüyoruz.
    cfg = "cfg/yolov4-custom.cfg"
    weights = "backup_yolov4/yolov4-custom_best.weights"
    net = cv2.dnn.readNetFromDarknet(cfg, weights)

    # Yukarıdaki yükleme işlemleri kesinlikle yapılmalıdır. Aşagıda ise deneme amaçlı yaptığım bir aşama var o kısımı kodunuzda uyarlayabilirsiniz.

    # image_file_path = "16_08_2021_gunduz_00010032.jpg"
    # image = cv2.imread(image_file_path)
    # class_pred = prediction(image, net, match_RL_model) # sahnede tespit ettiği nesnelerin değerlerini dönüyor.
    # print(class_pred)

    # while True:
    #    if cv2.waitKey(1) == 27: # esc basarsanız görsel kapanır.
    #        break



    my_images = [f for f in glob.glob("img_folder/*.png")]
    my_images.sort(key=lambda x: [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', x)])
    for image_file in my_images:
        image = np.array(Image.open(image_file).convert("RGB"))
        if image is not None:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            image = image[:, 1280:, :]
            class_pred = prediction(image, net, match=match_RL_model) # sahnede tespit ettiği nesnelerin değerlerini dönüyor. match kısmını None bırakırsanız
                                                                      # sağ sol gözetmeden prediction yapılır.
        if cv2.waitKey(1) == 27:
            break
