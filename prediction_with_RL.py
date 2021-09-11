import re
import cv2
import glob
import os
import torch
import time
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from yon_karar_edited import Decider

def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
    
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl:
            yield to_device(b, self.device)
    
    def __len__(self):
        """Number of batches"""
        return len(self.dl)

import torch.nn as nn
import torch.nn.functional as F

class ImageClassificationBase(nn.Module):
    """Calculate loss for a batch of traning data"""
    def training_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        return loss
    
    """Calculate loss & accuracy for a batch of validation data"""
    def validation_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        acc = accuracy(out, labels)
        return {"val_loss":loss.detach(), "val_acc": acc}
    
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['train_loss'], result['val_loss'], result['val_acc']))
        
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

def conv_block(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), 
              nn.BatchNorm2d(out_channels), 
              nn.ReLU(inplace=True)]
    if pool: layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)

class ResNet9(ImageClassificationBase):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        # Input: 128 x 3 x 64 x 64
        self.sigmoid = nn.Sigmoid()
        self.conv1 = conv_block(in_channels, 32)
        self.conv2 = conv_block(32, 16, pool=True) # 1  
        self.classifier = nn.Sequential(nn.AdaptiveMaxPool2d(1), # 1
                                        nn.Flatten(), # 128 x 512
                                        nn.Dropout(0.2),
                                        nn.Linear(16, num_classes))
        
    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.classifier(out)
        return out

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
                    img = cv2.resize(crop_img, (32, 32))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = img / 255.0
                    img = torch.from_numpy(img).permute(2, 0, 1)
                    img = img.unsqueeze(0)
                    img = img.float()
                    start = time.time()
                    pred = match(img.to("cuda"))
                    end = time.time()
                    pred = torch.max(pred, dim=1)
                    pred = int(pred.indices[0])
                    print(end - start) # yama işlemi için zaman bakma amaçlı print işlemi
                    if pred == 0:
                        class_ids[i] = 1
                    elif pred == 1:
                        class_ids[i] = 3
                    elif pred == 2:
                        class_ids[i] = 4
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
    device = 'cuda' if torch.cuda.is_available() and not False else 'cpu'
    match_RL_model = to_device(ResNet9(3, 4), device)
    match_RL_model = torch.load('sagsolpytorch.h5') # Sağ sol ayrımı için keras modelimizi yüklüyoruz.
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
