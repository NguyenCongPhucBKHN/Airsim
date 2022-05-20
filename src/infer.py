from asyncore import file_dispatcher
from json.tool import main
import re
from socket import TIPC_LOW_IMPORTANCE
import sys
from unittest import result

from matplotlib.cbook import contiguous_regions
sys.path.insert(0, "/home/ivsr/CV_Group/phuc/airsim_proj/STDC-Seg")
import torch
import cv2
from model.model_stages import BiSeNet
import model.model_stages 
import numpy as np
import transform_cv2 as T
import matplotlib.pyplot as plt
import glob
import os
import torchvision.ops.boxes as bops
from PIL import Image




import matplotlib.pyplot as plt

from datetime import datetime






class Model1:
    def __init__(self, yolov5, stdc, cnt_ths=0.5):
        self.yolov5=yolov5
        self.stdc=stdc
        self.cnt_ths=cnt_ths

    def inferDect(self, img_path):
            
            outputyolov5=[]
            results=self.yolov5(img_path)
            for i in range(results.xyxy[0].shape[0]):
                coord = results.xyxy[0][i].cpu().numpy() #Goc toa do nam o doc tren ben trai
                x1 = int(coord[0].item())  #x_chieungangbentrai
                y1 = int(coord[1].item())  #y_chieudoc_tren
                x2 = int(coord[2].item())   #x_chieungangbenPHAI
                y2 = int(coord[3].item())  #y_chieudoc_duoi
           
                cordinate=[x1, y1,x2,y2]
                outputyolov5.append(cordinate)
            print("output yolov5: ", outputyolov5)
            return(outputyolov5)
    def inferSeg(self, img_path):
        to_tensor=T.ToTensor(
            mean=(0.485, 0.456, 0.406),
            std=(0.485, 0.456, 0.406),
            )
        im=cv2.imread(img_path)[:,:, ::-1]
        im=cv2.resize(im,(1280, 720))
        im=to_tensor(dict(im=im, lb=None))['im'].unsqueeze(0).cuda()
        out,_,_=self.stdc(im)
        out=out.cpu()
        pred=torch.argmax(out, dim=1)
        pred=pred.numpy().squeeze()
        out=np.zeros(pred.shape)
        palette = np.array([0, 0, 0, 255, 255, 255]).reshape([2,3])
        pred = palette[pred]
        pred=pred.reshape(720, 1280,3).astype(np.uint8) 
        return pred
    

    def findContour(self, pred):
        C=[]
        im_bw = cv2.cvtColor(pred, cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(im_bw, (5,5), 0)
        im_bw = cv2.Canny(blur, 10, 90)
        contours, hierarchy = cv2.findContours(im_bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        print("Number of contours:" + str(len(contours)))
        if(len(contours)==2):
            print(contours[0].shape)
            print(contours[1].shape)
            # print("Differrent two contour: ",contours[1]-contours[0])

        
        for coutour in contours:
            rect = cv2.boundingRect(coutour)
            print("Output Contours: ",rect)
            rect=np.array(rect)
            rect[0]=rect[0]
            rect[1]=rect[1]
            rect[2]=rect[0]+rect[2]
            rect[3]=rect[1]+rect[3]
            C.append(rect)
        
        return C
    def select_cnt(self, img_path):
        out_dects=self.inferDect(img_path)
        out_segs=self.inferSeg(img_path)
        out_cnts=self.findContour(out_segs)
        i=1
        for det in out_dects:
            for cnt in out_cnts:
                det_arr=np.array([det])
                seg_arr=np.array([cnt])
                det_tensor=torch.tensor(det_arr, dtype=torch.float)
                seg_tensor=torch.tensor(seg_arr, dtype=torch.float)
                print("Tensor Detect: ",det_tensor)
                print("Tensor Segment: ",seg_tensor)

                iou = bops.box_iou(det_tensor, seg_tensor)
                print("IoU: ", iou)
                if (iou>self.cnt_ths):
                    file_name=img_path.split('/')[-1][:-4]
                    
                    file_name=f'/home/ivsr/CV_Group/phuc/airsim_proj/STDC-Seg/Diem1-20220519T121024Z-001/test{file_name}__{i}.jpg'
                    im=cv2.rectangle(out_segs, (cnt[0],cnt[1]), (cnt[2],cnt[3]), (255, 0, 0), 2)
                    cv2.imwrite(file_name, im)
                    i=i+1
                    return True
                else:
                    return False
                

                   
                
        

    


        












yolo=torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True,classes=0)
stdc=BiSeNet('STDCNet1446', 2)
stdc.load_state_dict(torch.load("/home/ivsr/CV_Group/phuc/airsim_proj/STDC-Seg/log/train_STDC2-Seg/pths/model_maxmIOU75.pth"))
stdc.cuda()
stdc.eval()
#A=glob.glob("/home/ivsr/CV_Group/phuc/airsim_proj/STDC-Seg/test2/test/*png")
A=glob.glob("/home/ivsr/CV_Group/phuc/airsim_proj/STDC-Seg/Diem1-20220519T121024Z-001/Diem1/*jpg")
#A=glob.glob("/home/ivsr/CV_Group/phuc/airsim_proj/STDC-Seg/test2/img_1/*.jpg")

model1= Model1(yolov5=yolo, stdc=stdc)
i=0
j=0
for a in A:
    print("Image: ",a)
    if(model1.select_cnt( a)):
        j=j+1
    i=i+1
print(i)
print("number image segmentation: ",j)
    
   
    




    