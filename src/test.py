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



import matplotlib.pyplot as plt
# plt.imshow(img)
# plt.show()

A=glob.glob("/home/ivsr/CV_Group/phuc/airsim_proj/STDC-Seg/test2/img_1/*.jpg")
B=os.listdir("/home/ivsr/CV_Group/phuc/airsim_proj/STDC-Seg/test2/img_1")

img=cv2.imread(A[0])
# print(img.shape)

def yolov5_infer(img_path):
    #Return x, y, w, h
    img=cv2.imread(A[0])
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True,classes=0)
    results=model(A[0])
    for i in range(results.xyxy[0].shape[0]):
        coord = results.xyxy[0][i].cpu().numpy() #Goc toa do nam o doc tren ben trai
        x1 = int(coord[0].item())  #x_chieungangbentrai
        y1 = int(coord[1].item())  #y_chieudoc_tren
        x2 = int(coord[2].item())   #x_chieungangbenPHAI
        y2 = int(coord[3].item())  #y_chieudoc_duoi
        crop_img = img[y1:y2, x1:x2] 
    #return y1, y2, x1, x2 
    return x1, y1, x2, y2
    
def stdc_infer(img_path, weight_path):
    #Return all coutours, each coutour contains Let (x,y) be the top-left coordinate of the rectangle and (w,h) be its width and height. 
    C=[]
    net=BiSeNet('STDCNet1446', 2)
    net.load_state_dict(torch.load(weight_path))
    net.cuda()
    net.eval()
    to_tensor=T.ToTensor(
        mean=(0.485, 0.456, 0.406),
        std=(0.485, 0.456, 0.406),
    )

    im=cv2.imread(img_path)[:,:, ::-1]
    im=to_tensor(dict(im=im, lb=None))['im'].unsqueeze(0).cuda()
    out,_,_=net(im)
    out=out.cpu()
    pred=torch.argmax(out, dim=1)
    pred=pred.numpy().squeeze()
    out=np.zeros(pred.shape)
    palette = np.array([0, 0, 0, 255, 255, 255]).reshape([2,3])
    pred = palette[pred]

    pred=pred.reshape(720, 1280,3).astype(np.uint8)
    
    im_bw = cv2.cvtColor(pred, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(im_bw, (5,5), 0)
    im_bw = cv2.Canny(blur, 10, 90)
    
    contours, hierarchy = cv2.findContours(im_bw, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for coutour in contours:
        rect = cv2.boundingRect(coutour)
        rect=np.array(rect)
        rect[0]=rect[0]
        rect[1]=rect[1]
        rect[2]=rect[0]+rect[2]
        rect[3]=rect[1]+rect[3]
        im=cv2.rectangle(pred, (rect[0],rect[1]), (rect[2],rect[3]), (255, 0, 0), 2)
        C.append(rect)
    
    # plt.imshow(im)
    # plt.show()
    return C 



out_stdc=stdc_infer(A[0],weight_path="/home/ivsr/CV_Group/phuc/airsim_proj/STDC-Seg/log/train_STDC2-Seg/pths/model_maxmIOU75.pth" )
print(out_stdc)



out_yolov5=yolov5_infer(A[2])
print("output yolov5: ",out_yolov5)

# box1 = torch.tensor(out_yolov5, dtype=torch.float)
# box2 = torch.tensor(out_stdc[0], dtype=torch.float)
# iou = bops.box_iou(box1, box2)

in1=np.array([out_stdc[1]])
in2=np.array([out_yolov5])
# print(in1)
# print(in2)
box1 = torch.tensor(in1, dtype=torch.float)
box2 = torch.tensor(in2, dtype=torch.float)
iou = bops.box_iou(box1, box2)
# print(iou)

print("out stdc: ", out_stdc)






















# for a in A:
#     im = cv2.imread(a)[:, :, ::-1]
#     im = to_tensor(dict(im=im, lb=None))['im'].unsqueeze(0).cuda()
#     print(im.shape)
#     out, _,_=net(im)
#     out=out.cpu()
#     print(out.shape)
#     pred = torch.argmax(out, dim=1)
#     print(pred.shape)
#     pred = pred.numpy().squeeze()
#     out = np.zeros(pred.shape)

#     palette = np.array([0, 0, 0, 255, 255, 255]).reshape([2,3])
#     # out[pred==1] =255.
#     # # plt.imshow(out, vmax=255.)
#     # # plt.show()
#     # np.save( '/content/drive/MyDrive/Segmentation/STDC-Seg/test/x=0,y=4.92,z=-2.47_4.npy',out)




#     # palette = np.random.randint(0, 256, (256, 3), dtype=np.uint8)
#     pred = palette[pred]
#     print(pred.shape)
    
#     pred=pred.reshape(720, 1280,3)
#     print(pred.shape)
#     cv2.imwrite(os.path.join("/home/ivsr/CV_Group/phuc/airsim_proj/STDC-Seg/test2/seg",B[i]), pred)
#     i+=1
