import torch
import glob
import os
import cv2
A=glob.glob("/home/ivsr/CV_Group/phuc/airsim_proj/STDC-Seg/test2/img_1/*.jpg")
B=os.listdir("/home/ivsr/CV_Group/phuc/airsim_proj/STDC-Seg/test2/img_1")
img=cv2.imread(A[0])
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True,classes=0)
results=model(A[0])
for i in range(results.xyxy[0].shape[0]):
        coord = results.xyxy[0][i].cpu().numpy()
        x1 = int(coord[0].item())
        y1 = int(coord[1].item())
        x2 = int(coord[2].item())
        y2 = int(coord[3].item())
        crop_img = img[y1:y2, x1:x2]
    
cv2.imwrite(os.path.join("/home/ivsr/CV_Group/phuc/airsim_proj/STDC-Seg/test2/seg","test_crop.jpg"), crop_img)