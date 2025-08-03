import streamlit as st
import cv2
import numpy as np

st.title("ตรวจจับวัตถุสีแดง")
img_file = st.file_uploader("เปิดไฟล์ภาพ")

if img_file is not None:    
    file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    #----------------------------------------------
    imgYCrCb  = cv2.cvtColor(img,cv2.COLOR_BGR2YCR_CB)

    channels = cv2.split(imgYCrCb)# Y 0   Cr 1   Cb 2
    Cr = channels[1]

    ret,BW = cv2.threshold(Cr,190,255,cv2.THRESH_BINARY)
 
    contours, hierarchy = cv2.findContours(BW,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0: #เจอ
        areas = [cv2.contourArea(c) for c in contours]
        max_index = np.argmax(areas) #[ 20000 350000 456 ] 0 1 2
        cnt = contours[max_index]

        x,y,w,h = cv2.boundingRect(cnt)
        
        img_out = img.copy()
        cv2.rectangle(img_out,(x,y),(x+w,y+h),(0,255,255),4) #BGR
        #----------------------------------------------
        col1, col2 = st.columns(2)
        col1.image(img, caption='ภาพ Input',channels="BGR")
        col2.image(Cr, caption='ภาพ Cr')
        
        col1, col2 = st.columns(2)
        col1.image(BW, caption='ภาพ BWCr')
        col2.image(img_out, caption='ภาพ Output',channels="BGR")


