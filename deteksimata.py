# Deteksi mata

import numpy as np
import cv2 

pengklasifikasiWajah = cv2.CascadeClassifier(
    "haarcascade_frontalface_default.xml")
pengklasifikasiMata = cv2.CascadeClassifier(
    "haarcascade_eye.xml")

citra = cv2.imread("lena.png")

if citra is None:
    print("Tidak dapat membaca berkas citra")
    
    
abuAbu = cv2.cvtColor(citra, cv2.COLOR_BGR2GRAY)
dafWajah = pengklasifikasiWajah.detectMultiScale(
    abuAbu, scaleFactor = 1.3, minNeighbors = 2)
for (x, y, w ,h) in dafWajah:
    cv2.rectangle(citra, (x, y), (x + w, y + h), 
                  (255, 0, 0), 2)
                  
    roiAbuAbu = abuAbu[y : y + h, x : x + w]
    roiWarna = citra[y: y + h, x : x + w]
    daftarMata = pengklasifikasiMata.detectMultiScale(
                     roiAbuAbu, 1.3, 2)
    for (mx, my, mw, mh) in daftarMata:
        cv2.rectangle(roiWarna, (mx, my),
                     (mx + mw, my + mh), 
                     (0, 255, 0), 2)
   
cv2.imshow('Citra wajah', citra)
cv2.waitKey(0) 

