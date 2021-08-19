# Pendeteksi wajah menggunakan LBP

import numpy as np
import cv2 

pengklasifikasiWajah = cv2.CascadeClassifier(
    "lbpcascade_frontalface.xml")

citra = cv2.imread("lena.png")

if citra is None:
    print("Tidak dapat membaca berkas citra")
    exit()    
    
abuAbu = cv2.cvtColor(citra, cv2.COLOR_BGR2GRAY)
dafWajah = pengklasifikasiWajah.detectMultiScale(
    abuAbu, scaleFactor = 1.3, minNeighbors = 1)

for (x, y, w ,h) in dafWajah:
    cv2.rectangle(citra, (x, y), (x + w, y + h), 
                  (255, 0, 0), 2)
    
cv2.imshow("Citra wajah", citra)
cv2.waitKey(0) 

