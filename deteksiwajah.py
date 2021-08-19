# Pendeteksi wajah

import numpy as np
import cv2 

pengklasifikasiWajah = cv2.CascadeClassifier(
    "haarcascade_frontalface_default.xml")

citra = cv2.imread("MIT.jpg")

if citra is None:
    print("Tidak dapat membaca berkas citra")
    exit()    
    
abuAbu = cv2.cvtColor(citra, cv2.COLOR_BGR2GRAY)
dafWajah = pengklasifikasiWajah.detectMultiScale(
    abuAbu, scaleFactor = 1.3, minNeighbors = 2)
    
print("Jumlah wajah terdeteksi:", len(dafWajah))

for (x, y, w ,h) in dafWajah:
    cv2.rectangle(citra, (x, y), (x + w, y + h), 
                  (255, 0, 0), 2)
    
cv2.imshow('Citra wajah', citra)

cv2.waitKey(5000)
cv2.destroyAllWindows()



