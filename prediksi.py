# Bagian untuk mengenali wajah seseorang 

import numpy as np
import os
import cv2 

pengenalWajah = cv2.face.LBPHFaceRecognizer_create()
detektor = cv2.CascadeClassifier(
              "haarcascade_frontalface_default.xml")

def prediksiWajah(namaBerkas):
    citra = cv2.imread(namaBerkas)

    if citra is None:
        print("Tidak dapat membaca berkas citra")
        return    
    
    abuAbu = cv2.cvtColor(citra, cv2.COLOR_BGR2GRAY)
    daftarWajah = detektor.detectMultiScale(
        abuAbu, scaleFactor = 1.3, minNeighbors = 5)
    if daftarWajah is None:
        print("Wajah tidak terdeteksi")
        return
        
    for (x, y, w ,h) in daftarWajah:
        cv2.rectangle(citra, (x, y), (x + w, y + h), 
                  (255, 0, 0), 2)
        wajah = abuAbu[y:y+h, x:x+w]     
        labelId, konfiden = pengenalWajah.predict(wajah)
        if konfiden < 500:
            cv2.putText(citra, "(%s) %.0f" % \
                               (labelId, konfiden), 
                        (x, y - 2), cv2.FONT_HERSHEY_PLAIN, 
                        1, (0, 255, 0))
        else:
            cv2.putText(citra, "Entah", (x, y - 2), 
                        cv2.FONT_HERSHEY_PLAIN, 1, 
                        (0, 255, 0))        
    
    cv2.imshow("Hasil", citra)
    cv2.waitKey(0)
     
# Program utama
pengenalWajah.read("pelatihan.yml")
prediksiWajah("data-wajah/s15/8.pgm")
prediksiWajah("data-wajah/s40/8.pgm")
prediksiWajah("data-wajah/s12/8.pgm")
prediksiWajah("data-wajah/s21/8.pgm")
prediksiWajah("lena.png")







