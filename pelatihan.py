# Pelatihan ke pengenal wajah

import numpy as np
import os
import cv2 

pengenalWajah = cv2.face.LBPHFaceRecognizer_create()
detektor= cv2.CascadeClassifier(
              "haarcascade_frontalface_default.xml")

def perolehCitraDanLabel(lintasan):
    # peroleh semua subfolder di lintasan
    daftarFolderCitra = [os.path.join(lintasan, f) \
        for f in os.listdir(lintasan)] 

    # Mula-mula, daftarSampelWajah dan daftarIdWajah
    #    berupa senarai kosong
    daftarSampelWajah = [] 
    daftarIdWajah= [] 
    
    # Proses semua berkas di subfolder
    for folderCitra in daftarFolderCitra:
        print("----------------------------")
        pencacah = 1
        daftarBerkas = os.listdir(folderCitra)
        for berkas in daftarBerkas:
            if pencacah == 9:
                break
                
            pencacah = pencacah + 1

            # Lanjutkan pemrosesan            
            berkasCitra = folderCitra + "\\" + berkas
            print("Pemrosesan berkas citra", berkasCitra)
            
            # Baca berkas citra. mode skala keabu-abuan
            citra = cv2.imread(berkasCitra, 0) 
      
            # Ambil angka saja (buang s)
            idWajah = os.path.basename(folderCitra)[1:]
            idWajah = int(idWajah)
        
            # Ambil wajah
            daftarWajah = detektor.detectMultiScale(citra)
            
            # Simpan wajah dan ID ke senarai
            for (x, y, w, h) in daftarWajah:
                daftarSampelWajah.append(
                    citra[y : y + h, x : x + w])
                daftarIdWajah.append(idWajah)
            
    return daftarSampelWajah, daftarIdWajah

# Proses pelatihan
daftarWajah, daftarIdWajah = perolehCitraDanLabel(
                                 "data-wajah")
pengenalWajah.train(daftarWajah, np.array(daftarIdWajah))
pengenalWajah.save("pelatihan.yml")

