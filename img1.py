import cv2
import numpy as np

def resmiklasordenal(dosyaadi):
  resim = cv2.imread("%s"%dosyaadi)
  return resim

girisverisi = np.array([])
for i in range(30):
 klasordenalinanresim = 0
 i = i + 1
 string = ("kalem\kalem1.jpg")
 klasordenalinanresim = resmiklasordenal(string)
 boyutlandirilmisresim = cv2.resize(klasordenalinanresim,(224,224))
 print(girisverisi)
 girisverisi = np.append(girisverisi,boyutlandirilmisresim)
 print(i)
#veri setinin düzenlenmesi
 girisverisi = np.reshape(girisverisi,(-1,224,224,3))
 np.save("girisverimiz",girisverisi)

 print(girisverisi.shape)


cv2.waitKey(0)
cv2.destroyAllWindows()
