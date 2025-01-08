 #context  windows10 / anaconda / python 3.2.0
import cv2
from PIL import Image



print(cv2.__version__) # 3.2.0
imgloc = "/mnt/netapp2/Home_FT2/home/usc/cursos/curso040/Documentos/tfg/codebase-light-velev/tfg_codebase_cesga/dataset/FIVES512/train/image/1_A.png" #this path works fine.  
# imgloc = "D:\\violettes\\Software\\Central\\test.jpg"   this path works fine also. 
#imgloc = "D:\violettes\Software\Central\test.jpg" #this path fails.



img = cv2.imread(imgloc, cv2.IMREAD_GRAYSCALE)
#height, width, channels = img.shape
#print (height, width, channels)
if img is None:
    print(f"Error: cv2.imread() no pudo cargar la imagen en {imgloc}")
else:
    print(f"Image loaded")
