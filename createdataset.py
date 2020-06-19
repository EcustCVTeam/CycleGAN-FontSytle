import os 
import cv2

def seprateData(imgdir, targetdir, orifilename):
    img = cv2.imread(imgdir)

    size  = 64 #图片大小为64*64
    h = img.shape[0] / size 
    w = img.shape[1] / size 

    print(w)
    print(h)

    path = targetdir
    if not os.path.exists(path):
        os.makedirs(path)
    
    i = 0
    
    for i in range(0,int(w)):
        ws = i*size
        im = img[0:64,ws:ws+64]
        filename = orifilename + str(i) + ".png"
        filepath = path+"\\"+filename
        cv2.imwrite(filepath,im)
        i = i+1



dir = "F:\\Resource\\Capitals64\\train\\"

img_list = [dir + i for i in os.listdir(dir)]
print(img_list)

save_path = "F:\\DataT"
for i in range(10):
    pname="p"+str(i)+"_"
    seprateData(img_list[i],save_path,pname)
