import matplotlib.pyplot as plt
from PIL import Image,ImageFile
import random
import os

ImageFile.LOAD_TRUNCATED_IMAGES = True

def random_crop(image):       
    
    ###image 是pil读取的，crop_shape是裁剪的大小
    h=int(1*image.size[0]/10)
    w=int(1*image.size[1]/10)
    nw = random.randint(0, image.size[0] - h)  ##裁剪图像在原图像中的坐标
    nh = random.randint(0, image.size[1] - w)
    image_crop = image.crop((nw, nh, nw + h, nh + w))
 
    return image_crop

 
if __name__ == "__main__":
    path = r'./TangkaDataset/'
    img_path = os.listdir(path)
    count=0
    for item in range(len(img_path)):
        count+=1
        img_name = img_path[item]
        image_path = os.path.join(path, img_name)
        image = Image.open(image_path)
        print(image.size)
        print(image.width)
        for i in range(10):
            # h=int(3*image.height/5)
            # w=int(3*image.width/5)
            hr= random_crop(image)
            
            name=os.path.splitext(img_name)[0]+'_'+str(i)+os.path.splitext(img_name)[1]
            hr.save(os.path.join('T10/', name))