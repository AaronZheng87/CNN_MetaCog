from PIL import Image
import random
import os
'''
https://onelinerhub.com/python-pillow/how-to-add-noise
'''
def add_noise(img_path, pixel, save_path):
    im = Image.open(img_path)
    for i in range(round(im.size[0]*im.size[1]/pixel)):
        im.putpixel(
            (random.randint(0, im.size[0]-1), random.randint(0, im.size[1]-1)),
            (random.randint(0,255),random.randint(0,255),random.randint(0,255))
        )
    im.save(save_path)

init_path = os.getcwd()


def add_to_all(pixel, num, path):
    os.chdir(init_path)
    os.chdir('../img/origin')
    path_origin = os.getcwd()
    for i in range(num):
        for filename in os.listdir(path_origin):
            if filename.endswith('.png') and filename.startswith("circle"):
                img_path = path_origin + '/'+ filename
                fold_path = "../circle_" + str(pixel)
                os.chdir(fold_path)
                save_path = "circle_" + str(pixel) + '-'+ str(i+1) + ".png"
                add_noise(img_path, pixel, save_path)

            elif filename.endswith('.png') and filename.startswith("square"):
                img_path = path_origin + '/'+ filename
                fold_path = "../square_" + str(pixel)
                os.chdir(fold_path)
                save_path = "square_" + str(pixel) + '-'+ str(i+1) + ".png"
                add_noise(img_path, pixel, save_path)

            elif filename.endswith('.png') and filename.startswith("triangle"):
                img_path = path_origin + '/'+ filename
                fold_path = "../triangle_" + str(pixel)
                os.chdir(fold_path)
                save_path = "triangle_" + str(pixel) + '-'+ str(i+1) + ".png"
                add_noise(img_path, pixel, save_path)


#add_to_all(0.5, 30)
##add_to_all(3, 30)