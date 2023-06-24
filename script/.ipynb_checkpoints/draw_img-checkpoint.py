from PIL import Image
import random
import os
random.seed(1)
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


def add_to_all(pixel, num):
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


def add_to_all2(pixel, num, tofold):

    os.chdir(init_path)
    os.chdir('../data/img/origin')
    path_origin = os.getcwd()
    for i in range(num):
        for filename in os.listdir(path_origin):
            if filename == "circle":
                fold_path = path_origin + '/'+ filename
                for img_name in os.listdir(fold_path):
                    if img_name.startswith('circle'):
                        img_path = fold_path + '/' + img_name
                        img_name2 = img_name[:-4]
                        save_path = "../" + str(tofold)+"/" + "circle"+ "/" +img_name2 +"_" +str(i+1) + ".png"
                        add_noise(img_path, pixel, save_path)
                    else:
                        pass
            elif filename == "square":
                fold_path = path_origin + '/'+ filename
                for img_name in os.listdir(fold_path):
                    if img_name.startswith('square'):
                        img_path = fold_path + '/' + img_name
                        img_name2 = img_name[:-4]
                        save_path = "../" + str(tofold)+"/" + "square"+ "/" +img_name2 +"_" +str(i+1) + ".png"
                        add_noise(img_path, pixel, save_path)
                    else:
                        pass

            elif filename == "triangle":
                fold_path = path_origin + '/'+ filename
                for img_name in os.listdir(fold_path):
                    if img_name.startswith('triangle'):
                        img_path = fold_path + '/' + img_name
                        img_name2 = img_name[:-4]
                        save_path = "../" + str(tofold)+"/" + "triangle"+ "/" +img_name2 +"_" +str(i+1) + ".png"
                        add_noise(img_path, pixel, save_path)
                    else:
                        pass


add_to_all2(0.55, 5, "test")
add_to_all2(0.5, 5, "validation")
add_to_all2(0.4, 10, "train")