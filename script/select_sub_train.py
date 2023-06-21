import numpy as np
import random
import pandas as pd
import shutil
np.random.seed(12)
import os
init_path = os.getcwd()

test_dir = "../data/img/test"
select = {'stim':[], 'correct_choice':[]}
for fold in os.listdir(test_dir):
    if fold.startswith("square"):
        path_new_square = "../data/img/sub_test/square"
        img_path = test_dir + "/" + "square"

        imgs = []
        for img in os.listdir(img_path):
            if img.endswith(".png"):
                img_single = img_path + "/" + img
                imgs.append(img_single)
            else:
                pass
        idx = random.sample(range(len(imgs)), 30) #each fold 120picture

        for id in idx:
            select['stim'].append(imgs[id])
            select['correct_choice'].append("Bad")
            shutil.copy(imgs[id], path_new_square)

    elif fold.startswith("triangle"):
        img_path = test_dir + "/" + "triangle"
        path_new_triangle = "../data/img/sub_test/triangle"
        imgs = []
        for img in os.listdir(img_path):
            if img.endswith(".png"):
                img_single = img_path + "/" + img
                imgs.append(img_single)
            else:
                pass
        idx = random.sample(range(len(imgs)), 30) #each fold 120picture

        for id in idx:
            select['stim'].append(imgs[id])
            select['correct_choice'].append("Good")
            shutil.copy(imgs[id], path_new_triangle)
    
    elif fold.startswith("circle"):
        img_path = test_dir + "/" + "circle"
        path_new_circle = "../data/img/sub_test/circle"
        imgs = []
        for img in os.listdir(img_path):
            if img.endswith(".png"):
                img_single = img_path + "/" + img
                imgs.append(img_single)
            else:
                pass
        idx = random.sample(range(len(imgs)), 30) #each fold 120picture

        for id in idx:
            select['stim'].append(imgs[id])
            select['correct_choice'].append("Non_info")
            shutil.copy(imgs[id], path_new_circle)

    else:
        pass

