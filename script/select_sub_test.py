import numpy as np
import random
import pandas as pd
import shutil
np.random.seed(12)
import os
init_path = os.getcwd()

test_dir = "../data/img/test"
select = {'stim':[], 'correct_choice':[]}
select_prac = {'stim':[], 'correct_choice':[]}
for fold in os.listdir(test_dir):
    if fold.startswith("square"):
        path_new_square = "../data/img/sub_test/square"
        path_new_square_prac = "../data/img/sub_train/square"
        img_path = test_dir + "/" + "square"

        imgs = []
        for img in os.listdir(img_path):
            if img.endswith(".png"):
                img_single = img_path + "/" + img
                imgs.append(img_single)
            else:
                pass
        idx = random.sample(range(len(imgs)), 130) #each fold 120picture

        for id in idx[:120]:
            select['stim'].append(imgs[id])
            select['correct_choice'].append("Bad")
            shutil.copy(imgs[id], path_new_square)

        for id2 in idx[120:]:
            select_prac['stim'].append(imgs[id2])
            select_prac['correct_choice'].append("Bad")
            shutil.copy(imgs[id2], path_new_square_prac)

    elif fold.startswith("triangle"):
        img_path = test_dir + "/" + "triangle"
        path_new_triangle = "../data/img/sub_test/triangle"
        path_new_triangle_prac = "../data/img/sub_train/triangle"
        imgs = []
        for img in os.listdir(img_path):
            if img.endswith(".png"):
                img_single = img_path + "/" + img
                imgs.append(img_single)
            else:
                pass
        idx = random.sample(range(len(imgs)), 130) #each fold 120picture and 10 for practice
        for id in idx[:120]:
            select['stim'].append(imgs[id])
            select['correct_choice'].append("Good")
            shutil.copy(imgs[id], path_new_triangle)

        for id2 in idx[120:]:
            select_prac['stim'].append(imgs[id2])
            select_prac['correct_choice'].append("Good")
            shutil.copy(imgs[id2], path_new_triangle_prac)

    
    elif fold.startswith("circle"):
        img_path = test_dir + "/" + "circle"
        path_new_circle = "../data/img/sub_test/circle"
        path_new_circle_prac = "../data/img/sub_train/circle"
        imgs = []
        for img in os.listdir(img_path):
            if img.endswith(".png"):
                img_single = img_path + "/" + img
                imgs.append(img_single)
            else:
                pass
        idx = random.sample(range(len(imgs)), 130) #each fold 120picture and 10 for practice
        for id in idx[:120]:
            select['stim'].append(imgs[id])
            select['correct_choice'].append("No_info")
            shutil.copy(imgs[id], path_new_circle)

        for id2 in idx[120:]:
            select_prac['stim'].append(imgs[id2])
            select_prac['correct_choice'].append("No_info")
            shutil.copy(imgs[id2], path_new_circle_prac)

    else:
        pass
'''
df_stim_formal = pd.DataFrame(select)
df_stim_prac = pd.DataFrame(select_prac)
df_stim_formal.to_csv("../stim_formal.csv")
df_stim_prac.to_csv("../stim_prac.csv")
'''