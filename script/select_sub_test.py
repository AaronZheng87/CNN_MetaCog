import numpy as np
import random
import pandas as pd
import shutil
np.random.seed(12)
import os
init_path = os.getcwd()

test_dir = "../data/img/test"

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
        idx = random.sample(range(len(imgs)), 130) #each fold 130 picture

        for id in idx[:120]:
            shutil.copy(imgs[id], path_new_square)

        for id2 in idx[120:]:
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
        idx = random.sample(range(len(imgs)), 130) #each fold 120 picture and 10 for practice
        for id in idx[:120]:
            shutil.copy(imgs[id], path_new_triangle)

        for id2 in idx[120:]:
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
            shutil.copy(imgs[id], path_new_circle)

        for id2 in idx[120:]:
            shutil.copy(imgs[id2], path_new_circle_prac)

    else:
        pass


select = {'stim':[], 'correct_choice':[]}
select_prac = {'stim':[], 'correct_choice':[]}
final_test_path = "../data/img/sub_test"
final_prac_path = "../data/img/sub_train"

for fold in os.listdir(final_test_path):
        if fold.startswith("square"):
            final_square_path = final_test_path + "/" + "square"
            for img in os.listdir(final_square_path):
                if img.endswith(".png"):
                    img_single = final_square_path + "/" + img
                    select['stim'].append(img_single)
                    select['correct_choice'].append("Bad")

        elif fold.startswith("triangle"):
            final_triangle_path = final_test_path + "/" + "triangle"
            for img in os.listdir(final_triangle_path):
                if img.endswith(".png"):
                    img_single = final_triangle_path + "/" + img
                    select['stim'].append(img_single)
                    select['correct_choice'].append("Good")

        elif fold.startswith("circle"):
            final_circle_path = final_test_path + "/" + "circle"
            for img in os.listdir(final_circle_path):
                if img.endswith(".png"):
                    img_single = final_circle_path + "/" + img
                    select['stim'].append(img_single)
                    select['correct_choice'].append("No_info")

for fold in os.listdir(final_prac_path):
        if fold.startswith("square"):
            final_square_path = final_prac_path + "/" + "square"
            for img in os.listdir(final_square_path):
                if img.endswith(".png"):
                    img_single = final_square_path + "/" + img
                    select_prac['stim'].append(img_single)
                    select_prac['correct_choice'].append("Bad")

        elif fold.startswith("triangle"):
            final_triangle_path = final_prac_path + "/" + "triangle"
            for img in os.listdir(final_triangle_path):
                if img.endswith(".png"):
                    img_single = final_triangle_path + "/" + img
                    select_prac['stim'].append(img_single)
                    select_prac['correct_choice'].append("Good")

        elif fold.startswith("circle"):
            final_circle_path = final_prac_path + "/" + "circle"
            for img in os.listdir(final_circle_path):
                if img.endswith(".png"):
                    img_single = final_circle_path + "/" + img
                    select_prac['stim'].append(img_single)
                    select_prac['correct_choice'].append("No_info")

df_stim_formal = pd.DataFrame(select)
df_stim_prac = pd.DataFrame(select_prac)
df_stim_formal.to_csv("../stim_formal.csv")
df_stim_prac.to_csv("../stim_prac.csv")