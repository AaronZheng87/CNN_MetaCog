import os
import random
import pandas as pd

recent_path = os.getcwd()

os.chdir("../img")
path = os.getcwd()


#for a, b, c in os.walk()

second_file = []
for i  in os.listdir(os.getcwd()):
    if i.split('_')[0] in ['circle', 'triangle', 'square']:
        file = os.getcwd() + '/' + str(i)
        second_file.append(file)
    else:
        pass

img = {'Img':[], 'Cat' : []}
for j in range(len(second_file)):
    for z in os.listdir(second_file[j]):
        if z.endswith('.png'):
            if z.startswith('square'):
                cat = 1
                image = str(second_file[j]) +'/' + str(z) 
                img['Img'].append(image)
                img['Cat'].append(cat)
            elif z.startswith('circle'):
                cat = 2
                image = str(second_file[j]) +'/' + str(z) 
                img['Img'].append(image)
                img['Cat'].append(cat)
            elif z.startswith('triangle'):
                cat = 3
                image = str(second_file[j]) +'/' + str(z) 
                img['Img'].append(image)
                img['Cat'].append(cat)

        else:
            pass

data = pd.DataFrame(img)


data.to_csv('dataset.csv')
