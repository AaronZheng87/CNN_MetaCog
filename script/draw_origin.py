from PIL import Image, ImageDraw
import os
import random
#im.save(path + '/s.png')
path = os.getcwd()
random.seed(1)

def draw_square():
    im = Image.new('RGB', (300, 300), 'grey')
    draw_square = ImageDraw.Draw(im)

    # # 创建一个正方形,fill 代表的为颜色
    draw_square.line([50, 50, 50, 250], fill='white', width=3)  # 左竖线
    draw_square.line([50, 50, 250, 50], fill='white', width=3)  # 上横线
    draw_square.line([250, 50, 250, 250], 'white', width=3) #右竖线
    draw_square.line([50, 250, 250, 250], 'white', width=3) #下横线
    #im.show()
    im.save('../data/img/origin/square/square.png')

def draw_circle():
    im = Image.new('RGB', (300, 300), 'grey')
    draw_circle = ImageDraw.Draw(im)
    draw_circle.ellipse([50, 50, 250, 250], outline='white', width=3)
    # # 创建一个正方形,fill 代表的为颜色
    
    #im.show()
    im.save('../data/img/origin/circle/circle.png')

def draw_square2():
    im = Image.new('RGB', (300, 300), 'grey')
    draw_square2 = ImageDraw.Draw(im)

    # # 创建一个正方形,fill 代表的为颜色
    draw_square2.rectangle([50, 50, 250, 250], outline='white', width=3)
    #im.show()
    im.save('../data/img/origin/square/square.png')

def draw_triangle():
    im = Image.new('RGB', (300, 300), 'grey')
    draw_triangle = ImageDraw.Draw(im)


    draw_triangle.line([50, 250, 250, 250], fill='white', width=3)  
    draw_triangle.line([50, 250, 100, 50], fill='white', width=3)  
    draw_triangle.line([100, 50, 250,250], fill='white', width=3) 
    #im.show()
    im.save('../data/img/origin/triangle/triangle.png')

#draw_triangle()
#draw_circle()
#draw_square()



def draw_rectangle3(img_width, img_height, num):
    x0 = []
    x1 = []
    y0 = []
    y1 = []

## in the function x0 must larger than x1
    for i in range(num): 
        x0_tmp = random.randint(0,300)
        x1_tmp = random.randint(0,300)
        while x0_tmp == x1_tmp:
            x1_tmp = random.randint(0,300)
        y0_tmp = random.randint(0,300)
        y1_tmp = random.randint(0,300)
        while y0_tmp == y1_tmp:
            y1_tmp = random.randint(0,300)

        if x1_tmp > x0_tmp:
            x0.append(x0_tmp)
            x1.append(x1_tmp)
        else:
            x0.append(x1_tmp)
            x1.append(x0_tmp)
        if y1_tmp > y0_tmp:
            y0.append(y0_tmp)   
            y1.append(y1_tmp)
        else: 
            y0.append(y1_tmp)   
            y1.append(y0_tmp)

    for j in range(num):

        im = Image.new('RGB', (img_width, img_height), 'grey')
        draw_square = ImageDraw.Draw(im)
        # # 创建一个正方形,fill 代表的为颜色
        draw_square.rectangle(xy=(x0[j], y0[j], x1[j], y1[j]), fill = None, outline ="white",width=3)
       #im.show()
        path = '../data/img/origin/square/square'
        save_path = path + str(j) + '.png'
        im.save(save_path)


def draw_triangle3(img_width, img_height, num):
    x0 = []
    x1 = []
    x2 = []
    y0 = []
    y1 = []
    y2 = []
    for i in range(num): 
        x0_tmp = random.randint(0,300)
        x1_tmp = random.randint(0,300)
        x2_tmp = random.randint(0,300) 
        while x0_tmp==x1_tmp or x0_tmp == x2_tmp or x1_tmp == x2_tmp:
            x0_tmp = random.randint(0,300)
            x1_tmp = random.randint(0,300)
            x2_tmp = random.randint(0,300)
        
        y0_tmp = random.randint(0,300)
        y1_tmp = random.randint(0,300)
        y2_tmp = random.randint(0,300)

        while y0_tmp==y1_tmp or y0_tmp == y2_tmp or y1_tmp == y2_tmp:
            y0_tmp = random.randint(0,300)
            y1_tmp = random.randint(0,300)
            y2_tmp = random.randint(0,300)
        
        
        
        x0.append(x0_tmp)
        x1.append(x1_tmp)
        x2.append(x2_tmp)
        y0.append(y0_tmp)
        y1.append(y1_tmp)
        y2.append(y2_tmp)

    for j in range(num):

        im = Image.new('RGB', (img_width, img_height), 'grey')
        draw_triangle = ImageDraw.Draw(im)
        # # 创建一个正方形,fill 代表的为颜色
        draw_triangle.polygon(xy=(x0[j], y0[j], x1[j], y1[j], x2[j], y2[j]), fill = None, outline ="white",width=3)
        #im.show()
        path = '../data/img/origin/triangle/triangle'
        save_path = path + str(j) + '.png'
        im.save(save_path)




def draw_circle3(img_width, img_height, num):
    x0 = []
    x1 = []
    y0 = []
    y1 = []


    for i in range(num): 
        x0_tmp = random.randint(0,300)
        x1_tmp = random.randint(0,300)
        while x0_tmp == x1_tmp:
            x1_tmp = random.randint(0,300)
        y0_tmp = random.randint(0,300)
        y1_tmp = random.randint(0,300)
        while y0_tmp == y1_tmp:
            y1_tmp = random.randint(0,300)

        if x1_tmp > x0_tmp:
            x0.append(x0_tmp)
            x1.append(x1_tmp)
        else:
            x0.append(x1_tmp)
            x1.append(x0_tmp)
        if y1_tmp > y0_tmp:
            y0.append(y0_tmp)   
            y1.append(y1_tmp)
        else: 
            y0.append(y1_tmp)   
            y1.append(y0_tmp)

    for j in range(num):

        im = Image.new('RGB', (img_width, img_height), 'grey')
        draw_circle = ImageDraw.Draw(im)
        # # 创建一个正方形,fill 代表的为颜色
        draw_circle.ellipse(xy=(x0[j], y0[j], x1[j], y1[j]), fill = None, outline ="white",width=3)
        #im.show()
        path = '../data/img/origin/circle/circle'
        save_path = path + str(j) + '.png'
        im.save(save_path)

draw_circle3(300, 300, 100)
draw_rectangle3(300, 300, 100)
draw_triangle3(300, 300, 100)