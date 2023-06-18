import random
x0 = []
x1 = []
y0 = []
y1 = []


for i in range(10): 
    x0_tmp = random.randint(0,300)
    x1_tmp = random.randint(0,300)
    while x0_tmp == x1_tmp:
        x1_tmp = random.randint(0, 300)
    y0_tmp = random.randint(0,300)
    y1_tmp = random.randint(0,300)
    while y0_tmp == y1_tmp:
        y1_tmp = random.randint(0, 300)

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

