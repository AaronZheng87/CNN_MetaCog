# Measuring Confidence in the Perceptual Matching Task: Exploring the Feasibility of Simulating the Process using Computational Models





## Contents

- [Goal](#goal)
- [Environment](#environment)
- [Method](#Method)
- [Result](#result)
- [Discussion](#discussion)
- [Contributors](#contributors)



## Goal
This project endeavors to examine the manifestation of confidence in the perceptual matching task and explore the potential of simulating this unconscious processing of human  through the utilization of deep neural networks. Moreover, the project aims to replicate the findings of Webb's experiment 1 as an additional objective.




## Environment
### Hardware

- [Platform](https://www.autodl.com)
- GPU: RTX 3090(24GB) * 1
- CPU: 14 vCPU Intel(R) Xeon(R) Gold 6330 CPU @ 2.00GHz

### Python environment
- python: 3.8.13
- numpy: 1.23.5
- pandas: 1.5.3 
- scikit-learn: 1.2.1
- pytorch: 1.12.1+cu113
## Method

### Stimuli and task

#### Stimuli(datasets)
The datasets were produced by [Pillow](https://pillow.readthedocs.io/en/stable/). The datasets have 10 possible image classes(triangle, circle and square). We used [py](https://github.com/AaronZheng87/CNN_Moral-MetaCog/blob/main/script/draw_origin.py) file drew 100 different 300x300 pixel images each image classed as original images:
```python
draw_circle3(300, 300, 100)
draw_rectangle3(300, 300, 100)
draw_triangle3(300, 300, 100)
```
#### Task



## Result





## Discussion





## Contributors

Y-R.Zheng conceived project and planned experiments. N.Mei directed the modeling of the project. This project's experiment referenced [Hu Chuanpeng's research](https://online.ucpress.edu/collabra/article/6/1/20/113065/Good-Me-Bad-Me-Prioritization-of-the-Good-Self). Y-R.Zheng implemented experiments and analyzed results.
