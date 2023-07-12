# Measuring Confidence in the Perceptual Matching Task: Exploring the Feasibility of Simulating the Process using Computational Models





## Contents

- [Goal](#goal)
- [Environment](#environment)
- [Method](#Method)
- [Result](#result)
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
The datasets were created using [Pillow](https://pillow.readthedocs.io/en/stable/). These datasets consist of 3 possible image classes, including triangles, circles, and squares. To generate the datasets, we utilized the [draw_origin.py](https://github.com/AaronZheng87/CNN_Moral-MetaCog/blob/main/script/draw_origin.py) file to draw 100 different 300x300 pixel images for each image class, resulting in original images:
```python
draw_circle3(300, 300, 100)
draw_rectangle3(300, 300, 100)
draw_triangle3(300, 300, 100)
```
Subsequently, noise was added to the original images, and this process was repeated multiple times. The method for adding noise can be found at https://onelinerhub.com/python-pillow/how-to-add-noise. The pipeline for adding noise and repeating the process can be found in the [draw_img.py](https://github.com/AaronZheng87/CNN_Moral-MetaCog/blob/main/script/draw_img.py) script. The training datasets contain varying levels of noise, while the validation and test datasets have a single noise level:

```python
add_to_all2(0.5, 5, "train")
add_to_all2(0.55, 5, "train")
add_to_all2(0.6, 5, "train")
add_to_all2(0.61, 5, "train")
add_to_all2(0.62, 5, "train")
add_to_all2(0.63, 5, "train")
add_to_all2(0.64, 5, "train")
add_to_all2(0.625, 5, "validation")
add_to_all2(0.65, 5, "test")
```



- The model and subjects would finally evaluate on [sub_test file](https://github.com/AaronZheng87/CNN_Moral-MetaCog/tree/main/data/img/sub_test), which was selected 120 images of each classes from test file.



#### Task

Prior to the experiment, subjectss were asked to learn a accosication between different class of geometric images and the moral labels: 

- Triangle images matching "Good Person"
- Circle images matching "Neutral Person"
- Square images matching "Bad Person"

Each trial commenced with participants fixating on a small white cross for 500ms, followed by the presentation of an image for 200ms. The task for participants was to discern the corresponding label of the image using their right hand ("j", "k", "l" keys) within 1500ms. After indicating the label corresponding to the image, participants were required to report their decision confidence using their left hand within 1500ms ("correct" or "incorrect", "a" or "d" keys). The mappings between buttons and moral labels, as well as confidence responses, were randomized across trials.

See the [exp file](https://github.com/AaronZheng87/CNN_Moral-MetaCog/tree/main/exp).

## Result

- The ROC AUC score of model's type 1 classification is `0.814`

- The ROC AUC score of model's confidence classification is `0.672`

  

  The probability of confidence given correctness of model and subject are: 

  |          | $p(low|correct)$   | $p(high|correct)$   |
  | -------- | ------------------ | ------------------- |
  | CNN      | 0.426              | 0.574               |
  | Subject1 | 0.156              | 0.844               |
  |          | $p(low|incorrect)$ | $p(high|incorrect)$ |
  | CNN      | 0.731              | 0.269               |
  | Subject1 | 0.591              | 0.409               |
  
  
  
  The (meta) dprime of model and subject: 
  
  |          | dprime | meta dprime |
  | -------- | ------ | ----------- |
  | CNN      | 1.276  | 0.910       |
  | Subject1 | 2.156  | 2.198       |
  
  The result can be foud at [1.3.test_model.ipynb](https://github.com/AaronZheng87/CNN_Moral-MetaCog/blob/main/script/1.3.test_model.ipynb)









## Contributors

Y-R.Zheng conceived project and planned experiments. N.Mei directed the modeling of the project. This project's experiment referenced [Hu Chuanpeng's research](https://online.ucpress.edu/collabra/article/6/1/20/113065/Good-Me-Bad-Me-Prioritization-of-the-Good-Self). Y-R.Zheng implemented experiments and analyzed the results.
