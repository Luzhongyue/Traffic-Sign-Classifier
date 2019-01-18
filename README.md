# Traffic-Sign-Classifier
---
## Overview

Implemention a convolutional neural networks based the [Lenet](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf) architecture using TensorFlow so it can decode traffic signs from natural images by using the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset).After the model is trained, then testing the model program on new images of traffic signs found on the web.

Here is the examples of some traffic sign images:

![](https://github.com/Luzhongyue/Traffic-Sign-Classifier/blob/master/Images/original.png)

The steps of this project are the following:
* Load the data set
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

## Dependencies

This project requires **Python 3.5** and the following Python libraries installed:

- [Jupyter](http://jupyter.org/)
- [NumPy](http://www.numpy.org/)
- [SciPy](https://www.scipy.org/)
- [scikit-learn](http://scikit-learn.org/)
- [TensorFlow](http://tensorflow.org)
- [Matplotlib](http://matplotlib.org/)
- [Pandas](http://pandas.pydata.org/) (Optional)

## Dataset and Repository

1. [Download the data set](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/traffic-signs-data.zip). This is a pickled dataset in which we've already resized the images to 32x32. It contains a training, validation and test set.
2. Clone the project, which contains the Ipython notebook and the writeup template.
```sh
git clone https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project
cd CarND-Traffic-Sign-Classifier-Project
jupyter notebook Traffic_Sign_Classifier.ipynb
```

## Data Set Summary & Exploration 

* The size of training set is 34799
* The size of the validation set is 4410
* The size of the test set is 12360
* The number of unique classes/labels in the data set is 43
* The shape of a traffic sign image is (32,32, 3)

![](https://github.com/Luzhongyue/Traffic-Sign-Classifier/blob/master/Images/training_set.png)
![](https://github.com/Luzhongyue/Traffic-Sign-Classifier/blob/master/Images/validation_set.png)
![](https://github.com/Luzhongyue/Traffic-Sign-Classifier/blob/master/Images/test_set.png)

## Data preprocessed

Because the origin training data is uniform and not pre-processed. So preprocessing images was needed. So the below methods were used to make the training data become more complex and not uniform.
* convert the images to grayscale
* equalize the images that make the images less blurry
*	normalize the images , so that the data has mean zero and equal variancethe 

Here is the examples of some traffic sign image after preprocessing.
![](https://github.com/Luzhongyue/Traffic-Sign-Classifier/blob/master/Images/processed.png)

To add more data to the the data set, so the images of each label made more balanced,I used the following techniques .
* random warp()
* random rotate()
* random bright()
* random translate()

![](https://github.com/Luzhongyue/Traffic-Sign-Classifier/blob/master/Images/augemented.png)

## Model Ariculture

### Lenet

Lenet is a convolutional neural network designed to recognize visual patterns directly from pixel images with minimal preprocessing. It works well on handling hand-written characters.
![](https://github.com/Luzhongyue/Traffic-Sign-Classifier/blob/master/Images/Lenet.png)

### My model

My model was modofied based on the Lenet.

|Layer                       | Shape    |
|----------------------------|:--------:|
|Input                       | 32x32x1  |
|Convolution (valid, 3x3x32) | 30x30x32 |
|Max Pooling (valid, 2x2)    | 15x15x32 |
|Activation  (ReLU)          | 15x15x32 |
|Convolution (valid, 3x3x64) | 13x13x64 |
|Max Pooling (valid, 2x2)    | 6x6x64   |
|Activation  (ReLU)          | 5x5x16   |
|Flatten                     | 9504     |
|fully connected             | 512      |
|Activation  (ReLU)          | 512      |
|fully connected             | 256      |
|Activation  (ReLU)          | 256      |
|fully connected             | 43       |

## Performance

* training set accuracy: 1.0
* validation set accuracy: 0.97
* test set accuracy: 0.956

## Test the Model on New Images

Choosing several German traffic signs found on the web to be calssified by the model. Here are ten German traffic signs that I found on the web:

![](https://github.com/Luzhongyue/Traffic-Sign-Classifier/blob/master/Images/total.png)

Here are the results of the prediction:

![](https://github.com/Luzhongyue/Traffic-Sign-Classifier/blob/master/Images/1.png)
![](https://github.com/Luzhongyue/Traffic-Sign-Classifier/blob/master/Images/2.png)
![](https://github.com/Luzhongyue/Traffic-Sign-Classifier/blob/master/Images/3.png)
![](https://github.com/Luzhongyue/Traffic-Sign-Classifier/blob/master/Images/4.png)
![](https://github.com/Luzhongyue/Traffic-Sign-Classifier/blob/master/Images/5.png)
![](https://github.com/Luzhongyue/Traffic-Sign-Classifier/blob/master/Images/6.png)

