**Traffic Sign Recognition** 

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/class_visualization.png "Class Visualization"
[image2]: ./examples/orig_images.png "Originals"
[image3]: ./examples/processed_images.png "Processed Images"
[image4]: ./examples/learn_curves.png "Learning Curves"
[image5]: ./new_examples/cropped/class_37.png "Traffic Sign 1"
[image6]: ./new_examples/cropped/class_1.png "Traffic Sign 2"
[image7]: ./new_examples/cropped/class_1_1.png "Traffic Sign 3"
[image8]: ./new_examples/cropped/class_11.png "Traffic Sign 4"
[image9]: ./new_examples/cropped/class_12.png "Traffic Sign 5"
[image10]: ./new_examples/cropped/class_12_1.png "Traffic Sign 6"
[image11]: ./new_examples/cropped/class_14.png "Traffic Sign 7"
[image12]: ./new_examples/cropped/class_17.png "Traffic Sign 8"
[image13]: ./new_examples/cropped/class_18.png "Traffic Sign 9"
[image14]: ./new_examples/cropped/class_18_1.png "Traffic Sign 10"

## Rubric Points
#### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

Here is a link to my [project code](https://github.com/tikei/CarND-Term1-P2/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a histogram showing the class(label) distribution of the data, pointing to an unbalanced dataset.

![alt text][image1]


### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

I decided to convert the images to grayscale for two main reasons. The color images would have required longer to process and learning on a much larger data set is computationally expensive. In addition the Yann LeCun research paper relating to the GTRSB Competition indicated that additional color channels do not add signifficantly to the performance of the model. 

I also normalised the image using the scikit-image library to improve contrast levels, as many images have been captured in a bad lighting conditions.
Using the skimage.exposure module's equalize_adapthist method I experimented with different kernel size parameters for the Adaptive Histogram Equalization, which did not produce noticeable difference in the learning performance, so in the final solution the default kernel size was used.

Here is an example of a traffic sign images before and after grayscaling and normalizing.

![alt text][image2]

![alt text][image3]


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image   							| 
| Convolution 5x5     	| 1x1 stride, same padding, outputs 32x32x16 	|
| TANH					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x16 				|
| Convolution 5x5	    | 1x1 stride, same padding, outputs 16x16x32      									|
| TANH     |            |
| Max pooling	      	| 4x4 stride,  outputs 4x4x32 				|
| Max pooling	      	| Extract branched output from first Convolution, 4x4 stride,  outputs 4x4x16 				|
| Fully connected		| Input 768, RELU, ouputs 120        									|
| Fully connected		| Input 120, RELU, ouputs 84       									|
| Output Layer		| Input 84, Logits, ouputs 43       									|
| Softmax				| Cross Entropy Cost Function      									|
| ADAM Optimiser

 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the following hyperparameters:
### Hyperparameters

EPOCHS = 30

BATCH_SIZE = 128

learn_rate = 0.005

Monitor accuracy improvement and use early stopping to avoid overfitting
early_stopping = True
no_improve_n = 7 # Number of epochs with no accuracy improvement 

regularization rate
lambda_rate = 0.0001 

Weights initialization variables
mu = 0
sigma = 0.1

I used AdamOptimizer to train the model. 

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.998
* validation set accuracy of 0.958
* test set accuracy of 0.945

The training results by epoch and a graph of the learning curves is contained in Cell 13 of the Ipython notebook.

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?

At first I used a more simple one Convolutional layer architecture, Max Pooling and two fully connected layers feeding into a Softmax output layer. It was loosely based on LeNet-5.

* What were some problems with the initial architecture?

The architecture was performing surprisingly well, given its lack of depth, however the Validation accuracy remained in the high 80's and it was badly overfitting.

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.

I changed the architecture by first adding an additional convolutional layer, then extracting ("raw") outputs from the first convolutional layer, to be fed, together with the outputs of the second convolutional layer, into the classifier.
I then added regularization, which helped somewhat with the overfitting. I also used early stopping, which helped optimize the EPOCHS hyperparameter, as well as mitigate against overfitting.

* Which parameters were tuned? How were they adjusted and why?

I initially used a lower learning rate, however, I then introduced a more aggressive rate and early stopping with a learning rate schedule (halving the learing rate, when the accuracy does not improve over the last n epochs)  

Adjusting the regularization rate did not significantly improve the results.

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

Convolutional layers work well with this problem as they learn different features (from simple shapes (lines, curves) to more complex ones), which helps the network classify the traffic signs correctly.
Dropout and regularization can help, as the network has large number of parameters relative to the size of the training set, which leads to overfitting. The persistent difference between training and validation accuracy as shown on the learning curve below points to overfitting.

![alt text][image4]

Nevertheless, the final validation and test accuracies around 95% provides a good basis for further importvement.


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are twelve German traffic signs that I found on the web:

![alt text][image5] ![alt text][image6] ![alt text][image7] 
![alt text][image8] ![alt text][image9] ![alt text][image10]
![alt_text][image11] ![alt text][image12] ![alt text][image13]
![alt_text][image14] ![alt text][image15] ![alt text][image16]


Some images could be hard to classify, due to 
- graffity (Class 17, No Entry sign), 
- rotations (Class 1 Speed Limit 30, Class25 Road Works, Class18- the second General Caution sign), 
- obstructions (Class 12 Priority Road), 
- shading (Class 11, Right Of Way at next intersection)

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the predictions:

| Predicted: | Correct label: | Top 5 predicted labels: |
|:--------------------------:|:-------------------------------:|:-----------------------------:|
| Predicted label: 18 | Correct label: 18 | 18, 37, 26, 25, 35 |
| Predicted label: 5  | Correct label: 1 |  5,  1,  6,  2, 3 |
| Predicted label: 37 | Correct label: 18 | 37, 18, 26, 27, 24 |
| Predicted label: 8  | Correct label: 1 | 8, 14,  7, 40,  2 |
| Predicted label: 12 | Correct label: 12 | 12, 40, 35, 15, 32 |
| Predicted label: 12 | Correct label: 17 | 12, 40, 11, 32, 35 |
| Predicted label: 30 | Correct label: 25 | 30, 25, 31, 29, 21 |
| Predicted label: 37 | Correct label: 37 | 37, 33, 40, 35, 18 |
| Predicted label: 12 | Correct label: 12 | 12, 38, 13, 40, 15 |
| Predicted label: 40 | Correct label: 11 | 40, 12, 19, 25, 22 |
| Predicted label: 14 | Correct label: 14 | 14,  5, 34,  7, 12 |
| Predicted label: 28 | Correct label: 28 | 28, 29, 24, 11, 30 |






The model was able to correctly guess 6 of the 12 traffic signs, which gives an accuracy of 50%. This compares unfavorably to the accuracy on the test set of 94.5% Part of the problem could be the training on an unbalanced training set, as well as the image quality (resolution, graffity,  rotations and stretches). This could be mitigated by artificially extending the training set. 

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

Cell 19 in the Ipython notebook contains the barcharts of the softmax probabilities for each of the 12 predictions. The model seems "over-confident" in predicting a label, possibly due to overfitting. More (artificial) data and dropout could improve that.

The code for making predictions on my final model is located in the 19th cell of the Ipython notebook.


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

The neural network seems to use different contrast lines and shapes, some form of color thresholding based on the intensity of different pixels.

