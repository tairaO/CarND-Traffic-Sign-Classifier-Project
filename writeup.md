# **Traffic Sign Recognition**

## Writeup

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

[image1]: ./fig/sample_distribution.png "Sample distribution of training set"
[image2]: ./fig/grayscale.png "Grayscaling"
[image3]: ./fig/RGB_example.png "RGB space"
[image4]: ./fig/rate0.0001_epochs50_BATCH256_dropout1.0_2.png "Learning rate:0.0001"
[image5]: ./fig/rate0.001_epochs20_BATCH256_dropout1.0_2.png "Learning rate:0.001 Keeping rate:1.0"
[image6]: ./fig/rate0.01_epochs10_BATCH256_dropout1.0_2.png "Learning rate:0.001"
[image7]: ./fig/websample_0.png "Traffic Sign 1"
[image8]: ./fig/websample_1.png "Traffic Sign 2"
[image9]: ./fig/websample_2.png "Traffic Sign 3"
[image10]: ./fig/websample_3.png "Traffic Sign 4"
[image11]: ./fig/websample_4.png "Traffic Sign 5"
[image12]: ./fig/websample_5.png "Traffic Sign 6"
[image13]: ./fig/first_layer_feature_0.png "The first layer visualization"
[image14]: ./fig/first_layer_feature_1.png "The first layer visualization"
[image15]: ./fig/rate0.001_epochs120_BATCH256_dropout0.3_1.png "Keeping rate:0.3"
[image16]: ./fig/rate0.001_epochs70_BATCH256_dropout0.5_2.png "Keeping rate:0.5"
[image17]: ./fig/rate0.001_epochs50_BATCH256_dropout0.8_2.png "Keeping rate:0.8"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.


You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pickle library to load data and used numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data distributes.

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because grayscale images extract the important features in the images.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

Here is each space of a traffic sign image. As you can see, each space has almost same tendency and we, human, can classify the label from each of them. Therefore, I thought grayscale images have enough information to classify.

![alt text][image3]


As a last step, I normalized the image data because, if the set has 0 mean, the classification result becomes better. I calculated the mean of the grayscale intensity of training set and normalized images by using it.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x1 grayscale image   							|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| Dropout     	| keeping rate 0.5 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 5x5     	| 1x1 stride, same padding, outputs 10x10x16 	|
| Dropout     	| keeping rate 0.5 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 10x10x16 		|
| Fully connected		| outputs 120       									|
| Dropout     	| keeping rate 0.5 	|
| RELU					|												|
| Fully connected		| outputs 84       									|
| Dropout     	| keeping rate 0.5 	|
| RELU					|												|
| Fully connected		| outputs 43       									|
| Softmax				|       									|
| 				|       									|



#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used cross entropy as loss function and Adam optimizer because they are popular. I had to decide 4 hyperparameters, which are learning rate, epochs, batch size, and keeping rate of dropout. I decided them as follows.
First, I set the parameters as follows:

| conditions         		|     values	        					|
|:---------------------:|:---------------------------------------------:|
| (Learning rates, epochs)         		|  (0.01, 10), (0.001,20), (0.0001,50)  							|
| Batch sizes         		|  256  							|
| Keeping rates     	| 1.0 |

The results are below. As you can see, when learning rate was 0.01, it almost converged at 4th epochs, and improvement of training accuracy stops. On the other hand, when learning rate is 0.0001, it keeps improving after 50 epochs, so it is too late. Therefore, I decided learning rate was 0.001. After 13 epochs, it converges, and the training accuracy is more than 0.99. Also, the complexity of CNN is enough because the training accuracy was such a high value.

![alt text][image4]
![alt text][image5]
![alt text][image6]

Second, I checked keeping rates of dropout. I set parameters as follows.(I added epoch if the training rate is too low) When keeping rate was 0.3, it did not converge. On the other hand, when keeping rate was 0.8, it seems overfitting because the training accuracy was clearly high(0.99), but the validation rate is not. Therefore I decided that keeping rate is 0.5.:

| conditions         		|     values	        					|
|:---------------------:|:---------------------------------------------:|
| Learning rates        		|  0.001				|
| Batch sizes         		|  256  							|
| Keeping rates     	| 0.3, 0.5, 0.8, 1.0 |

![alt text][image15]
![alt text][image16]
![alt text][image17]
![alt text][image5]



#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.986
* validation set accuracy of 0.944
* test set accuracy of 0.937

I chose LeNet5 architecture. This is because LeNet5 has 5 layers and succeeded in recognizing handwritten characters. In this traffic signs classification tasks, the class label is only 43, so I thought it has enough parameters to overfit. After several test, its training accuracy was more than 0.99, so I judged the number of layer and size of convolutions is enough. Only deference is that I introduce dropout to prevent overfitting. It works very well because when keeping rate is 1.0([image5]), the validation rate was 0.91, however when keeping rate was 0.5([image16]), the validation rate is more than 0.94. Therefore, dropout improved overfitting.


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image7] ![alt text][image8] ![alt text][image9]
![alt text][image10] ![alt text][image11]![alt text][image12]

The second image might be difficult to classify because it is not in center. Also, fifth one might be difficult because it is slightly large in the image.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| Right-of-way at the next intersection      		| Right-of-way at the next intersection    									|
| Yield     			| Speed limit (30km/h)										|
| Speed limit (70km/h)					| Speed limit (70km/h)											|
| Road work	      		| Road work					 				|
| Stop			| Stop     							|
|Turn right ahead | Turn right ahead |


The model was able to correctly guess 5 of the 6 traffic signs, which gives an accuracy of 83%. This compares favorably to the accuracy on the test set. The discussion is in next section.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 14th cell(In[37]:) of the Ipython notebook.

For the first image, the model is almost sure that this is a priority road (probability of 0.88), and the image does contain a priority road. The top five soft max probabilities were

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .88         			| Priority road   									|
| .06     				| Double curve 										|
| .03					| Beware of ice/snow										|
| .01	      			| Pedestrians					 				|
| .00	      			| Roundabout mandatory					 				|


For the second image , the model is certainly sure that this is a Speed limit (50km/h)  (probability of 0.54), but the image is Yield. I think the reason why my model misclassified is that it contains noisy background(See figure below) and the sign is not in the center. The top five soft max probabilities were

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .54         			| Speed limit (50km/h)   									|
| .15     				| Speed limit (30km/h) 										|
| .10					| Priority road											|
| .06	      			| Yield					 				|
| .04				    | Keep right      							|

![alt text][image14]

For the third image , the model is almost sure that this is a speed limit (70km/h), and the image does contain it. Moreover, the second and third one are also speed limit signs because they has the same shape. The top five soft max probabilities were

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .83         			| Speed limit (70km/h)   									|
| .14     				| Speed limit (20km/h) 										|
| .03					| Speed limit (30km/h)										|
| .00	      			| Stop					 				|
| .00	      			|Keep left				 				|

For the forth image, the model is almost sure that this is a Road work, and the image does contain it. The top five soft max probabilities were

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .92         			| Road work   									|
| .03     				| Bicycles crossing 										|
| .01					| Bumpy road										|
| .00	      			| Dangerous curve to the right				 				|
| .00	      			| Beware of ice/snow				 				|

For the fifth image, the model is unclearly sure that this is a stop sign, and the image does contain it. I think the reason why my model classifies it unclearly is that the image is large and the outline focused out. The top five soft max probabilities were

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .31         			| Stop sign   									|
| .23     				| Speed limit (30km/h) 										|
| .08					| Yield											|
| .05	      			| Bumpy Road					 				|
| .05				    | Slippery Road      							|

For the sixth image, the model is almost sure that this is a turn right ahead, and the image does contain it. The top five soft max probabilities were

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .96         			| Turn right ahead   									|
| .00     				| Ahead only 										|
| .00					| Roundabout mandatory										|
| .00	      			| Priority road					 				|
| .00	      			| Speed limit (30km/h)					 				|


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

The visualization of the feature maps is below:

![alt text][image13]

Feature maps could extract the edges and the symbol. As you can see, extracted edge lines have specific orientation. For example, feature map 2 extracts horizontal line, 3 extracts 135 degree line, and 5 extracts 45 degree line. By means of that, it can recognize the outline of the signs. Also, feature map 0, 4 extract almost all of edges. On the other hand feature map 1 extracts only tilted edges. To sum up, each feature map has different selectivity of edges.
