#**Traffic Sign Recognition** 

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

[image1]: ./examples/visualization.png "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/preprocess.png "Pre-process"
[image4]: ./examples/test_001.jpg "Traffic Sign Test 001"
[image5]: ./examples/test_002.jpg "Traffic Sign Test 002"
[image6]: ./examples/test_003.jpg "Traffic Sign Test 003"
[image7]: ./examples/test_004.jpg "Traffic Sign Test 004"
[image8]: ./examples/test_005.jpg "Traffic Sign Test 005"
[image9]: ./examples/test_images.png "Traffic Sign Test Images"
[image10]: ./examples/rotate.png "Rotate"
[image11]: ./examples/blur.png "Blur"
[image12]: ./examples/original_train.png "Original Train"
[image13]: ./examples/top5_001.png "Top5 Soft Max Test 001"
[image14]: ./examples/top5_002.png "Top5 Soft Max Test 002"
[image15]: ./examples/top5_003.png "Top5 Soft Max Test 003"
[image16]: ./examples/top5_004.png "Top5 Soft Max Test 004"
[image17]: ./examples/top5_005.png "Top5 Soft Max Test 005"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/gizmo/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is **34799**
* The size of the validation set is **4410**
* The size of test set is **12630**
* The shape of a traffic sign image is **(32, 32, 3)**
* The number of unique classes/labels in the data set is **43**

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data is distributed.  The 43 classes are not distributed equally.  There are a few classes that have a large number of samples and the majority have relatively few samples.

![alt text][image1]

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I tried to train without any pre-processing.  The images are full 3-channel images.  It got reasonable results but then I decided to convert the images to grayscale (resulting in 1-channel) while still retaining good validation accuracy.  By reducing to 1-channel grayscale we can train faster.

Here is an example of a traffic sign image before and after grayscaling and min-max normmalization.

![alt text][image2]

As a last step, I normalized the image data because during learning of the features from these images we want the features to fall within a common range rather than differ wildly from one feature to another.  Without normalization the computation of the gradients could be high for one feature and very low for another.  For this normalization process I chose to perform a minmax normalization which placed the images in the range 0.0 to 1.0 and then I followed that with a subtraction of 0.5 resulting in a range of -0.5 to +0.5.

I decided to generate additional data because as stated earlier the distribution is uneven.  The additional data generated is chosen such that we fill up the instances of a classes that have low counts resulting in a distribution that is more or less equal.  I do this by first determining the largest class. The largest class is class 2 with 2010 examples and this is the figure I try to approach when increasing the size of the other classes.   I can compute the number of additional images needed to generate for each original image of each class.  The data generated come in two forms: rotation and blur.  For rotations, I select from a uniform random distributions of angles between -80 and +80 degrees.  For blur, I select from a uniform random distribution of sigma between 0.1 and 1.75.  I continue selecting from these two forms of augmentation and fill the target number of images for a given image in a given class.  The total number of original images plus augmented images is 90297.

Here is an example of an original image and a rotated (augmented) image:

![alt text][image12]
![alt text][image10]

The difference between the original data set and the rotated data set is that the image is rotated around the center by approximately 30 degrees counter-clockwise.

Here is an example of an original image and a blurred (augmented) image:

![alt text][image12]
![alt text][image11]

The difference between the original data set and the augmented data set is blurred using a sigma parameter of about 1.75.


####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale min-max normalized image   							| 
| Convolution 5x5     	| 1x1 stride, same padding, outputs 28x28x64 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x16 				|
| Convolution 5x5     	| 1x1 stride, same padding, outputs 10x10x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				|
| Fully connected		| Outputs 200        									|
| RELU					|												|
| Dropout				| Keep probability 0.8        									|
| Fully connected		| Outputs 84        									|
| RELU					|												|
| Dropout				| Keep probability 0.8        									|
| Fully connected		| Outputs 43        									|
| Softmax				| 43        									|
|						|												|
|						|												|
 


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used a batch size of 128 and trained each batch using the Adam optimizer with a learning rate of 0.0007.  mu was 0.0 and sigma was 0.1.  I trained the model for 20 epochs which was a few epochs more than perhaps was necessary but chose a few more to ensure convergence was achieved.

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.985
* validation set accuracy of 0.935
* test set accuracy of 1.00

I started with the LeNet architecture but initially started without any preprocessing or augmentation.  This mean that I initially trained on the full 3-channel RGB.  This was found to be training reasonably well but I did not get the validation accuracy above 90%.  Training did take some time.  I then iterated to applying grayscale preprocessing.  The training were comparable to the non-preprocessing step so I opted to stay with the 1-channel grayscale as it was faster to train.  The next iteration involved normalizing the images to the range -0.5 to 0.5.  This helped a little in the validation accuracy but not by much.  I then adjusted the learning rate from the initial 0.01 to 0.001 to 0.0005 until finally settling on 0.0007.  At this learning rate I found it was more reliable avoiding ocillations in the validation accuracy and more likely to converge.
After introducing image augmentation the score seemed to improve further above 90%.  The augmentation was relevant to the traffic sign application because we filled in more examples for classes that were under represented.  By rotating and blurring we introduce a more robust set of images to train on (one that may appear in the real world).  I modified the architecture further to implement 0.8 drop out after the first two fully connected layers.  This really helped to push the accuracy higher and past 93%.  The dropouts helped to regularize the training and to reduce overfitting.  I ran training several times on this configuration and it always ended up over 93% and more often than not it reached 95% and so I was confident that the model was working well.


###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image9]

The fifth image might be difficult to classify because it is captured at an oblique angle.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Right-of-way at the next intersection      		| Right-of-way at the next intersection   									| 
| Priority road     			| Priority road 										|
| Yield					| Yield											|
| Speed limit (30 km/h)	      		| Speed limit (30 km/h)					 				|
| Road work			| Road work      							|


The model was able to correctly predict 5 of the 5 traffic signs from the internet.  This is certainly a plausible test accuracy considering the validation accuracy was relatively high at 93.5%. The validation accuracy is also not so high that we should also expect that other novel traffic signs can fail prediction.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a Right-of-way at the next intersection sign (probability of 0.9998), and the image does contain a Right-of-way at the next intersection sign. The top five soft max probabilities are below and it can be seen that the next highest soft max probability is quite small at 0.0001 for the Children crossing sign.  Thus the Right-of-way at the next intersection prediction is quiet robust though not perfect.  Its possible with some further augmentation and hyperparameter adjustments this could be improved.


| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.9998         			| Right-of-way at the next intersection   									| 
| 0.0001     				| Children crossing 										|
| 0.0000					| Beware of ice/snow											|
| 0.0000	      			| Double curve					 				|
| 0.0000				    | Slippery road      							|

![alt text][image13]

For the second image ... 

For the second image, the model is relatively sure that this is a Priority road sign (probability of 0.9999), and the image does contain a Priority road sign. The top five soft max probabilities are below and it can be seen that the next highest soft max probability is essentially 0.0000 for the Bicycles crossing sign.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.9999         			| Priority road   									| 
| 0.0000     				| Bicycles crossing 										|
| 0.0000					| Speed limit (50km/h)											|
| 0.0000	      			| Yield					 				|
| 0.0000				    | Speed limit (30km/h)      							|

![alt text][image14]

For the third image ... 

For the third image, the model is relatively sure that this is a Yield sign (probability of 1.0000), and the image does contain a Yield sign. The top five soft max probabilities are below and it can be seen that the next highest soft max probability is essentially 0.0000 for the Bicycles crossing sign.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0000         			| Yield   									| 
| 0.0000     				| Bicycles crossing 										|
| 0.0000					| Priority road											|
| 0.0000	      			| Bumpy road					 				|
| 0.0000				    | Traffic signals      							|

![alt text][image15]

For the fourth image ... 

For the fourth image, the model is relatively sure that this is a Speed limit (30km/h) sign (probability of 1.0000), and the image does contain a Speed limit (30km/h) sign. The top five soft max probabilities are below and it can be seen that the next highest soft max probability is essentially 0.0000 for the Speed limit (20km/h) sign.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0000         			| Speed limit (30km/h)   									| 
| 0.0000     				| Speed limit (20km/h) 										|
| 0.0000					| End of speed limit (80km/h)											|
| 0.0000	      			| Go straight or left					 				|
| 0.0000				    | Speed limit (70km/h)      							|

![alt text][image16]

For the fifth image ... 

For the fifth image, the model is relatively sure that this is a Road work sign (probability of 0.9899), and the image does contain a Road work sign. The top five soft max probabilities are below.  Note however that the next highest soft max probability for the General caution sign is comparatively higher than the other test examples at 0.0057.  This is also true of the subsequent soft max probabilities for this fifth image.  I suspect the reason for this higher level of uncertainty is due to the fact that the test image for the Road work sign is taken at an oblique perspective rather than more or less head on like the other test images.  One way to improve this case is to include affine transformations to the augmentation training set to train the system on more examples of these types of oblique views.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.9899         			| Road work   									| 
| 0.0057     				| General caution 										|
| 0.0035					| Traffic signals											|
| 0.0006	      			| Road narrows on the right					 				|
| 0.0003				    | Dangerous curve to the right      							|

![alt text][image17]

