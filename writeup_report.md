#**Behavioral Cloning** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode and it has been modified to convert RGB channel into HSV.
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results
* run1.mp4 final ouput video

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 

python drive.py model.h5 --epo 5


####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

I have used NVIDIA architecture and I found it perform really well on any dataset. As a datapreprocessing converting my data images using Cropping 2d (converting images from (160, 320,3) into (76, 320, 3)) and lambday layer for normalization (converting every pixel value in the range of -0.5 to 0.5)

Please see below the image of my ConvNet and it is developed using Keras Visualization API's.

![alt text](https://github.com/ankit2grover/CarND-Behavioral-Cloning-P3/blob/master/images/model.png)

####1. An appropriate model architecture has been employed
My model consists of below mentioned layers.
1) Cropping2D that crops 60 pixels from top and 24 pixels from the bottom of the image as that is irrelevant data information and only contains, trees. It doesn't help much in training the model. (model.py line 110)
2) Lambday layer that normalized the image pixels in the range of -0.5 to 0.5. (model.py line 111)
3) Convolution neural network with 5x5 and 3x3 filter sizes and depths of 24,36,48 and 64, 64 respectively.(model.py lines 112-119).
4) Fully Connected layers of hidden sizes 100, 50, 10.
4) Drop out in fully connected layers of probability 0.5, 0.2, 0.2 to avoid overfitting.

The model includes ELU layers to introduce nonlinearity for every convolution layereee, and the data is normalized in the model using a Keras lambda layer (code line 18).  I also tried with RELU activation and found that ELU layer provides better in comparison to RELU.

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in the fully connected layers in order to reduce overfitting with probability of 0.5, 0.2, 0.2 (model.py lines 21). 

The model was trained after shuffling and validated on different shuffled data sets to ensure that the model was not overfitting (code line 31). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, and the learning rate is 1e-3. Model performed better really well on the default learning rate.

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I struggled in creating my own data, that's why used udacity sample data to train the model.

I have converted RGB channel images into HSV in helpers/data.py file as I found that sharp turn dirty road is not detecting well with RGB images.

Also, I have removed 70% off steering angles == 0's to make sure that data is evenly distributed. As you can see below in the image that data is unevenly distributed and most of the data is at steering angle == 0's, it is making my model biased towards steering angles =0 ouput. 

![alt text](https://github.com/ankit2grover/CarND-Behavioral-Cloning-P3/blob/master/images/steering.png)

I just added some fake data by flipping the images 180 degree and teaching the model how image with right steering will look like.

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to make sure that model always drive the car on the centre of the road and is never underfit and overfit.

My first step was to use a use a cropping layer to make sure that only relevant road information is fed into the model.
Second, I designed the model similarily like NVIDIA paper. Also added drop out layers to avoid any overfitting.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set and a low mean squared error on the validation set. This implied that the model was underfitting. I adjusted the correction parameter to .25 (line 62 model.py)   

To combat the overfitting, I modified the model and added drop out in FCs layers

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.




####2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes.

![alt text](https://github.com/ankit2grover/CarND-Behavioral-Cloning-P3/blob/master/images/model.png)


![alt text][image1]

####3. Creation of the Training Set & Training Process

To augment the data set, I also flipped images and angles thinking that this would . For example, here is an image that has then been flipped:

![alt text](https://github.com/ankit2grover/CarND-Behavioral-Cloning-P3/blob/master/images/original.png)
![alt text](https://github.com/ankit2grover/CarND-Behavioral-Cloning-P3/blob/master/images/flip.png)


Randomly shuffled the data set and put 20% of the data into a validation set. 

Before augumentation I had 12058 number of data samples consist of centre, left and right camera images. Then I shuffled the data and divided 80% of training and 20% of validation data

Then I changed the brightness and flipped the centre images to feed more data so that my model should be able to perform well on any brightness well and flipping the images helped in generalizing my model well.

After augumentation, I had 559,419 number of data samples. I then preprocessed this data by using keras Cropping2D and lambda layers.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as validation accuracy doesn't change much after that. 

I used an adam optimizer so that manually training the learning rate wasn't necessary.