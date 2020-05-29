**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./output_images/center_image.png "Center camera image"
[image2]: ./output_images/flipped.png "Flipped image"
[image3]: ./output_images/cropped.png "Cropped image"
[video1]: ./run1.mp4 "Output video"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* run1.mp4 is the output video of car in autonomous mode
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5 run1
```
and the video was recorded in the run1 folder. The video.py file was then used to compile the images in the run1 folder into a .mp4 file.

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the preprocessing of the data and the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

I followed the model suggested in the classroom. The Nvidia model with a few additions provided very good results.
My model consist of an initial Lambda layer for normalizing and correcting for the mean for each image. The next layer is a cropping layer which cuts the image along the vertical axis as shown in the image below.

![alt text][image3]

This helps in faster processing of the images and also prevents the neural network from focussing on features development outside the concern of lane following.
My model consists of a convolution neural network with 5x5 and 3x3 filter sizes and depths between 24 and 64 (model.py lines 18-24) with strides of (2,2). These help in systematically identiying the features in the images and can capture even curved lines, lack of lines and also the bridge in the track. 

RELU and dropout layers are added to increase non linearity and reduce bias in the model.

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting. EarlyStopping callback has been added with the intention of monitoring 'val_loss' output and terminate model training when the model starts overfitting. High value for number of epochs was chosen so that the model trains and stops before it overfits.

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

![alt text][video1]

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually. Batch_size and Dropout rate were adjusted to achieve low mse values and optimum val_loss. Sometimes, intermediate training models are better than the final training model achieved. So ModelCheckpoint parameter is added in order to evaluate all intermediate models for the best fit.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road and also data from both the tracks. 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to use convolutional layers to get the features in the images and the correlation between the image and the output steering angle. Multiple convolutional layers were chosen to derive advanced features like turning edge lines, bridges and the chicane strips. Dropouts and RELU actication layers were added to randomize the training by introducing non linearity and making the network robust. Finally, three fully connected layers were added to get the final output prediction.

The initial model consisted of just one convolutional layer and one dense layer at the end. This predicted straight line motion, but with increasing data complexity, more layers were required. The NVIDIA network was a good start for the final network and some modifications were made in order to improve its predictions.

The initial data to the network was only the center camera images and the steering angles. To augment these observations and improve the car's capacity to better predict the edges of the road, I made use of the left and right camera images as well. The corresponding steering angles for these images were given a +/- offset value of 0.2 to the steering angle respectively. This method helped in improving the performance of the vehicle especially at the spot where definitive lane lines disappear in some parts of the track.

The usage of callbacks and dropout layers helped in combating overfitting of the model.

At times, when the car ran off the track, I collected more data in those places to augment the training data and better improve the predictions. This can happen as the split between the training and validation data is random and so more data collection of a particular scenario helps in telling the model to focus more on such features.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.


#### 2. Creation of the Training Set & Training Process

All the data collected for this project is in the folder './refined_data'.
To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image1]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would try to get back to the center as much as possible. I additonally recorded more data near different features like bridge and other parts to improve predictions in such parts of the track.

Then I repeated this process on track two in order to get more data points.

The dataset was further augmented by using left and right camera images and the corresponding angles were given an offset of +/- 0/2. 

To augment the data sat, I flipped images and angles thinking that this would balance the dataset with respect to right and left turning. For example, here is an image that has then been flipped:

![alt text][image2]

As the images were read in by using cv2 library, they were preprocessed into the required RGB format.
Preprocessing of the data was done in the model pipeline using Lambda and Cropping2D functions to make training faster and predictions focus on the road features only as much as possible. 

I finally randomly shuffled the data set and put 30% of the data into a validation set using the train_test_split function from the sklearn module.

As the data collected was too large, I made use of generators for batch processing for the neural network. The output of the generator was again shuffled to make the model more robust.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The use of callbacks on the val_loss parameter helped me set a high epoch value and thus remove that from the tuning parameter set. I used an adam optimizer so that manually training the learning rate wasn't necessary. I used a batch size of 128 as it proved to provide superior results and faster training between the values of 32,64 and 128.
