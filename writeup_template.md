# **Behavioral Cloning** 

## Writeup Template

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
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 5x5 filter sizes and depths between 24 and 64 (model.py lines 84-92)

The model includes several RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer (code line 81).

#### 2. Attempts to reduce overfitting in the model

The model contains 2 dropout layers in order to reduce overfitting (model.py lines 95 & 98).

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 113-117). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 106).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used the
data provided by udacity.

For details about how I created the augmented training data, see the next section.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

While testing, I started with a basic convolutional neural network while I worked on adding preprocessing steps and augmenting data.

I added more data by flipping images and adding the left and right camera images.

Since the top part of the image had several distracting elements (like trees) which weren't helpful to the model, and the hood of the car is in the bottom portion of the image, I ended up removing the top 70 pixels and the bottom 20 pixels.

I also initially tried out only using one color channel. After testing red, green, and blue color channels individually, I found that green worked best and I was able to make it past the first curve and onto the bridge. My mean squared error was around .07 and definitely overfitting as you can see from the increase in the validation set on the last few epochs:

<img src="loss_plots/plot6.png?raw=true">

Next, I moved on to trying better models. First, I tried out the model from comma.ai. Surprisingly, while I saw an improvement in the mean squared error, I still could only make it past the first first curve.

<img src="loss_plots/plot6.png?raw=true">

Then, I tried out nvidia's model, which also didn't show much improvement. My car kept running off the road at this point:

<img src="image_examples/crash-dirt.png?raw=true">

I suspected that this was due to a lack of grass on either side of the road since I was only using the green color channel at this point. After switching to using all three color channels, I was able to make it past this area.

I tinkered with adding in additional RELU activation layers and added two dropout layers to prevent overfitting.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

Here is a summary of the architecture:
```
____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to
====================================================================================================
lambda_1 (Lambda)                (None, 160, 320, 3)   0           lambda_input_1[0][0]
____________________________________________________________________________________________________
cropping2d_1 (Cropping2D)        (None, 70, 320, 3)    0           lambda_1[0][0]
____________________________________________________________________________________________________
convolution2d_1 (Convolution2D)  (None, 33, 158, 24)   1824        cropping2d_1[0][0]
____________________________________________________________________________________________________
activation_1 (Activation)        (None, 33, 158, 24)   0           convolution2d_1[0][0]
____________________________________________________________________________________________________
convolution2d_2 (Convolution2D)  (None, 15, 77, 36)    21636       activation_1[0][0]
____________________________________________________________________________________________________
activation_2 (Activation)        (None, 15, 77, 36)    0           convolution2d_2[0][0]
____________________________________________________________________________________________________
convolution2d_3 (Convolution2D)  (None, 6, 37, 48)     43248       activation_2[0][0]
____________________________________________________________________________________________________
activation_3 (Activation)        (None, 6, 37, 48)     0           convolution2d_3[0][0]
____________________________________________________________________________________________________
convolution2d_4 (Convolution2D)  (None, 4, 35, 64)     27712       activation_3[0][0]
____________________________________________________________________________________________________
activation_4 (Activation)        (None, 4, 35, 64)     0           convolution2d_4[0][0]
____________________________________________________________________________________________________
convolution2d_5 (Convolution2D)  (None, 2, 33, 64)     36928       activation_4[0][0]
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 4224)          0           convolution2d_5[0][0]
____________________________________________________________________________________________________
dropout_1 (Dropout)              (None, 4224)          0           flatten_1[0][0]
____________________________________________________________________________________________________
activation_5 (Activation)        (None, 4224)          0           dropout_1[0][0]
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 100)           422500      activation_5[0][0]
____________________________________________________________________________________________________
dropout_2 (Dropout)              (None, 100)           0           dense_1[0][0]
____________________________________________________________________________________________________
activation_6 (Activation)        (None, 100)           0           dropout_2[0][0]
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 50)            5050        activation_6[0][0]
____________________________________________________________________________________________________
activation_7 (Activation)        (None, 50)            0           dense_2[0][0]
____________________________________________________________________________________________________
dense_3 (Dense)                  (None, 10)            510         activation_7[0][0]
____________________________________________________________________________________________________
activation_8 (Activation)        (None, 10)            0           dense_3[0][0]
dense_4 (Dense)                  (None, 1)             11          activation_8[0][0]
====================================================================================================
Total params: 559,419
Trainable params: 559,419
Non-trainable params: 0
____________________________________________________________________________________________________
```

#### 3. Creation of the Training Set & Training Process

To augment the data set, I flipped images and angles. The first track goes in a circle making it biased to left turns. Flipping images doubles the training set, and gives the model right turns to learn from.

```
augmented_images, augmented_measurements = [], []
for image,measurement in zip(images, measurements):
    image_flipped = np.fliplr(image)
    measurement_flipped = -measurement

    augmented_images += [image_flipped]
    augmented_measurements += [measurement_flipped]
```

For example, here is an image that has then been flipped:

<img src="example_images/image_flipped.jpg?raw=true" width="50">

I also incorporated images from the left and right cameras to add even more images to the data set. I settled on a steering correction of 0.2. For the left camera image, I added the steering correction to the steering angle and subtracted it for the right image. This allowed me to triple the data set.

```
images = []
measurements = []
for batch_sample in batch_samples:
    steering_center = float(batch_sample[3])
    steering_left   = steering_center + STEERING_CORRECTION
    steering_right  = steering_center - STEERING_CORRECTION

    img_center = process_image(batch_sample[0])
    img_left   = process_image(batch_sample[1])
    img_right  = process_image(batch_sample[2])

    images += [img_center, img_left, img_right]
    measurements += [steering_center, steering_left, steering_right]
```

###### Left Image
<img src="example_images/image_left.jpg?raw=true" width="50">

###### Center Image
<img src="example_images/image_center.jpg?raw=true" width="50">

###### Right Image
<img src="example_images/image_right.jpg?raw=true" width="50">

After the collection process, I had 8036 data points.

I finally randomly shuffled the data set and put 20% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting.
I settled on 20 epochs since the mean squared error continued to decrease.

<img src="loss_plots/plot13.png?raw=true">

I used an adam optimizer so that manually training the learning rate wasn't necessary.
