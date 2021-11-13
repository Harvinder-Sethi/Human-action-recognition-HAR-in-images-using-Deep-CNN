# Human-action-recognition-HAR-in-images-using-Deep-CNN

# Problem Definition
To develop a Deep Convolutional network to identify the actions of people from sill images (Human Action Recognition). HAR has wide range of applications in IOT, streaming web, monitoring/security. Mostly HAR is performed with videos to achieve the above, but due to less compute, resources, low end devices also present in market we may have to do it on still images as well, and that is what we will be designing.


# Exploratory Data Analysis
The dataset of images consists of 9532 images including test and train along with their action and action_class label values in csv format files. This problem is called as Multi-Class Multi-Label Image classification problem.
The ‘action’ and ‘action_class’ target classes consist of 21 and 5 labels respectively. To explore the distribution in train dataset, I have plotted a histogram for both Fig1.2, we can see classes slightly. I also visualised and printed the image, and we notice that all images are of variable size, actions, and action_class types Fig1.3.

# Evaluation Framework 
Evaluation metric- we can see classes slightly imbalance and ideally will take f1 score (balance between precision & recall) into consideration. But we choose Accuracy (Categorical Accuracy) as it’s a classification task and its very intuitive and even the classes are not that imbalanced.
Optimizer- It can be said that the choice of optimiser can dramatically change the performance and time of the model. One argument about the optimizers can be easily seen that SGD better generalizes than ADAM but ADAM converges faster [3]. We tried Adam optimizer with InverseTimeDecay learning rate then tried SGD with same lr and default momentum, we noticed Adam converges faster, and is mostly considered in image classification tasks. and according to our problem of image classification and less time and resource we have selected ADAM with decaying learning rate=0.0001 using InverseTimeDecay.
Cost Function/Loss- Selecting a reasonable cost function according to problem is very important and its usually differential in deep learning problems. As our problem is not binary classification rather it is categorical as said before being multi- class (more than 2) and multi- label. We selected CategoricalCrossentropy.


# Approach & Diagnostic Instrumentation
As this problem of HAR is not novel and has many solutions on the web, I did extensively investigate research paper [1] and paper [2]. Based on the findings and understanding as this is an image classification task with roughly 10000 images (less than usual), we can come up with a Deep CNN as baseline model and then tune it accordingly to our desired result. For our problem we are taking a base Accuracy score in between 85-90%.
As you can see in Fig1.1 we first started with Function Definition, we created call back using keras.callback.earlystopping and this call back will stop the training when there is no improvement in the VAL_LOSS for 20 consecutive epochs. We then created a plotter function to print the history, class1 train-validation accuracy and class2 respectively. We did our EDA as explained above in that we used Label encoding classes to change them to unique numerical values for further evaluation. Further splitting the df_train into train-validation-test data-frame implementing hold-out validation (60-20-20% split). We then moved to Creating our Custom Data Generator, for each split and defined data augmentation inside that like flipping left-right, rotating, and shifting the images so that we can have an augmented and non-augmented training data set while setting up generator for train-test-validate splits.

# Experiments, Tuning & Analysis
In the next step in Fig1.1, based on our research form [1] [2] we finalised our baseline model to be of structure as of RESNET50 (also famous for solving vanishing gradient problems via its skip connection and traversing weight during back prop. And also can be extended in future for very larger image classification tasks) but training it/all layers fully with weights=None and custom hyperparameter tuning. We imported that from keras.applictaions and then added a globalaveragepooling2d layer (takes the average of all values acc. to the last axis returning (n_samples, last_axis)). Then we appended our output layer i.e., action and action class and named it as base_resnet, we fit the model on non-augmented data and for 30 epochs, we saw huge overfitting as training accuracy went 100% and validation roughly 20-25%. For Hyper-parameter tuning the base_resnet,

We then tried the same base_resnet model with data augmentation to see if that reduces overfitting. When we compiled this model with same Evaluation framework as described above and fit the model we named as as base_resnet_aug. we noticed roughly 10% increase in accuracy but still huge overfitting.

We then added two separate flatten and dropout (0.3) layers for both output classes right after the globalaveragepooling2d layer in the structure, we named the model as base_resnet_aug_drop and compile and fit the model with same configuration and data augmentation for 30 epochs to see if the overfitting reduces, we can see it does reduces the overfitting and generalises well (again approx. 10% increase in accuracy, around 40%) but the accuracy is still very poor with higher epochs also it still overfits.

We then added l2 Regularisation to the two output class layers with alpha – 1e-05 (trying alpha=0.001 model showed underfitting) to the above model and named it as base_resnet_aug_drop_reg and compile and fit the model with same configuration and data augmentation for 50 epochs this time (as this seems to be our last approach for Hyper parameter tuning) to see if the overfitting reduces, we noticed the model  gave approx. training accuracy of 75% while validation to be just 30% for action class and 50% for action_class.
We will now move to different approach as we need to move more up to our initial expectation performance.

We now moved to Resnet50 with, weights= ‘imagenet’ (pretrained weights on imagenet problem for better generalisation of this model )+ Data Augmentation as hyper parameter tuning and rest same structure. We compile the model we named as resnet_ntransfer and then fit the model and run for 50 epochs. We noticed, a drastic increase in the accuracy and model generalisation. Action accuracy turns out to be 82% approx. while Action_class 90%. 

We then added l2 Regularisation to the two output class layers with alpha – 1e-05, to the above model and named it as resnet_ntransfer_reg and compile and fit the model with same configuration and data augmentation for 30 epochs this time., We noticed, a small increase in the accuracy and smoother model generalisation and graph. Now, Our Action accuracy turns out to be 86% approx. while Action_class turns out to be 95 % [Fig1.4]
Ultimate Judgment & Limitations
As we are close to our initial goal/expectation of performance of our model [1][2] we can stop here and finalise this as the best model we have, namely resnet_ntransfer_reg. We have then save the model using keras.save() function, so as we don’t loose our work and can load and use this model anytime in future. Now we can move to, Evaluating the final model on the testing data generator split, we can see we have got Action accuracy of 87.1% and Action_class accuracy of 94.09% [Fig1.5]. 
Then for the Predictions on Final Model, we created a 2d- numpy array of size test data size and input dimensions. We the read and iterate each file in the test data (skipped 1 corrupted file Img_3201.jpg) open the image, convert that image to numpy array, resize that image array as array(image_data, input dim, then normalise the image array data pixels by (data/255.0), and store that data back in the numpy array we created at start, then Predicting (model.predict(image array data))and storing the action and action_class o/p preserving the indexes.
For Conversion And Inverse Transform we converted form categorical to numerical(np.argmax) then Decoding the labels encoded at start using same label_encoder.inverse_transform for both the classes respectively. Then we converted both action and action_class arrays to data frame and concatenated this data frame with the test_df and we named it as final_df_predicted [Fig1.6]. We then write to csv 
( df.to_csv(file_path) ) and named it as s3821903_predictions.csv.

Coming to the Limitations, we can say that it’s a complex model with 5 stages each having different pooling, conv2d, batch normalisation, activations and approx. 23 million trainable parameters hence it is tough to understand the structure completely. It can only work best when image size is equal or less than 224*224*3. This model is computationally very greedy and expensive when extended for similar problems with larger data. The duration of the network is unknown: Reducing the network to a certain value of the sampling error implies completing the training[11].

# References:

1.	B. Yao, X. Jiang, A. Khosla, A.L. Lin, L.J. Guibas, and L. Fei-Fei. Human Action 
Recognition by Learning Bases of Action Attributes and Parts. Internation Conference on Computer Vision (ICCV), Barcelona, Spain. November 6-13, 2011.

2.	Serpush, F., Rezaei, M. Complex Human Action Recognition Using a Hierarchical Feature Reduction and Deep Learning-Based Method. SN COMPUT. SCI. 2, 94 (2021). https://doi.org/10.1007/s42979-021-00484-0

3.	Understanding ResNet50 architecture. (2021). Retrieved 1 September 2021, from https://iq.opengenus.org/resnet50-architecture/

4.	A 2021 Guide to improving CNNs-Optimizers: Adam vs SGD. (2021). Retrieved 7 September 2021, from https://medium.com/geekculture/a-2021-guide-to-improving-cnns-optimizers-adam-vs-sgd-495848ac6008

5.	Deep Learning using Transfer Learning -Python Code for ResNet50. (2021). Retrieved 9 September 2021, from https://towardsdatascience.com/deep-learning-using-transfer-learning-python-code-for-resnet50-8acdfb3a2d38

6.	Jayaraman, A., Bevilacqua, H., & Imran, A. (2021). What is the use of verbose in Keras while validating the model?. Retrieved 9 September 2021, from https://stackoverflow.com/questions/47902295/what-is-the-use-of-verbose-in-keras-while-validating-the-model

7.	Brownlee, J. (2021). Display Deep Learning Model Training History in Keras. Retrieved 10 September 2021, from https://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/

8.	AI Powered Search for Extra ... - Towards Data Science. Retrieved 10 September 2021, from https://towardsdatascience.com/ai-powered-search-for-extra-terrestrial-intelligence-signal-classification-with-deep-learning-6c09de8fd57c

9.	Understanding and Coding a ResNet in Keras | by Priya… Retrieved 11 September 2021, from https://towardsdatascience.com/understanding-and-coding-a-resnet-in-keras-446d7ff84d33

10.	keras, w., Snoopy, D., Gervais, N., Soul, T., & Cerliani, M. (2021). what is the difference between Flatten() and GlobalAveragePooling2D() in keras. Retrieved 10 September 2021, from https://stackoverflow.com/questions/49295311/what-is-the-difference-between-flatten-and-globalaveragepooling2d-in-keras/63502664

11.	What are the disadvantages of using residual neural network?. (2021). Retrieved 18 September 2021, from https://www.quora.com/What-are-the-disadvantages-of-using-residual-neural-network

# APPENDIX 

Fig.1.1

![image](https://user-images.githubusercontent.com/29870980/141604442-56912b93-6102-45f2-b36d-5d029eecf94d.png)

Fig.1.2

![image](https://user-images.githubusercontent.com/29870980/141604451-93e18bca-2cd4-4348-92db-fabf9144c259.png)
![image](https://user-images.githubusercontent.com/29870980/141604455-fee6721a-acd8-433a-84e4-eaf239165cf4.png)


Fig.1.3
![image](https://user-images.githubusercontent.com/29870980/141604457-f8466f4e-e32d-4821-afc1-cdfce11dedbb.png)
![image](https://user-images.githubusercontent.com/29870980/141604461-63446d8f-e8dd-4088-aae4-fbc8823aca01.png)

  
Fig.1.4

 ![image](https://user-images.githubusercontent.com/29870980/141604465-f9fc9ccd-d325-4d4d-9c21-4407098422f3.png)
![image](https://user-images.githubusercontent.com/29870980/141604466-9a48161d-051c-4427-9ce5-0c22fb6a86c7.png)

Fig.1.5
![image](https://user-images.githubusercontent.com/29870980/141604470-f6cde448-693e-4e56-8a63-6b06ea3a639f.png)

 
Fig1.6
![image](https://user-images.githubusercontent.com/29870980/141604425-4b840c10-2b4f-4d92-a9ec-eb0a5bcb3b12.png)
