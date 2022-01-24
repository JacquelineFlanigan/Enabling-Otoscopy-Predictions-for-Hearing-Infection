# Otoscopy Predictions

Using otoscopy images with neural networks to predict whether someone's ear has an infection. Here we will use some convolutional neural networks to predict if the otoscopy contains an ear that is abnormal (has acute otis media) or normal (no infection).


### Methodology
The file used:

Otoscopy Dataset: https://www.kaggle.com/omduggineni/otoscopedata

### Filtering and Transformation
In this dataset there are a total of nine categories but only two were used, as listed above, to better focus our model. In it were a total of 654 otoscopies that were split into three groups to better work within the models used. The three groups were test, train and validations; each set containing around 39-41 images of cases of abnormal ears and around 178-179 images of cases of normal ears. The train and test set do just as they infer with their labels for our model but with our validation set, it allows us the opportunity to improve the quality/quantity of the data. This is shown by reducing the bias and variance of the model since if the validation accuracy is greater than the training accuracy, then there is a higher chance of our model being overfitted to the data.

Also since acute otitis media doesn't depend on color for diagnosis and otoscopy imagery could be affected by the type of lightbulb it has, we have grayscaled all the images to hopefully simplify our data further. We have also used the Keras ImageDataGenerator to alter and increase the images in the train set to improve the performance of our model as it allows for better generalization and helps prevent overfitting.

### Models

## Convolutional Neural Network
Our first model used was a generic convolutional neural network to see what the prediction accuracy would be. We included early stopping within the model, which will stop the epochs (otherwise known as how many passes the dataset has gone forwards and backwards in its entirity) based on the conditions we've given it. We've also made sure to balance the class weights since our normal class includes more imagery than our abnormal. With this, and with our model knowing when to stop going through iterations when further progress isn't being made, we've reduced the learning rate and improved the quality of the model over all. Unfortunately, it scored relatively low with only a 41 percent score of predicting a normal or abnormal otoscope so the next step was to fine tune the parameters. 


## CNN Part Two

From what we were able to gather from our first model, it had stopped at the eighth epoch and after the fourth epoch, the accuracy declined significantly. Changing the model based on this information, we reran the model to see if this would improve the results. Unfortunately, with the changes made, our model returned with worse results with having an accuracy of around 17 percent. This isn't uncommon with testing models, as usually the generic model has already been modified for efficiency. This did lead to the conclusion that perhaps the CNN model is not the right type of model needed for this dataset.



###INCLUDE IMAGERY HERE OF BAD RESUTS FROM TEST 2



Although we have already come to the deduction that the CNN model isn't the model we will stick with, we did want to show visually as to how it isn't the right match for us. In the graph above, randomly selected images have had their hues changed and our model tried to identify which ones were which. Again, this just verifies that the CNN model isn't our best choice of model for the dataset.


## VGG16
The VGG model was the next choice for the dataset as it is known for it's classification and localization of images, with it being pretrained on over 1,000 images. Implementing this type of model, we saw our accuracy improve greatly going from 17 percent up to around 80 percent. This indicated that the VGG model is better over all for what we hope to achieve, which makes sense given the type of model it is. In addition to this, with this model we did implement data augmentation as to help flush out the imbalance of the classes as well as used the "adam" parameter since it is known for running well with sparse data. The predictive power of this model also faired well given the tricky dataset it dealt with, so therefore we can confidently choose this as the model we will use.


### LIME
Although we have chosen our model that works well with our dataset, there is another process that we wished to include. This is the LIME process (or "Local Interpretable Model-agnostic Explanations") that we have decided to incorporate into our project. LIME explains why our model comes to the predictive conclusions it does in images as well as tries to predict what the image is as well. Obviously, once more our small dataset will run into some issues since otoscopies are not as common as images of cats and dogs. Therefore, LIME was included due to it's fascinating process but is not very helpful at this time.  

###NCLUDE LIME IMAGERY HERE; TALK ABOUT IT WHAT THAT IMAGE IS BELOW?


### Results
As stated above, the VGG16 model was our best fit for our data with an accuracy of 80 percent in it's predictive power and doesn't take much time to run. Given the contraints of the size and the atypical nature of the dataset, this model worked well but could be improved in the future if given the chance to include more otoscopy imagery and same could be said for LIME.  
