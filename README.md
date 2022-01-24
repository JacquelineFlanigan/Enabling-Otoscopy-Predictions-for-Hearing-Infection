## Otoscopy Predictions

Using otoscopy images with neural networks to predict whether someone's ear has an infection. Here we will use some convolutional neural networks to predict if the otoscopy contains an ear that is abnormal (has an acute otis media) or normal (no infection).



##Methodology
The file used:

Otoscopy Dataset: https://www.kaggle.com/omduggineni/otoscopedata

Filtering and Transformation
In this dataset there are a total of nine categories but only two were used, as listed above, to better focus our model. In it were a total of 654 otoscopies that were were split into three groups to better work within the models used. The three groups were test, train and validations; each set containing around 39-41 images of cases of abnormal ears and around 178-179 images of cases of normal ears. 

##Models

Convolutional Neural Network
Our first model used was a generic convolutional neural network to see what the prediction accuracy would be. It scored relatively low with only a 41 percent score of predicting a normal or abnormal otoscope. So our next step was to tweak it to hopefully end up with better results. 


CNN Part Two
After changing how many epochs would be run, in addition to changing the weight class to balanced, we ran the CNN model once more. Unfortunately, it turned out that with the unchanged model, it would perform better before than it did now. Instead of being improving from 41%, it dropped to an accuracy score of 17%. This isn't too disheartning however, since usually the parameters of the generic model tend to work better. This did indicate that another type of neural network model may be better suited however.

VGG16
Here we chose to work with the VGG model since it is known for it's classification and localization of images, with it being pretrained on over 1,000 images. Using the generic model once more, we saw our accuracy improve greatly, going from the low 17% from before all the way up to around 80%! This was a great indication that our small dataset simply needed a model more aligned to our needs. 


##Confusion Matrix and LIME
Here we will see the confusion matrix of our modified CNN model to recap how we were able to improve our results. In addition to this, we will show one of the images that the LIME technique provided. It is important to note that while LIME was used, do to our dataset being smaller and used uncommon imagery, LIME was not terribly helpful but was interesting to use and was therefore left in. 

##Results
Our final results ended up with the VGG16 model being chosen for the best results and accuracy score. Eighty percent of the time, it will correctly predict whether an ear has an infection or doesnt. This model could be improved by adding more images to the dataset.
