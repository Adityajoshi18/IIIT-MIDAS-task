# IIIT-MIDAS-task

This task was done with tensorflow.keras framework.

#Data Preprocessing
At first,the dataset was imported and converted into pandas dataframe.Then the columns which didn't effect the prediction of the product_category_tree were dropped.The product_category_tree column was processed by splitting the text by pattern('>>') and by using .strip() method,the unnecessary characters were removed.Then some categories whose value_counts were less than 100 were dropped because of lack of data.The categories were then integer coded and after that were one-hot encoded.
The 'description' column was also preprocessed by removing punctuations in between the characters.The keras tokenizer function was used to tokenize each word in the 'description' column.Sentences were represented as a sequence of integers.By calculating the max length of any sentence in that column,all other sequences were padded accordingly.

#Creating training, validation, test data
As there was no test data,the dataset was splitted using train_test_split function.The test size was kept 20% of the dataset.The remaining part was again splitted into training data and validation data,with validation set being 20% of that part.

#Model
An embedding layer was used first to create word embeddings.Then GlobalMaxPooling was used to minimize the dimensions.Then the 2d matrix was converted into 1d by a dense layer.2 more dense layers were used and the categories were predicted using softmax activation function.The last dense layer had nodes equal to the categories in the dataset.The optimizer used was Adam and loss used was categorical_crossentropy.The callbacks used were Earlystopping and model checkpoint(to get the best weights).The weights were saved into a file with file path mentioned.The model was then fitted on training data and validated on validation data.

#Prediction
The values were predicted and the resultant values were numpy array with 18 elements,each being the probability of each node of last dense layer.Highest probability node was calculated using argmax function.Likewise for all elements the predicted category was calculated.The predicted and test categories were compared and model was evaluated.the accuracy was around 96%.But this accuracy was average accuracy.As this dataset was unbalanced, some categories with very less value_counts would have been wrongly predicted even though it wasn't reflecting on average accuracy.So the accuracy for each product_category_tree was calculated.

To improve the accuracy of the model,I would use a pre trained word embeddings like Glove or Word2vec and then use those embeddings with the keras embedding layer.

I would also use kfold cross validation to improve the accuracy.





