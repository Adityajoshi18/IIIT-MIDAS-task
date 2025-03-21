# IIIT-MIDAS Task

This project focuses on text classification using deep learning techniques implemented with the tensorflow.keras framework. The primary objective is to classify products based on their textual descriptions into appropriate product categories.

## Dataset and Preprocessing

- The dataset was imported and converted into a Pandas DataFrame.
- Non-relevant columns that did not contribute to the prediction of product_category_tree were dropped.
- The product_category_tree column was processed:
    - Split using the delimiter >>.
    - Unnecessary characters were stripped.
    - Categories with fewer than 100 samples were removed due to insufficient data.
    - Categories were integer-encoded and subsequently one-hot encoded.
- The description column was processed:
    - Punctuation was removed.
    - Tokenization was performed using Keras Tokenizer.
    - Sentences were converted into sequences of integers.
    - Sequences were padded to a uniform length based on the maximum sequence length in the dataset.

## Data Splitting

- As there was no predefined test set, the dataset was split using train_test_split:
    - 80% training, 20% test.
    - The training set was further split into 80% training and 20% validation.

## Model Architecture

- The neural network comprises the following layers:
    - **Embedding Layer**: Generates word embeddings for the input sequences.
    - **GlobalMaxPooling**: Reduces dimensionality by selecting the maximum value for each feature map.
    - **Fully Connected Dense Layers**:
        - Converts 2D matrix into a 1D representation.
        - Two additional dense layers for feature extraction.
        - The final dense layer uses softmax activation for multi-class classification.
    - **Optimizer & Loss Function**: 
        - Adam optimizer.
        - Categorical Cross-Entropy loss function.

## Training and Evaluation

- **Callbacks Used**:
    - EarlyStopping: Monitors validation loss to prevent overfitting.
    - ModelCheckpoint: Saves the best model weights.
- The model was trained on the training set and validated using the validation set.
- Predictions were generated and evaluated using argmax to determine the highest probability category.
- The model achieved an **average accuracy of ~96%**.
- Due to class imbalance, per-category accuracy was computed to assess performance across different classes.

## Results and Improvements

- The model performed well on majority classes but struggled with underrepresented categories.
- **Improvements to consider**:
    - Utilizing pre-trained word embeddings such as GloVe or Word2Vec to enhance semantic understanding.
    - Implementing K-Fold Cross-Validation to better generalize across different dataset splits.

## Future Enhancements

- Experimenting with transformer-based architectures like **BERT** for improved contextual understanding.
- Implementing data augmentation techniques for underrepresented categories.
- Fine-tuning hyperparameters for better performance.

## Requirements

To run this project, install the following dependencies:

```bash
   pip install tensorflow pandas numpy scikit-learn
```

## Usage

- Run the Jupyter Notebook
- Modify hyperparameters and dataset processing as needed to fine-tune performance.






<!-- This project aims to build a machine learning model for paraphrase detection. Paraphrase detection is the task of determining whether two given sentences convey the same meaning. The model is trained and evaluated on a dataset consisting of pairs of sentences labeled as paraphrases or non-paraphrases.

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

I would also use kfold cross validation to improve the accuracy. -->





