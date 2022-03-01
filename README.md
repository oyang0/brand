# Brand Extrapolation
## Introduction
From May 2021 to September 2021, I worked as an AL/ML Scientist Summer Student (i.e. ATB 101 Data Scientist Student) at ATB Financial. There, I was tasked with researching and developing the Vendor Propensity Engine, a recommender system for Brightside by ATB. The problem was to help Brightside by ATB decide what local businesses to invite to their Friends with Benefits program, which the Vendor Propensity Engine solved by generating a list of recommended businesses for every businesses in the Friends with Benefits program. This involved obtaining relevant data from data warehouses, scrubbing data (e.g. filtering out duplicate merchants, filtering out merchants that are not local), exploring data (i.e. initial data analysis, exploratory data analysis), and modelling data (in this case, fitting a classifier implementing k-nearest neighbours vote).

A subproblem in the research and development of the Vendor Propensity Engine was that of determining if two merchants belonged to the same business, based on data such as their names (e.g. "A&W #1806," "A & W 1558/AWDN1"), descriptions (e.g. "fast food restaurants," "restaurants or eating places"), and categories (e.g. "store," "food"). My solution at the time was to generate numerical features from merchants' names with approximate string matching algorithms (e.g. Damerau–Levenshtein distance, Jaro-Winkler distance) then training a model on the numerical data available (in this case, fitting a random forest classifier, a logistic regression classifier, a support vector classifier, a linear classifier with SGD training, a decision tree classifier, and a multi-layer perceptron classifier and then selecting the model with the greatest average accuracy when subjected to 5-fold cross-validation).

Inspired by this subproblem, I tried to solve the problem of extrapolating from product names to brand names. This problem is similar to the subproblem in that if the brand names of products are correctly extrapolated, then determining if two products are from the same brand may be accomplished with simple string matching.

NOTE: due to the file size of the models, they have not been pushed to this repository. To download the missing folder containing the models, download the [ZIP file](https://drive.google.com/file/d/1S05fix48BAa6l5kU9FgeGHsSQ9kmgsNU/view?usp=sharing) containing the folder and extract said folder to this directory.
## Obtaining Data
The relevant data here are product names and their brand names. Such data may be obtained from Hackathon_Ideal_Data.csv available at [Store Transaction data](https://www.kaggle.com/iamprateek/store-transaction-data). The MBRD column contains brand names and the MRD column contains product names.
## Scrubbing Data
The brand names and product names are cleaned up as sentences are in "EDA: Easy Data Augmentation Techniques for Boosting Performance on Text Classification Tasks" by Jason Wei and Kai Zou. The training data is created from 70% of the data, the validation data is created from 21% of the data, and the test data is created from the remaining 9% of the data.

The training data is augmented with two data augmentation techniques. The first data augmentation technique is easy data augmentation from "EDA: Easy Data Augmentation Techniques for Boosting Performance on Text Classification Tasks" by Jason Wei and Kai Zou. The paper contains the authors' recommended usage parameters, which are also used here. The second data augmentation technique is brand replacement. This technique is to replace all instances of a product's brand name in its name with a brand name chosen at random. This technique does not conserve labels except when the chosen brand name is the product's brand name or when there are no instances of the product's brand name in its name to replace.
## Exploring Data
I performed some data exploration on the training data prior to data augmentation. The distributions explored were the number of words in product names, if product names contained their brand names, if product names started with their brand names, and if product names contained their brand names but did not start with them. These distributions were used to appraise the difficulty of extrapolating brand names from product names, informed the choice of model architecture, and informed the choice of baseline.
| # of Words    | Model         |
| ------------- | ------------- |
| 1             | 6764          |
| 2             | 2373          |
| 3             | 748           |
| 4             | 21            |
| 5             | 12            |
| 6             | 4             |

| Brand in Name | Count         |
| ------------- | ------------- |
| False         | 759           |
| True          | 9163          |

| Brand @ Start | Count         |
| ------------- | ------------- |
| False         | 903           |
| True          | 9019          |

| In, not Start | Count         |
| ------------- | ------------- |
| False         | 9778          |
| True          | 144           |
## Modelling Data
The problem of extrapolating from product names to brand names is a sequence-to-sequence problem. This problem may be thought of as "translating" product names to brand names. From the data exploration, it is known that some product names do not contain their brand names. To extrapolate from product names to brand names in these cases requires inforamation external to the product names. For these reasons, a pretrained sequence-to-sequence model may be good for this problem. Sequence-to-sequence models are trivially good for sequence-to-sequence problems and pretraining is a possible means of embedding inforamation external to the product names into the model. One such pretrained sequence-to-sequence model is the T5 model.

With two data augmentaion techniques, there were four training datasets: a training dataset with no data augmentation, a training dataset augmented with easy data augmentation, a training dataset augmented with brand replacement, and a training dataset augmented with both. To decide what dataset to use, the T5 model was finetuned on each of the four datasets then a dataset was selected by selecting the corresponding model with the greatest accuracy on the validation data.

Mainly due to time and space constraints, the selected optimizer for all finetuning was AdamW with a learning rate of 0.0001, the batch size for all finetuning was 64, and the number of epochs was selected so as to achieve as close to 60000 optimization steps as possible. Copies of the T5 model were saved every epoch. To generate brand names, greedy search was selected as the decoding method of choice.

From the data exploration, it is known that the majority of product names contain only a single word. Furthermore, it is known that the majority of product names start with their brand names. For these reasons, I selected a regular expression that extracts the first word in a string as the baseline.
## Interpreting Data
The T5 model finetuned on any of the four training datasets achieves greater accuracy on the validation data than the baseline. Interestingly, easy data augmentation and brand replacement do not appear to boost the performance of the T5 model finetuned for the problem of extrapolating from product names to brand names.
| Training Data | Model         | Optimizer     | Batch Size    | Epochs        | Val Accuracy  | Test Accuracy |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| N/A           | Baseline      | N/A           | N/A           | N/A           | 66.2%         | 66.8%         |
| None          | T5 Small      | AdamW, lr=1e-4| 64            | 273           | 99.4%         | 99.3%         |
| None          | T5 Small      | AdamW, lr=1e-4| 64            | 387           | 99.3%         | 99.3%         |
| EDA           | T5 Small      | AdamW, lr=1e-4| 64            | 8             | 99.3%         | 99.1%         |
| EDA           | T5 Small      | AdamW, lr=1e-4| 64            | 65            | 99.2%         | 99.2%         |
| BR            | T5 Small      | AdamW, lr=1e-4| 64            | 70            | 99.3%         | 99.3%         |
| BR            | T5 Small      | AdamW, lr=1e-4| 64            | 77            | 99.2%         | 99.2%         |
| EDA + BR      | T5 Small      | AdamW, lr=1e-4| 64            | 27            | 99.3%         | 99.2%         |
| EDA + BR      | T5 Small      | AdamW, lr=1e-4| 64            | 39            | 99.1%         | 99.1%         |
## Conclusion
Finetuning the T5 model on the training data with no data augmentation for 273 epochs yielded an accuracy of 99.3% on the test data. This is significantly better than the baseline and its accuracy of 66.8% on the test data.
