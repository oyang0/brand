# Brand Extrapolation
## Introduction
From May 2021 to September 2021, I worked as an AL/ML Scientist Summer Student (i.e. ATB 101 Data Scientist Student) at ATB Financial. There, I was tasked with researching and developing the Vendor Propensity Engine, a recommender system for Brightside by ATB. The problem was to help Brightside by ATB decide what local businesses to invite to their Friends with Benefits program. The Vendor Propensity Engine solved this by generating a list of recommended businesses for every businesses in the Friends with Benefits program. This involved obtaining relevant data from data warehouses, scrubbing data (e.g. filtering out duplicate merchants, filtering out merchants that are not local), exploring data (i.e. initial data analysis, exploratory data analysis), and modelling data (in this case, fitting a classifier implementing k-nearest neighbours vote).

A subproblem in the research and development of the Vendor Propensity Engine was that of determining if two merchants belonged to the same business, based on data such as their names (e.g. "A&W #1806," "A & W 1558/AWDN1"), descriptions (e.g. "fast food restaurants," "restaurants or eating places"), and categories (e.g. "store," "food"). My solution at the time was to generate numerical features from merchants' names with approximate string matching algorithms (e.g. Damerau–Levenshtein distance, Jaro-Winkler distance) then training a model on the numerical data available (in this case, fitting a random forest classifier, a logistic regression classifier, a support vector classifier, a linear classifier with SGD training, a decision tree classifier, and a multi-layer perceptron classifier and then selecting the model with the greatest average accuracy when subjected to 5-fold cross-validation).

Inspired by this subproblem, I tried to solve the problem of extrapolating from product names to brand names. This problem is similar to the subproblem in that if the brand names of products are correctly determined, then determining if two products are from the same brand may be accomplished with string matching.
## Obtaining Data
The relevant data here are product names and their corresponding brand names. Such data may be obtained from Hackathon_Ideal_Data.csv available at [Store Transaction data](https://www.kaggle.com/iamprateek/store-transaction-data). MBRD values are brand names and MRD values are product names.
## Scrubbing Data
The brand names and product names are cleaned up as is done in "EDA: Easy Data Augmentation Techniques for Boosting Performance on Text Classification Tasks" by Jason Wei and Kai Zou. The training data is created from 70% of the data. The validation data is created from 21% of the data. The test data is created from the remaining 9% of the data.

The training data is augmented with two data augmentation techniques to create three additional datasets. The first data augmentation technique is easy data augmentation from "EDA: Easy Data Augmentation Techniques for Boosting Performance on Text Classification Tasks" by Jason Wei and Kai Zou. The second data augmentation technique is to replace all instances of a product's brand name in its name with a randomly sampled brand name from the training data. This data augmentation technique does not conserve labels except when the randomly sampled brand name is the product's brand name and when there are no instances of the product's brand name in its name to replace.
## Exploring Data
First, I explored the distribution of the number of words in product names. The intent here was to partially determine the difficulty of extrapolating brand names from product names. Presumably, the greater the number of words in a product name, the more difficult it will be to extrapolate to its brand name on average. Fortunately, the majority of product names are contain only 1 or 2 words.
| # of Words    |Model          |
| ------------- | ------------- |
| 1             | 6764          |
| 2             | 2373          |
| 3             | 748           |
| 4             | 21            |
| 5             | 12            |
| 6             | 4             |

Second, I explored the distribution of if a product's brand name is in its name. The intent was not only to partially determine the difficulty of extrapolating from product names to brand names, but also to inform my choice of model architecture. Presumably, extrapolating a product's brand name from its name when it does not contain said brand name is a significantly difficult problem, one that necessitates information beyond the product name to solve. A model architecture that has access to such information may be necessary for these cases, if they exist. Unfortunately, such cases do exist. Fortunately, they are in the minority.
| Brand in Name | Count         |
| ------------- | ------------- |
| False         | 759           |
| True          | 9163          |

Third, I explored the distribution of if a product's name starts with its brand name. The intent was, once again, not only to partially determine the difficulty of extrapolating brand names from product names, but also to inform my choice of a baseline. Presumably, the more likely it is that a product name starts with its brand name, the easier it will be to extrapolate to its brand name. In addition, the more likely it is that a product name starts with its brand name, the more viable the strategy of extracting some number of words from the start of a product name as its predicted brand name would be. Fortunately, the majority of product names do start with their brand names.
| Brand @ Start | Count         |
| ------------- | ------------- |
| False         | 903           |
| True          | 9019          |

Lastly, I explored the distribution of if a product's brand name is in its name but its name does not start with it. The intent was to partially determine the difficulty of extrapolating from product names to brand names. If cases of a product's brand name being in its name but its name not starting with it do exist, then it demonstrates that the problem of extraploating to brand names is not as easy as always extracting some number of words from the start of a product's name as its predicted brand name. Unfortunately, such cases do exist. Fortunately, they are in the minority.
| In, not Start | Count         |
| ------------- | ------------- |
| False         | 9778          |
| True          | 144           |
## Modelling Data
I tried to extrapolate from product names to brand names with three different pretrained transformers: GPT2, GPT2 Medium, and GPT Neo 125M. These models were selected in the hopes that their pretraining may serve as information external to what is contained in product names. This may permit these models to handle cases in which products names do not contain their corresponding brand names. These models were selected in particular because GPT2 and GPT2 Medium are different in model size while GPT2 and GPT Neo 125M are different in the corpus they were pretrained on. My selected baseline extracts the first word in a product's name as its predicted brand name.
## Interpreting Data
| Training Data | Model         | Optimizer     | Batch Size    | Epochs        | Val Accuracy  | Test Accuracy |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| N/A           | Baseline      | N/A           | N/A           | N/A           | 66.2%         | 66.8%         |
| Data #1       | GPT2          | SGD, lr=1e-3  | 64            | 30            | 79.2%         | 80.8%         |
| Data #2       | GPT2          | SGD, lr=1e-3  | 64            | 5             | 79.5%         | 80.8%         |
| Data #3       | GPT2          | SGD, lr=1e-3  | 64            | 6             | 74.8%         | 77.0%         |
| Data #4       | GPT2          | SGD, lr=1e-3  | 64            | 3             | 0.0%          | 0.2%          |
| Data #1       | GPT2 Medium   | SGD, lr=1e-3  | 64            | 30            | 0.0%          | 0.0%          |
| Data #2       | GPT2 Medium   | SGD, lr=1e-3  | 64            | 5             | 0.0%          | 0.0%          |
| Data #3       | GPT2 Medium   | SGD, lr=1e-3  | 64            | 6             | 0.0%          | 0.0%          |
| Data #4       | GPT2 Medium   | SGD, lr=1e-3  | 64            | 3             | 0.0%          | 0.0%          |
| Data #1       | GPT Neo 125M  | SGD, lr=1e-3  | 64            | 30            | 0.0%          | 0.0%          |
| Data #2       | GPT Neo 125M  | SGD, lr=1e-3  | 64            | 5             | 0.0%          | 0.0%          |
| Data #3       | GPT Neo 125M  | SGD, lr=1e-3  | 64            | 6             | 0.0%          | 0.0%          |
| Data #4       | GPT Neo 125M  | SGD, lr=1e-3  | 64            | 3             | 0.0%          | 0.0%          |

## Conclusion
Finetuning GPT2 on the training data augmented with easy data augmentation yielded the greatest accuracy on the validation data with 79.5%. This model achieved an accuracy of 80.8% on the test data. This is significantly better than the baseline and its accuracy on 66.8% on the test data. Nonetheless, an accuracy of 80.8% is not excellent. If time permitted, some interesting avenues of research could be to pretrain models on a corpus specific to brands and products, finetune GPT2 Medium and GPT Neo 125M with larger batch sizes, and finetune all the models for longer. 
