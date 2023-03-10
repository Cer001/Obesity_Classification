# Obesity Classification

## Business Objective: 
To improve marketing efforts for obese and overweight customers based on lifestyle metrics, age, and gender for Accenture Marketing.

### Questions we are hoping to answer:
#### How can we identify if someone is Obese without asking for their height and weight?

#### What lifestyle characteristics are the most important in predicting obesity?

#### Can we predict if someone may be obese based on their search history profile?

We want to answer these questions because marketing is difficult with irrelevant data. We want to transform irrelevant data we have about someone's lifestyle, derived from search history or a survey, to useful data we can leverage in a predictive model for classifying obesity. If we are able to classify obesity without asking for the height or weight of an individual, we can use previously useless data we have stored about that individual to more effectively market to them. For example, knowing someone's propensity to snack turns out to be one of the strongest predictors of obesity. 

Accenture marketing can use this machine learning algorithm to improve marketing segmentation for Obese, Overweight, Normal, and Underweight individuals either by using the most important predictive features output by this algorithm or by using the algorithm to predict obesity based on lifestyle characteristics of a person derived from their search history profile. We will use machine learning to create a predictive model to classify for obesity based on these metrics:

### Legend:
Frequent consumption of high caloric food (FAVC)

Frequency of consumption of vegetables (FCVC)

Number of main meals (NCP)

Consumption of food between meals (CAEC)

Consumption of water daily (CH20)

Consumption of alcohol (CALC)

Calories consumption monitoring (SCC)

Physical activity frequency (FAF)

Time using technology devices (TUE)

Transportation used (MTRANS)

Gender

Age

The target variable NObeyesdad is a multi-class variable with 7 classes binned according to the following parameters:

#### BMI = weight (kg) / [height (m)]2

0: Underweight less than 18.5

1: Normal 18.5 to 24.9

2: Overweight I: 25.0 to ~27.5

3: Overweight II: ~27.5 to 29.9

4: Obesity I 30.0 to 34.9

5: Obesity II 35.0 to 39.9

6: Obesity III Higher than 40


## Approach:
The goal is to produce an accurate machine learning model which can classify for obesity based on lifestyle metrics, Age, and Gender. I have not used any data about BMI which was included in the original data, as it would result in data leakage. I investigated if there are any signals which could best predict for obesity, and found three. These signals can be used in marketing segmentation algorithms as weights, giving more weight to predictive features and less weight to less predictive features in a marketing segmentation algorithm. 

I leveraged machine learning techniques such as Decision Trees and Random Forests which produced favorable results. Other algorithms such as Gaussian Bayes, K Nearest Neighbor, Logistic Regression, and stacking did not produce favorable results when compared to the alternatives. In the end, the best model was a XGBoosted Decision Tree. 

Please note this data was in large part synthetically generated, so it is uncertain how well the model may perform in real life applications. The model is further limited because of the exclusion of Height and Weight, which are used in the original research paper to produce more favorable results than this model, but are a form of data leakage to be in a final model. Since weight classes are calculated using BMI, including metrics related to BMI in this model would not showcase the model's true predictive potential on lifestyle characteristics. 

The objective of the final model is to have the highest test accuracy compared to other models, and an overall very decent cross-validation accuracy. When comparing relative models in initial iterations, I compared cross-validated accuracy to determine which models were typically more performative. I iterated using the following approach:

1. Test no hyperparameters for each valid model: KNN, LogReg, RandomForest, DecisionTree, XGBoost, GaussianBayes
2. Compare results
3. Dismiss worst models - LogReg, KNN, GaussianBayes and set baseline as RandomForest
4. Look back at data and see if the data can be further cleaned or if values could be preprocessed differently with different transformers
5. Search with GridSearchCV to find the best ranges for hyperparameters for RandomForest, DecisionTree, XGBoost
6. Refine and tune the hyperparameters for the best models (dismissed DecisionTree in favor of RandomForest)
7. Compare RandomForest vs XGBoost tuned
8. Determine XGBoost tuned with StandardScaling produced the best results

The various iterations are stored in the model_testing_obesity_classification file if you want to look. Please be mindful it may take a long time to run if you choose to uncomment the code blocks in the grid search parameters. 

## Data Sources
Download: https://archive.ics.uci.edu/ml/datasets/Estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition+

Data Description: https://www.sciencedirect.com/science/article/pii/S2352340919306985?via%3Dihub

Research Paper highlighting the applications of this data set for a decision tree model:
https://thescipub.com/pdf/jcssp.2019.67.77.pdf

### Data Description

This data is for the estimation of obesity levels in individuals from the countries of Mexico, Peru and Colombia, based on their eating habits and physical condition. The data contains 17 attributes and 2111 records, the records are labeled with the class variable NObesity (Obesity Level), that allows classification of the data using the values of Insufficient Weight, Normal Weight, Overweight Level I, Overweight Level II, Obesity Type I, Obesity Type II and Obesity Type III. 77% of the data was generated synthetically using the Weka tool and the SMOTE filter, 23% of the data was collected directly from users through a web platform. This data can be used to generate intelligent computational tools to identify the obesity level of an individual and to build recommender systems that monitor obesity levels. For discussion and more information of the dataset creation, please refer to the full-length article “Obesity Level Estimation Software based on Decision Trees” (De-La-Hoz-Correa et al., 2019).

Fabio Mendoza Palechor, Alexis de la Hoz Manotas,
Dataset for estimation of obesity levels based on eating habits and physical condition in individuals from Colombia, Peru and Mexico,
Data in Brief,
Volume 25,
2019,
104344,
ISSN 2352-3409,
https://doi.org/10.1016/j.dib.2019.104344.
(https://www.sciencedirect.com/science/article/pii/S2352340919306985)
Abstract: This paper presents data for the estimation of obesity levels in individuals from the countries of Mexico, Peru and Colombia, based on their eating habits and physical condition. The data contains 17 attributes and 2111 records, the records are labeled with the class variable NObesity (Obesity Level), that allows classification of the data using the values of Insufficient Weight, Normal Weight, Overweight Level I, Overweight Level II, Obesity Type I, Obesity Type II and Obesity Type III. 77% of the data was generated synthetically using the Weka tool and the SMOTE filter, 23% of the data was collected directly from users through a web platform. This data can be used to generate intelligent computational tools to identify the obesity level of an individual and to build recommender systems that monitor obesity levels. For discussion and more information of the dataset creation, please refer to the full-length article “Obesity Level Estimation Software based on Decision Trees” (De-La-Hoz-Correa et al., 2019).
Keywords: Obesity; Data mining; Weka; SMOTE

----------------------------------------------------------
# Conclusion
## This final XGBoost model has:

#### Cross Validation score = 79.50%

#### Test Accuracy = 80.38%

This model is preferred over the base model, so we can accept the alternative hypothesis (XGBoost Model) in favor of the null hypothesis (RandomForest Model).

## The base RandomForest model has:

#### Cross Validation score = 79.56%

#### Test Accuracy = 78.25%

## Implications
This model can be deployed immediately to assign weights to lifestyle metrics Accenture has in its database for it target customers. Higher weights can be given to the tendency to snack, family history with obesity, and Gender. Alternatively the model can be used to predict if someone is obese if they have provided answers to the survey questions originally asked of participants in this dataset. Even if the answers are implied (answers are inferred from other data), this model can provide a decent predictive accuracy for how obese that person is. Furthermore, if the objective is to simply classify if someone is obese, overweight, normal, or underweight, this algorithm would be fine tuned to predict for those broader classes. 
