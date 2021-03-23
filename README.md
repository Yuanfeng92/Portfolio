# Hello!  Welcome to my <u>Portfolio</u>! :wave:

---

Hi there! I'm Yuanfeng and welcome to my portfolio! This page serves as a summary of the projects and things that I have done on my Data Science journey! 

If you will like to get in touch with me, you can contact me via [email](mailto:yuanfeng92@hotmail.com ) or check out my [LinkedIn](https://www.linkedin.com/in/yuanfeng-luan/)! :smiley:

#### Contents: 

- [Data Science Projects](#Data-Science-Projects)
- [Other Projects](#Other-Projects )
- [Skills & Education](#Skills-and-Education)
- [Pet Project](#Pet-Project)

---

## Data Science Projects

All projects below are primarily coded in Python. :snake:

### 1. [Analysis of SAT & ACT](https://github.com/Yuanfeng92/Analysis-of-SAT-ACT)

**Problem Statement**

For this project, i analysed the SAT and ACT scores and participation rates from 2017 and 2018. As the participation rates varied greatly by state, I aim to identify the key factors influencing SAT participation rate and provide recommendations to College Board to improve participation rates of SAT.

**Findings and Recommendations**

Most states will favour only 1 test, which results in low participation rate in another. This is especially true for states that made either of the test mandatory. As such, College Board should avoid states with mandatory ACT exams as effort will likely prove ineffective.

In general, there are a greater number of states with high participation rate for ACT (as compared to SAT). Another trend that was found is that as participation rate increases, the scores achieved by the state drops, signalling mandatory exams will result in a lower state (mean) score compared to voluntary testing.

College Board should target states that does not have high ACT participation rate (i.e. ACT not mandatory) and low SAT participation rate in both 2017 and 2018, this states have the potential to have a greater participation for SAT. For example, based on the data, College Board should focus more effort on Iowa. College Board could incentivize the students at Iowa to take SAT by introducing SAT School Day and the use of digital marketing of SAT, highlighting the benefits of SAT over ACT.

**Data Science Techniques Used**

- data cleaning
- exploratory data analysis
- analysis of information and offer recommendations
- presentation of findings

**Key Libraries**

```python
Numpy, pandas, Matplotlib, Seaborn
```



### 2. [Ames Housing Price Prediction](https://github.com/Yuanfeng92/Price-Prediction-and-Study-of-Ames-Housing-Data)

**Problem Statement**

In this project, I examined housing data from city of Ames, USA. The data contains information on properties sold between the 2006 to 2010 and has over 80 features. The goal of this project is to:

1. Identify factors that potential home sellers can do to improve the value of their properties with the highest return on investment (ROI).
2. Create a model that predicts the saleprice of a given property.

**Findings and Recommendations**

The analysis revealed that the most important features determining the saleprice of a property are its area, condition, age (& if remodelled) and location. 

For potential home sellers, they should focus on improvements to the condition and quality of their homes as these will likely bring the highest ROI. For example, they could repair any defects in the house, repaint the house, repair/upgrade exterior material, etc. Apart from making improvements, home sellers should aim to sell their property as soon as they made up their mind as age is a huge factor in the deterioration of value.

Aside from the analysis, using a selected number of features, a model was created to predict the saleprice of the property. The prediction should be used as a reference to ensure that potential home sellers or home buyers do not get tricked into selling or buying properties at unreasonable prices. However, do note that the mode has its limitation, and the predicted saleprice should only be used as a basis for negotiation. The RMSE of the model is at about 11%. 

**Data Science Techniques Used**

- data cleaning
- exploratory data analysis
- data wrangling (dealing with missing data, outliers, etc) 
- feature engineering
- feature selection
- model building, tuning and evaluation
- analysis of information and offer recommendations
- presentation of findings

**Key Libraries** 

```python
scikit-learn: impute, preprocessing, model_selection (GridSearchCV, RandomizedSearchCV), linear_model (LinearRegression, Ridge, Lasso, ElasticNet)
```



### 3. [Reddit Classification - r/Dadjokes vs r/Antijokes](https://github.com/Yuanfeng92/Classification-of-Dadjokes-and-Antijokes)
<img src="Visualizations\Dadjokes VS Antijokes Top Bigram.png" alt="Dadjokes VS Antijokes Top Bigram" style="zoom: 50%;" />

**Problem Statement**

In this project, I tried to classify posts from two different subreddits: r/Dadjokes and r/Antijokes. This is challenging as dadjokes and antijokes are usually told in similar format and with similar words & phrases. The main difference between them is the contextual understanding of the sentence. Nonetheless, the goal is to create a model that classify jokes into dadjokes or antijokes and identify the words and topics that are most deterministic of each. 

**Findings**

I developed numerous word-frequency based models to classify which subreddit did a joke belong to. The final model chosen was Logistic Regression Classifier as it had the highest precision and F1 score while maintaining relatively low overfitting. The model had a final accuracy score of 67.1%. Aside from that, the model is better at classifying antijokes as compared to dadjokes. This suggest that there are more identifiable words for antijokes.

Interestingly, there is no evident topics that strongly identifies dadjokes. The top most identifying words for dadjokes are common words such as "my", "got", "it" and "but". On the other hand, "walked in a bar" was a strong predictor for antijokes. Apart from that, top identifying words for antijokes are again common words such as "and", "what", "man", "you". 

From the above, it is evident that the dadjokes and antijokes use very similar (common) words, so it is difficult to distinguish them purely based on the words. This agrees with the understanding that the difference between dadjokes and antijokes lies largely in the contextual understanding of the jokes.

**Data Science Techniques Used**

- data scrapping
- data wrangling (dealing with missing data, outliers) 
- exploratory data analysis
- feature engineering (length of title, length of post)
- feature selection
- text processing and tokenisation (using bag of words and term frequency–inverse document frequency)
- model building, tuning and evaluation
- analysis of information and offer recommendations
- presentation of findings

**Key Libraries**

```
requests, time, glob, regex, wordcloud, nltk, scikit-learn: metrics (confusion_matrix, f1_score, recall_score, precision_score), feature_extraction.text (CountVectorizer, TfidfVectorizer), LogisticRegression, KNeighborsClassifier, MultinomialNB, RandomForestClassifier, 
```



### 4. [West Nile Virus Analysis and Prediction](https://github.com/Yuanfeng92/West-Nile-Virus-Analysis)
<img src="Visualizations\West Nile Virus Spray Cluster.png" alt="West Nile Virus Spray Cluster" style="zoom:50%;" />

**Problem Statement**

West Nile virus (WNV) is the virus that cause West Nile fever. It is mainly spread by infected mosquitoes and about 20% of infected victims develop symptoms ranging from a persistent fever, to serious neurological illnesses that can result in death. Due to the outbreak of WNV in Chicago, the City of Chicago and Chicago Department of Public Health (CDPH) established a comprehensive surveillance and control program to check different locations for the presence of WNV infected mosquitoes. The goal of this project is to predict the locations where WNV may be present and recommend strategies to spray pesticides for WNV control. 

**Findings and Recommendations**

From the best performing model (using XGBoost), the most important features to identify WNV presence is location features, followed by weather features (wind speed, sea level and temperature) and time feature (year and week of the year). This suggests that there are locations that are more at risk than others and also points to the seasonality of WNV outbreaks (between July to October) , hence, heavy spraying control should be done at these locations and times.

A cost benefit analysis was conducted. The preliminary socio-economic cost estimates at USD 4,909,695 arising from people falling ill from WNV. On the other hand, we estimate about USD 413,424 spraying cost to spray all predicted locations between July to October, about 12% of the estimated socio-economic cost. While there is a high redundancy in spraying during the entire outbreak period, our team believes that it is of utmost importance to ensure public safety against WNV, moreover, it is only at a fraction of the estimated socio-economic cost. That said, we offer alternative strategies should the proposed policy be unfavourable. 

**Data Science Techniques Used**

- data cleaning, feature engineering, selection of weather data
- model building, tuning and evaluation of RandomForest, XGBoost and Neural Network classifier
- analysis of information and offer recommendations
- presentation of findings

**Key Libraries** 

```python
scikit-learn, imblearn (SMOTE, SMOTETomek, make_pipeline), xgboost, tensorflow (KerasClassifier, optimizers, Sequential, Dense, Dropout, EarlyStopping)
```

*This is done as a group project as 4, data science techniques and libraries used are for the areas that I have covered.



### 5. [Analysis of Genshin Impact Reviews (Sentiment Analysis and Topic Extraction)](https://github.com/Yuanfeng92/Analysis-of-Genshin-Impact-Reviews)
<img src="Visualizations\Genshin Impact Reviews Trend Analysis.png" alt="Genshin Impact Reviews Trend Analysis" style="zoom:50%;" />

**Problem Statement**

Popular mobile apps can often receive a huge amount of reviews from users on platforms such as Google Play Store and Apple App Store. For example, the app that I am working on, Genshin Impact, received over 100,000 English reviews within 3 months of launch. This is a huge amount of reviews/feedbacks for the customer service personnel and app developers to look through manually. 

The goal of this project is to use machine learning and deep learning techniques to extract important topics within the reviews for the relevant team to handle (i.e. highlight critical technical issues to technical team, design issues to development team). This will assist the developers to focus on the most critical issues faced by majority of its users without spending excessive amount of time looking through enormous amounts of feedback. The secondary purpose fo this project is to use the reviews and their scores to create a sentiment analysis model. The sentiment analysis model can then be deployed to identify sentiments of non-rated feedbacks (such as feedbacks in surveys, comments in forum, YouTube, etc).

**Findings and Recommendations**

I trained multiple LDA models on the dataset, and picked the model that had the best coherence score and best interpretability. The final LDA models revealed significant trends in the user reviews across multiple updates. For example, there is a huge spike of negative reviews after a major update, signifying that the developers need to better manage the update file size and process. The insights from the trend is a high level insight for development teams and management to understand issues faced by users, while the details from these topics allows the teams to know exactly what are the issues to focus on. 

For this project, I have also trained multiple models to create a sentiment analysis model. The reviews have scores from 1 to 5, however, due to severe class imbalance, subjectivity of review scores and multiple testing, I decided to proceed with a two-class classification model where the reviews can either be a positive or negative review. The final model used is an optimized LSTM model with an accuracy of 90.5% and F1-score of 73.0%.

**Data Science Techniques Used**

- scrape reviews from Google Play Store and Apply App Store
- data wrangling (dealing with missing data, outliers, removing non-English reviews, text pre-processing)
- feature engineering and feature selection
- exploratory data analysis
- topic modelling and evaluation
- sentiment analysis using machine learning and deep learning models
- analysis of information and offer recommendations
- presentation of findings

**Key Libraries** 

```Python
app_store_scraper, google_play_scraper, nltk: tokenize, corpus, stem, SpaCy, Gensim: models (LdaModel, CoherenceModel), wordcloud, scikit-learn, tensorflow/keras (Sequential, LSTM, Tokenizer, pad_sequence, Embedding)
```



## Other Projects 

(click on the logos to visit my profiles!)

### 1. Tableau Public

[<img align="left" src="Visualizations\Tableau Logo 300.png" alt="Tableau" />](https://public.tableau.com/profile/yuanfeng5611#!/)







This is my Tableau Public profile. I try to practise making new Tableau dashboard once a week. I love the social data project [#MakeoverMonday](https://www.makeovermonday.co.uk/). I was able to learn many Tableau techniques from them and aim to attempt their weekly challenges!



### 2. HackerRank 

[<img align="left" src="Visualizations\HackerRank Logo 300.png" alt="HackerRank" />](https://www.hackerrank.com/yuanfeng92)







For now, I am mainly using HackerRank as a platform for daily SQL practise.



---

## Pet Project

### PoGoRaid Alert :alarm_clock:

I used to play a decent amount of PokemonGo with my family and friends. In PokemonGo, there is a mechanic called *Raids* where players can play together to take down and capture very powerful pokemons. However, raids happen at random timing and it was troublesome to check for raid timing regularly, resulting in many raids missed even though I was available to join them! To put an end to missing raids, I decided to write a simple Python script that sent an alert whenever there are raids at designated locations! 

To do this, the script scraped an online source that has raid information, checked for whether there is raid in a number of pre-determined locations and if there is a raid available, it sent an alert 30 mins and 5 mins before the raid. No more missed raids! :laughing:



## Skills and Education 

### Skills 

1. Python
2. Tableau
3. SQL
4. Microsoft Office

**Skills I am working on:**

1. AWS Cloud Practitioner
2. Docker



### Education :school:

**Data Science Immersive (DSI)** | General Assembly, Singapore | 2020 - 2021

**Renaissance Engineering Programme (REP)** | NTU, Singapore | 2014 - 2018



**MOOCs** :computer:

**SQL Masterclass: SQL for Data Analytics**  | Udemy | 2021

**Tableau 2020 A-Z: Hands-On Tableau Training for Data Science**  | Udemy | 2021

**Python for Data Science and Machine Learning Bootcamp**   | Udemy | 2020

**Complete Python Developer: Zero to Mastery**  | Udemy | 2019 - 2020

