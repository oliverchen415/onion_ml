# Machine learning with Onion headlines

## NEW LINK:
[Streamlit Share](https://share.streamlit.io/oliverchen415/onion_ml/onion_refactor.py)

The above link goes to Streamlit's new hosting service for Streamlit apps. Streamlit Share should load faster than Heroku's, since Streamlit Share doesn't seem to go to sleep. Please leave an issue if anything goes wrong. The old Heroku link is preserved below.

## HEROKU LINK (OLD):
[Is it Onion?](https://isitonion.herokuapp.com)

---

## Goal:
### Determining whether a headline is from the Onion or not

The best place to get absurd headlines is the Onion. Sometimes real life beats out the Onion in absurdity.
Here, I use two different estimators from Python's scikit-learn (logistic regression and naive Bayes) as well as the Natural Language Toolkit (NLTK) to process the headlines. The app uses these estimators to determine whether or not a given headline is from the Onion or r/NotTheOnion.

---
## Instructions:

![img](https://github.com/boblandsky/onion_ml/raw/master/Annotation%202020-06-24%20161827.png)

The app allows you to select three different models:
  * Logistic Regression, no NLTK processing
  * ~~Naive Bayes, no NLTK processing~~ Currently not working, crashes the Heroku app
  * Naive Bayes, NLTK processed


Clicking on the Onion or Not button will run your selected model and return a prediction of whether or not the headline is from the Onion or r/NotTheOnion.
