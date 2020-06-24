import ast
import random
import re
import string
import pickle

import nltk
import pandas as pd
import streamlit as st
from nltk import NaiveBayesClassifier, classify
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

# ! used for preprocessing data
# data = pd.read_csv('./data/onionornot.csv')
# df_min = data[data['label'] == 1]
# df_maj = data[data['label'] == 0]

# df_maj = df_maj.sample(len(df_min), random_state=0)
# df = pd.concat([df_maj, df_min])
# df = df.sample(frac=1, random_state=0)

data_url = 'https://github.com/boblandsky/onion_ml/raw/master/onion_resampled.csv'
df = pd.read_csv(data_url)
df.drop(columns=['Unnamed: 0'], inplace=True)
df_headlines = df['text'].values
df_labels = df['label'].values

hl_train, hl_test, l_train, l_test = train_test_split(df_headlines, df_labels, random_state = 69, test_size = 0.3)

vectorizer = TfidfVectorizer()

vectorizer.fit(hl_train)

X_train = vectorizer.transform(hl_train)
X_test = vectorizer.transform(hl_test)

onion = df[df['label'] == 1].copy()
not_onion = df[df['label'] == 0].copy()

onion.drop(columns = ['label'], inplace=True)
not_onion.drop(columns = ['label'], inplace=True)

onion_list = onion['text'].tolist()
not_onion_list = not_onion['text'].tolist()

st.title('Onion or Not?')

model_picker = st.sidebar.radio('Select different methods for prediction.',
                        ('Logistic Regression, Quick and Dirty',
                        #  'Naive Bayes, Quick and Dirty',
                         'Naive Bayes, NLTK Processed')
                        )
st.sidebar.warning('There was going to be a "Naive Bayes, Quick and Dirty", but it '
                   'crashes the Heroku instance, so it has been removed and may '
                   'whenever I get around to fixing whatever causes it.'
                   )
st.sidebar.markdown("The logistic regression model uses first vectorization using scikit-learn's "
                    '[TFIDF](https://en.wikipedia.org/wiki/Tf%E2%80%93idf) Vectorizer  '
                    'to adjust how important a word is to the corpus. '
                    # 'The second and third models use a '
                    '[A naive Bayes classifer ](https://en.wikipedia.org/wiki/Naive_Bayes_classifier)'
                    'is the other model for classification. Processing '
                    '(e.g. [lemmatization](https://en.wikipedia.org/wiki/Lemmatisation)) '
                    'is done using the Natural Language Toolkit (NLTK), '
                    'a Python library used for natural language processing (NLP).'
                    )
st.sidebar.info('"Quick and Dirty" means that I did not further process the data. '
                'The full dataset from the Github is what is going into the models. '
                'The NLTK processed data is split into test and train sets. '
                'The train/test splits for all methods are 70/30.'
                )

st.warning('Despite me saying "Quick and dirty", '
           'running these models can take some time, please be patient.')

if model_picker == 'Logistic Regression, Quick and Dirty':
    st.header('Model used: Logistic Regression')

    clf_adj = LogisticRegression()
    clf_adj.fit(X_train, l_train)
    score = clf_adj.score(X_test, l_test)
    rounded_score = round(score, 4)*100
    # print('Accuracy: ', score)
    st.write(f'Accuracy on test set: {rounded_score}%')

    test_headline = st.text_input("Give me a headline to predict. A sample one is provided.",
                                "MLS Commissioner Relieved That Nobody Knows Him by Name")

    if st.button('Onion or not?'):
        test_vect = vectorizer.transform([test_headline])
        results = clf_adj.predict(test_vect)
        #st.write(results)
        if results[0] == 0:
            st.write("It's not from the Onion!")
        else:
            st.write("It's from the Onion!")

    # lr_worked = st.radio('Did the logistic regression model make an accurate prediction?',
    #                     ('Yes', 'No')
    #                     )

    # if lr_worked == 'No':
    #     st.write('Bummer.')
    # else:
    #     st.write('Sweet!')

elif model_picker == 'Naive Bayes, NLTK Processed':
    st.header('Model used: Naive Bayes, processed with NLTK')
    # ! only used for initial preprocessing
    # st.warning('This can be slow. Please wait for the model to complete. ')
    # df['headline_tokens'] = df['text'].apply(lambda x: word_tokenize(x))
    stop_words = stopwords.words('english')

    @st.cache(persist=True, show_spinner=True)
    def remove_noise(headline_tokens, stop_words = ()):
        cleaned_tokens = []

        for token, tag in pos_tag(headline_tokens):
            if tag.startswith('NN'):
                pos = 'n'
            elif tag.startswith('VB'):
                pos = 'v'
            else:
                pos = 'a'

            lemmatizer = WordNetLemmatizer()
            token = lemmatizer.lemmatize(token, pos)

            if len(token) > 0 and token not in string.punctuation and token.lower() not in stop_words:
                cleaned_tokens.append(token.lower())
        return cleaned_tokens

    # ! only use for initial preprocessing of data
    # headline_tokens_cleaned = [remove_noise(x, stop_words) for x in headline_tokens]
    # onion_tokens = [word_tokenize(x) for x in onion_list]
    # not_onion_tokens = [word_tokenize(x) for x in not_onion_list]
    # onion_tokens_list = [remove_noise(tokens, stop_words) for tokens in onion_tokens]
    # not_onion_tokens_list = [remove_noise(tokens, stop_words) for tokens in not_onion_tokens]

    # st.write(onion_tokens_list[0])

    # def headlines_for_model(headlines):
    #     for headline_token in headlines:
    #         yield dict([token, True] for token in headline_token)

    # onion_list_for_model = headlines_for_model(onion_tokens_list)
    # not_onion_list_for_model = headlines_for_model(not_onion_tokens_list)

    # onion_dataset = [(onion_dict, 1) for onion_dict in onion_list_for_model]
    # not_onion_dataset = [(onion_dict, 0) for onion_dict in not_onion_list_for_model]

    # dataset = onion_dataset + not_onion_dataset
    # random.shuffle(dataset)

    # train_data = dataset[:12600]
    # test_data = dataset[12600:]

    # # ! offload the data into my github then call it for the classifier
    # pd.DataFrame(train_data).to_csv('train_data.csv', index=False)
    # pd.DataFrame(test_data).to_csv('test_data.csv', index=False)

    train_url = 'https://github.com/boblandsky/onion_ml/raw/master/train_data.csv'
    test_url = 'https://github.com/boblandsky/onion_ml/raw/master/test_data.csv'

    train_data = pd.read_csv(train_url)
    test_data = pd.read_csv(test_url)

    train_data['0'] = train_data['0'].apply(lambda x: ast.literal_eval(x))
    test_data['0'] = test_data['0'].apply(lambda x: ast.literal_eval(x))

    train_data_tuple = list(zip(train_data['0'], train_data['1']))
    test_data_tuple = list(zip(test_data['0'], test_data['1']))
    # st.write(type(train_data['0'][0]))

    clf_nb = NaiveBayesClassifier.train(train_data_tuple)
    nb_acc = round(classify.accuracy(clf_nb, test_data_tuple), 4)*100
    st.write(f'Accuracy on test set: {nb_acc}%')

    test_nb_headline = st.text_input("Give me a headline to predict. A sample one is provided.",
                                      "MLS Commissioner Relieved That Nobody Knows Him by Name")

    if st.button('Onion or not? Round 3'):
        test_nb_tokens = remove_noise(word_tokenize(test_nb_headline))
        results_nb = clf_nb.classify(dict([token, True] for token in test_nb_tokens))
        if results_nb == 1:
            st.write("It's from the Onion!")
        else:
            st.write("It's not from the Onion!")

    # nb_worked = st.radio('Did the Naive Bayes model make an accurate prediction?',
    #                     ('Yes', 'No')
    #                     )

    # if nb_worked == 'No':
    #     st.write('Bummer.')
    # else:
    #     st.write('Sweet!')

# if model_picker == 'Naive Bayes, Quick and Dirty':
#     clf_gnb = GaussianNB()
#     clf_gnb.fit(X_train.toarray(), l_train)
#     score = clf_gnb.score(X_test.toarray(), l_test)
#     rounded_score = round(score, 4)*100
#     # print('Accuracy: ', score)
#     st.write(f'Accuracy on test set: {rounded_score}%')

#     test_gnb_headline = st.text_input("Give me a headline to predict. A sample one is provided.",
#                                       "MLS Commissioner Relieved That Nobody Knows Him by Name")

#     if st.button('Onion or not? Round 2'):
#         test_vect = vectorizer.transform([test_gnb_headline])
#         results = clf_gnb.predict(test_vect.toarray())
#         #st.write(results)
#         if results[0] == 0:
#             st.write("It's not from the Onion!")
#         else:
#             st.write("It's from the Onion!")

st.markdown('---')
st.subheader('Background Info')
st.write('Taking a dataset of headlines that are either from the Onion or '
         "from the r/NotTheOnion subreddit, let's develop a model that can "
         'determine whether or not the headline is an Onion headline or not!'
         )
st.write('The Onion is a satirical newspaper that publishes articles '
         'on current events, oftentimes making everyday events as '
         'newsworthy. r/NotTheOnion is a subreddit where users can post '
         'news articles that sound like they belong on the Onion.'
         )
st.subheader('Sample of dataset used to develop the model (18000 samples)')
st.markdown('The dataset used to develop the model is found '
            '[here. ](https://www.kaggle.com/chrisfilo/onion-or-not)'
            'The dataset for the model is found on the project '
            '[Github.](https://github.com/boblandsky/onion_ml/)'
            )
st.info('1 indicates a headline by the Onion, '
        'while 0 is a real headline found on r/NotTheOnion. '
        'The data was originally 24000 samples, and rebalanced to 18000 (9000 samples of each). '
        'The first five rows are shown below.'
        )
st.table(df.head())