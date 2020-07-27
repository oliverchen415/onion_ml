import ast
import string
import pandas as pd
import streamlit as st

from nltk import classify
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

data_url = 'https://github.com/oliverchen415/onion_ml/raw/master/onion_resampled.csv'
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
                         'Naive Bayes, NLTK Processed')
                        )
st.sidebar.warning('**Update:** Still have no solutions for the crashing with the naive Bayes model, '
                   'perhaps I will look for another model in place of the naive Bayes. '
                   )
st.sidebar.markdown("The logistic regression model uses first vectorization using scikit-learn's "
                    '[TFIDF](https://en.wikipedia.org/wiki/Tf%E2%80%93idf) Vectorizer  '
                    'to adjust how important a word is to the corpus. '
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

st.warning('Running these models can take some time, please be patient.')

if model_picker == 'Logistic Regression, Quick and Dirty':
    st.header('Model used: Logistic Regression')
    linreg_pickle_url = 'https://github.com/oliverchen415/onion_ml/raw/master/linreg.pkl'
    clf_adj = pd.read_pickle(linreg_pickle_url)
    score = clf_adj.score(X_test, l_test)
    rounded_score = round(score, 4)*100
    st.write(f'Accuracy on test set: {rounded_score}%')

    test_headline = st.text_input("Give me a headline to predict. A sample one is provided.",
                                "MLS Commissioner Relieved That Nobody Knows Him by Name")

    if st.button('Onion or not?'):
        test_vect = vectorizer.transform([test_headline])
        results = clf_adj.predict(test_vect)
        if results[0] == 0:
            st.write("It's not from the Onion!")
        else:
            st.write("It's from the Onion!")

elif model_picker == 'Naive Bayes, NLTK Processed':
    st.header('Model used: Naive Bayes, processed with NLTK')

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

    train_url = 'https://github.com/oliverchen415/onion_ml/raw/master/train_data.csv'
    test_url = 'https://github.com/oliverchen415/onion_ml/raw/master/test_data.csv'

    train_data = pd.read_csv(train_url)
    test_data = pd.read_csv(test_url)

    train_data['0'] = train_data['0'].apply(lambda x: ast.literal_eval(x))
    test_data['0'] = test_data['0'].apply(lambda x: ast.literal_eval(x))

    train_data_tuple = list(zip(train_data['0'], train_data['1']))
    test_data_tuple = list(zip(test_data['0'], test_data['1']))

    nb_pickle_url = 'https://github.com/oliverchen415/onion_ml/raw/master/nb_nltk.pkl'
    clf_nb = pd.read_pickle(nb_pickle_url)
    nb_acc = round(classify.accuracy(clf_nb, test_data_tuple), 4)*100
    st.write(f'Accuracy on test set: {nb_acc}%')

    test_nb_headline = st.text_input("Give me a headline to predict. A sample one is provided.",
                                      "MLS Commissioner Relieved That Nobody Knows Him by Name")

    if st.button('Onion or not? NLTK Edition'):
        test_nb_tokens = remove_noise(word_tokenize(test_nb_headline))
        results_nb = clf_nb.classify(dict([token, True] for token in test_nb_tokens))
        if results_nb == 1:
            st.write("It's from the Onion!")
        else:
            st.write("It's not from the Onion!")

st.markdown('---')
if st.checkbox('Show background info'):
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
                '[Github.](https://github.com/oliverchen415/onion_ml/)'
                )
    st.info('1 indicates a headline by the Onion, '
            'while 0 is a real headline found on r/NotTheOnion. '
            'The data was originally 24000 samples, and rebalanced to 18000 (9000 samples of each). '
            'The first five rows are shown below.'
            )
    st.table(df.head())