import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# ! used for preprocessing data
# data = pd.read_csv('./data/onionornot.csv')
# df_min = data[data['label'] == 1]
# df_maj = data[data['label'] == 0]

# df_maj = df_maj.sample(len(df_min), random_state=0)
# df = pd.concat([df_maj, df_min])
# df = df.sample(frac=1, random_state=0)

data_url = 
df = pd.read_csv()

st.title('Onion or Not?')
st.write('Taking a dataset of headlines that are either from the Onion or '
         "from the r/NotTheOnion subreddit, let's develop a model that can "
         'determine whether or not the headline is an Onion headline or not!'
         )
st.info('1 indicates a headline by the Onion, '
        'while 0 is a headline that could be found on r/NotTheOnion.')
st.header('Model used: Logistic Regression')
st.subheader('Sample of dataset used to develop the model (18000 samples)')
st.markdown('The dataset used to develop the model is found '
            '[here.](https://www.kaggle.com/chrisfilo/onion-or-not)'
            )
st.info('The data was originally 24000 samples, and rebalanced to 18000 (9000 samples of each).')
st.write(df.head())

df_headlines = df['text'].values
df_labels = df['label'].values

hl_train, hl_test, l_train, l_test = train_test_split(df_headlines, df_labels, random_state = 69)

vectorizer = TfidfVectorizer()

vectorizer.fit(hl_train)

X_train = vectorizer.transform(hl_train)
X_test = vectorizer.transform(hl_test)

clf_adj = LogisticRegression()
clf_adj.fit(X_train, l_train)
score = clf_adj.score(X_test, l_test)
rounded_score = round(score, 4)*100
# print('Accuracy: ', score)
st.write(f'Accuracy on test set: {rounded_score}%')

test_headline = st.text_input("Give me a headline to predict. A sample one is provided.",
                              "CIA Realizes It's Been Using Black Highlighters All These Years")

if st.button('Onion or not?'):
    test_vect = vectorizer.transform([test_headline])
    results = clf_adj.predict(test_vect)
    #st.write(results)
    if results[0] == 0:
        st.write("It's not from the Onion!")
    else:
        st.write("It's from the Onion!")