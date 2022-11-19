import re
import numpy as np
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
from keras.models import load_model

MODEL_WEIGHTS_H5 = 'notebooks/final_question_pairs_model_deeper=False_wider=False_lr=0.001_dropout=0.3.h5'

# Raise exception if MODEL_WEIGHTS_H5 is not found
import os
if not os.path.exists(MODEL_WEIGHTS_H5):
    raise Exception('Model weights not found. Please place the model weights in the same folder as this notebook.')



print("1")
# A list of contractions from http://stackoverflow.com/questions/19790188/expanding-english-language-contractions-in-python
contractions = {
    "ain't": "am not",
    "aren't": "are not",
    "can't": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'd've": "he would have",
    "he'll": "he will",
    "he's": "he is",
    "how'd": "how did",
    "how'll": "how will",
    "how's": "how is",
    "i'd": "i would",
    "i'll": "i will",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it would",
    "it'll": "it will",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "must've": "must have",
    "mustn't": "must not",
    "needn't": "need not",
    "oughtn't": "ought not",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "she'd": "she would",
    "she'll": "she will",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "that'd": "that would",
    "that's": "that is",
    "there'd": "there had",
    "there's": "there is",
    "they'd": "they would",
    "they'll": "they will",
    "they're": "they are",
    "they've": "they have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'll": "we will",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "where'd": "where did",
    "where's": "where is",
    "who'll": "who will",
    "who's": "who is",
    "won't": "will not",
    "wouldn't": "would not",
    "you'd": "you would",
    "you'll": "you will",
    "you're": "you are"
}

# Find the number of times each word was used and the size of the vocabulary
word_counts = {}

# import clean_headlines from ../data/interim/clean_headlines.pkl
import pickle
with open('data/interim/clean_headlines.pkl', 'rb') as f:
    clean_headlines = pickle.load(f)

for date in clean_headlines:
    for headline in date:
        for word in headline.split():
            if word not in word_counts:
                word_counts[word] = 1
            else:
                word_counts[word] += 1

print("Size of Vocabulary:", len(word_counts))
print("2. Loading Glove")
# Load GloVe's embeddings
embeddings_index = {}
with open('models/glove/glove.840B.300d.txt', encoding='utf-8') as f:
    for line in f:
        values = line.split(' ')
        word = values[0]
        embedding = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = embedding

print('Word embeddings:', len(embeddings_index))

# Find the number of words that are missing from GloVe, and are used more than our threshold.
missing_words = 0
threshold = 10

for word, count in word_counts.items():
    if count > threshold:
        if word not in embeddings_index:
            missing_words += 1

missing_ratio = round(missing_words/len(word_counts),4)*100

print("Number of words missing from GloVe:", missing_words)
print("Percent of words that are missing from vocabulary: {}%".format(missing_ratio))

# Limit the vocab that we will use to words that appear ≥ threshold or are in GloVe

#dictionary to convert words to integers

print("3. Words to int")
vocab_to_int = {}

value = 0
for word, count in word_counts.items():
    if count >= threshold or word in embeddings_index:
        vocab_to_int[word] = value
        value += 1

# Special tokens that will be added to our vocab
codes = ["<UNK>","<PAD>"]

# Add codes to vocab
for code in codes:
    vocab_to_int[code] = len(vocab_to_int)

# Dictionary to convert integers to words
int_to_vocab = {}
for word, value in vocab_to_int.items():
    int_to_vocab[value] = word

usage_ratio = round(len(vocab_to_int) / len(word_counts),4)*100

print("Total Number of Unique Words:", len(word_counts))
print("Number of Words we will use:", len(vocab_to_int))
print("Percent of Words we will use: {}%".format(usage_ratio))

def news_to_int(news):
    '''Convert your created news into integers'''
    ints = []
    for word in news.split():
        if word in vocab_to_int:
            ints.append(vocab_to_int[word])
        else:
            ints.append(vocab_to_int['<UNK>'])
    return ints

def clean_text(text, remove_stopwords = True):
    '''Remove unwanted characters and format the text to create fewer nulls word embeddings'''

    # Convert words to lower case
    text = text.lower()

    # Replace contractions with their longer forms
    if True:
        text = text.split()
        new_text = []
        for word in text:
            if word in contractions:
                new_text.append(contractions[word])
            else:
                new_text.append(word)
        text = " ".join(new_text)

    # Format words and remove unwanted characters
    text = re.sub(r'&amp;', '', text)
    text = re.sub(r'0,0', '00', text)
    text = re.sub(r'[_"\-;%()|.,+&=*%.,!?:#@\[\]]', ' ', text)
    text = re.sub(r'\'', ' ', text)
    text = re.sub(r'\$', ' $ ', text)
    text = re.sub(r'u s ', ' united states ', text)
    text = re.sub(r'u n ', ' united nations ', text)
    text = re.sub(r'u k ', ' united kingdom ', text)
    text = re.sub(r'j k ', ' jk ', text)
    text = re.sub(r' s ', ' ', text)
    text = re.sub(r' yr ', ' year ', text)
    text = re.sub(r' l g b t ', ' lgbt ', text)
    text = re.sub(r'0km ', '0 km ', text)

    # Optionally, remove stop words
    if remove_stopwords:
        text = text.split()
        stops = set(stopwords.words("english"))
        text = [w for w in text if not w in stops]
        text = " ".join(text)

    return text

# Limit the length of a day's news to 200 words, and the length of any headline to 16 words.
# These values are chosen to not have an excessively long training time and 
# balance the number of headlines used and the number of words from each headline.
max_headline_length = 16
max_daily_length = 200

def padding_news(news):
    '''Adjusts the length of your created news to fit the model's input values.'''
    padded_news = news
    if len(padded_news) < max_daily_length:
        for i in range(max_daily_length-len(padded_news)):
            padded_news.append(vocab_to_int["<PAD>"])
    elif len(padded_news) > max_daily_length:
        padded_news = padded_news[:max_daily_length]
    return padded_news

# Normalize opening prices (target values)
max_price = 768.0400389999995
min_price = -926.5498050000006
mean_price = -3.26566921026157
def normalize(price):
    return ((price-min_price)/(max_price-min_price))

def unnormalize(price):
    '''Revert values to their unnormalized amounts'''
    price = price*(max_price-min_price)+min_price
    return(price)

if __name__=='__main__':
    # Load the model
    print("Loading model...")
    model = load_model(MODEL_WEIGHTS_H5)

    # Новостей должно быть довольно много, чтобы модель могла сделать хорошее предсказание
    # Примерно столько сколько здесь
    # Только английские

    create_news = "U.S.-led fight on ISIS have killed 352 civilians: Pentagon \
        Woman offers undercover officer sex for $25 and some Chicken McNuggets \
        Ohio bridge refuses to fall down after three implosion attempts \
        Jersey Shore MIT grad dies in prank falling from library dome \
        New York graffiti artists claim McDonald's stole work for latest burger campaign \
        SpaceX to launch secretive satellite for U.S. intelligence agency \
        Severe Storms Leave a Trail of Death and Destruction Through the U.S. \
        Hamas thanks N. Korea for its support against ‘Israeli occupation’ \
        Baker Police officer arrested for allegedly covering up details in shots fired investigation \
        Miami doctor’s call to broker during baby’s delivery leads to $33.8 million judgment \
        Minnesota man gets 15 years for shooting 5 Black Lives Matter protesters \
        South Australian woman facing possible 25 years in Colombian prison for drug trafficking \
        The Latest: Deal reached on funding government through Sept. \
        Russia flaunts Arctic expansion with new military bases\
    "

    clean_news = clean_text(create_news)

    int_news = news_to_int(clean_news)

    pad_news = padding_news(int_news)

    pad_news = np.array(pad_news).reshape((1,-1))

    pred = model.predict([pad_news,pad_news])

    price_change = unnormalize(pred)


    print( )

