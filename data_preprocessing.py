# Importing necessary libraries
import nltk
from nltk.stem import PorterStemmer
import json
import pickle
import numpy as np

# Download NLTK data
nltk.download('punkt')

# List of words to be ignored while creating the dataset
ignore_words = ['?', '!', ',', '.', "'s", "'m"]

# Open the JSON file and load data from it
train_data_file = open('intents.json')
data = json.load(train_data_file)
train_data_file.close()

# Function to stem words
def get_stem_words(words, ignore_words):
    stemmer = PorterStemmer()
    stem_words = [stemmer.stem(word.lower()) for word in words if word.lower() not in ignore_words]
    return sorted(list(set(stem_words)))

# Function to create bot corpus
def create_bot_corpus(words, classes, pattern_word_tags_list, ignore_words):
    for intent in data['intents']:
        for pattern in intent['patterns']:
            pattern_words = nltk.word_tokenize(pattern)
            words.extend(pattern_words)
            pattern_word_tags_list.append((pattern_words, intent['tag']))

        if intent['tag'] not in classes:
            classes.append(intent['tag'])

    stem_words = get_stem_words(words, ignore_words)
    stem_words = sorted(stem_words)
    classes = sorted(classes)

    print('stem_words list:', stem_words)
    print('classes list:', classes)

    return stem_words, classes, pattern_word_tags_list

# Function to create Bag of Words encoding
def bag_of_words_encoding(stem_words, pattern_word_tags_list):
    bag = []
    for word_tags in pattern_word_tags_list:
        pattern_words = word_tags[0]
        stemmed_pattern_words = get_stem_words(pattern_words, ignore_words)
        bag_of_words = [1 if stem_word in stemmed_pattern_words else 0 for stem_word in stem_words]
        bag.append(bag_of_words)

    return np.array(bag)

# Function to create class label encoding
def class_label_encoding(classes, pattern_word_tags_list):
    labels = []
    for word_tags in pattern_word_tags_list:
        tag = word_tags[1]
        label_encoding = [1 if tag == label else 0 for label in classes]
        labels.append(label_encoding)

    return np.array(labels)

# Function to preprocess the training data
def preprocess_train_data():
    stem_words, tag_classes, word_tags_list = create_bot_corpus(words, classes, pattern_word_tags_list, ignore_words)
    
    # Save stem_words and tag_classes to a Python pickle file
    with open('stem_words.pkl', 'wb') as file:
        pickle.dump(stem_words, file)

    with open('tag_classes.pkl', 'wb') as file:
        pickle.dump(tag_classes, file)

    train_x = bag_of_words_encoding(stem_words, word_tags_list)
    train_y = class_label_encoding(tag_classes, word_tags_list)
    
    return train_x, train_y

# Uncomment the following lines after completing the code
 #print("First BOW encoding:", bow_data[0])
# print("First Label encoding:", label_data[0])
