'''
	stemming will remove all the words like 'ing','ed' all the forms
	but it will create a word that does not have a true meaning
	lemmatizing will create a word that will have a true meaning which you can search in dictionary
'''

'''
	tense matter in the sense 
	'i like this product'-----> positive
	'i used to like this product'----> negative
	we may use lemmatizer in cases where we just want the meaning and not what is actually said in the statement
'''

'''
	it tokenizes the string for example 
	string = 'i pulled the chair' will become
	['i','pulled','the','chair']

'''


from textblob import TextBlob
from textblob import Word
from collections import Counter
import pickle
import numpy as np
import sys
import random


def tokenize_Count(inital_List,given_File,lower_frequency = 30,higher_frequency = 1000):
	
	lexicon = []

	with open(given_File,'r') as f:
		contents = f.readlines()
		
		for l in contents[:]:
			zen = TextBlob(l)
			all_words = zen.words
			lexicon += list(all_words)
			# apply lemmatizer on each word because it works only on word
		lexicon = [Word(i).lemmatize() for i in lexicon]

		w_count = Counter(lexicon) # it returns a dictionary containing the element with count one
		l2 = []

		for w in w_count:
			if higher_frequency > w_count[w] > lower_frequency:
				l2.append(w)

		l3 = inital_List + l2
		return l3
	pass


def sampling_handling(sample, lexicon, classification):
	featureSet=[]

	with open(sample,'r') as f:
		contents = f.readlines()
		for l in contents[:]:
			current_sen = TextBlob(l.lower())
			current_words = current_sen.words
			current_words = [Word(i).lemmatize() for i in current_words]

			features = np.zeros(len(lexicon))
			for word in current_words:
				if word.lower() in lexicon:
					index_value = lexicon.index(word.lower())
					features[index_value] += 1

			features = list(features)
			featureSet.append([features,classification])

	return featureSet


def create_feature_sets_and_labels(pos, neg, train_size = 0.9):
	lexicon = tokenize_Count([], pos, 50, 1000)
	lexicon = tokenize_Count(lexicon, neg, 50, 1000)

	features = []
	features += sampling_handling(pos,lexicon,[1,0])
	features += sampling_handling(neg,lexicon,[0,1])

	random.shuffle(features)
	features = np.array(features)

	train_size_value = int(features.shape[0]*train_size)


	train_x = list(features[:,0][:train_size_value])
	train_y = list(features[:,1][:train_size_value])
	test_x = list(features[:,0][train_size_value:])
	test_y = list(features[:,1][train_size_value:])

	# divide the array into test and train
	return train_x,train_y,test_x,test_y

	pass


if __name__ == '__main__':
	train_x,train_y,test_x,test_y = create_feature_sets_and_labels('pos.txt', 'neg.txt')
	# if you want to pickle this data:
	with open('sentiment_train_test/sentiment_set.pickle','wb') as f:
		pickle.dump([train_x,train_y,test_x,test_y],f)
		print('data pickled ')
