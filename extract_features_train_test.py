import numpy as np
import unicodecsv
from nltk.corpus import stopwords
import re

def extract_features_train_test(qfile_test='question_test.csv', qfile_train='question_train.csv', catfile='category.csv', 
								subcats=True, outfile='features_train_test.npz'):
	"""
	extract the numerical features from text documents
		qfile_test - .csv file containing the test data
		qfile_train - .csv file containing the train data
		catfile - .csv file containing the categories
		subcats - Boolean parameter which decides whether to perform the analysis for categories or subcategories
		outfile - .npz file to store the output arrays, dictionary and list
	"""
	# read from the .csv files
	question_train, question_test, category = read_files(qfile_train, qfile_test, catfile)
	# create a dictionary with the category names
	categories, par_sub_relation = create_categories_dict(category, subcats)
	questions_train = []
	questions_test = []
	featurenames = set()
	valid_questions_train = []
	valid_questions_test = []
	# from all the questions remove the ones that don't have a category or whose length after the preprocessing is 0
	valid_questions_train, featurenames = preprocess(question_train[1:], featurenames, 'train')
	valid_questions_test, featurenames = preprocess(question_test[1:], featurenames, 'test')
	featurenames = list(featurenames)
	categoryids_train = np.zeros((1, len(valid_questions_train)))
	# iterate over the valid train questions and create the categoryids array with the question id-s and the list of questions
	for i in valid_questions_train:
		if subcats:
			category_id = i[3]
		else:
			category_id = par_sub_relation[int(i[3])]
		categoryids_train[:, valid_questions_train.index(i)] = category_id
		category_text = i[4]
		questions_train.append(category_text)
	# iterate over the valid test questions and create the list of questions
	for i in valid_questions_test:
		category_text = i[4]
		questions_test.append(category_text)
	# make the features matrix where each column is a data point and each row is a vector with a position for each word from the
	# featurenames list -> the column contains 1-s if the word appears in that question and 0-s if the word doesn't appear
	features_train = np.zeros((len(featurenames), len(valid_questions_train)))
	features_test = np.zeros((len(featurenames), len(valid_questions_test)))
	for feature_id in range(len(featurenames)):
		for q_id in range(len(questions_train)):
			if featurenames[feature_id] in questions_train[q_id]:
				features_train[feature_id, q_id] = 1.
		for q_id in range(len(questions_test)):
			if featurenames[feature_id] in questions_test[q_id]:
				features_test[feature_id, q_id] = 1.
	# save the arrays to the output file
	np.savez(outfile, features_train=features_train, features_test=features_test, featurenames=featurenames, 
			categoryids_train=categoryids_train, categories=categories)
	
def read_files(qfile_train, qfile_test, catfile):
	"""
	read from .csv files
		qfile_train - .csv file containing the SMS Guru questions for the train set
		qfile_test - .csv file containing the SMS Guru questions for the test set
		catfile - .csv file containing the categories
	"""
	with open(qfile_train, 'rb') as csvfile:
		question_train = list(unicodecsv.reader(csvfile, delimiter=",", quoting=unicodecsv.QUOTE_ALL, escapechar="\\", encoding='utf-8'))
	with open(qfile_test, 'rb') as csvfile:
		question_test = list(unicodecsv.reader(csvfile, delimiter=",", quoting=unicodecsv.QUOTE_ALL, escapechar="\\", encoding='utf-8'))
	with open(catfile, 'rb') as csvfile:
		category = list(unicodecsv.reader(csvfile, delimiter=",", quoting=unicodecsv.QUOTE_ALL, escapechar="\\", encoding='utf-8'))
	return question_train, question_test, category

def create_categories_dict(category, subcats):
	"""
	create a dictionary with the category id as the key and the category description as the valid_questions
		category - list of entries from the category.csv file
		subcats - Boolean parameter which decides whether to perform the analysis for categories or subcategories

		return:
			categories - dictionary of (sub)categories
			par_sub_relation - dictionary containing the relations between categories and subcategories
	"""
	categories = {}
	par_sub_relation = {}
	# iterate through all the categories
	for cat in range(1, len(category)):
		if int(category[cat][1]) != 0:
			subcat_id, subcat_description = int(category[cat][0]), category[cat][2]
			parent_id, parent_description = [(int(category[i][0]), category[i][2]) for i in range(1, len(category)) if category[i][0]==category[cat][1]][0]
			# create a dictionary par_sub_relation with categories and corresponding subcategories, might be useful later in the project...
			par_sub_relation[subcat_id] = parent_id
			# two cases, when subcats is True or False, depending on that we extract subcategories (66) or categories(14) 
			if subcats:
				categories[subcat_id] = subcat_description
			else:
				categories[parent_id] = parent_description
	return categories, par_sub_relation


def preprocess(question, featurenames, name):
	"""
	preprocess the questions
		question - string containing the text of the question
		featurenames - set of feature names
		name - string containing 'train' or 'test', depending on what file we're currently preprocessing

		return:
			valid_questions - list of non-empty questions
			featurenames - list of feature names
	"""
	# make a list with german stop-words
	stop_words = stopwords.words('german')
	stop_words = [i.decode('utf-8') for i in stop_words]
	valid_questions = []
	for i in question:
		# check if the question in the train data is valid (i.e. has a category)
		if 'name'=='train' and i[3] == 'N':
			continue
		category_text = i[4]
		# remove the punctuation
		category_text = re.sub(r'[^a-zA-Z ]',' ', category_text)
		# remove the stop words and split questions into words
		category_text = category_text.split()
		category_text = [w for w in category_text if w not in stop_words]
		for k in range(len(category_text)):
			# convert words to lowercase
			category_text[k] = category_text[k].lower()
			#save the words as features if they're from the training set (because feature selection is only being done on the training set)
			if name=='train' and category_text[k] :
				featurenames.add(category_text[k])
		# if the text of the question is not empty, append the question to the list of valid questions
		if category_text:
			i[4] = category_text
			valid_questions.append(i)
	return valid_questions, featurenames


if __name__ == '__main__':
	# extract the features
	extract_features_train_test(subcats=False)
