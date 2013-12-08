import numpy as np
import unicodecsv
from nltk.corpus import stopwords
from nltk import SnowballStemmer
import re
import enchant

def extract_features(qfile='question_train.csv', qcatfile='question_category_train.csv', 
					catfile='category.csv', subcats=True, outfile='features.npz', spelling_correction = False):
	"""
	extract the numerical features from text documents
		qfile - .csv file containing the SMS Guru questions
		qcatfile - .csv file containing the relation between questions and category
		catfile - .csv file containing the categories
		subcats - Boolean parameter which decides whether to perform the analysis for categories or subcategories
		outfile - .npz file to store the output arrays, dictionary and list
		spelling_correction - an optional Boolean parameter which decides whether to use spelling correction or not, by default False
	"""
	# read from all the .csv files
	with open(qfile, 'rb') as csvfile:
		question_train = list(unicodecsv.reader(csvfile, delimiter=",", quoting=unicodecsv.QUOTE_ALL, escapechar="\\", encoding='utf-8'))
	with open(qcatfile, 'rb') as csvfile:
		question_category_train = list(unicodecsv.reader(csvfile, delimiter=",", quoting=unicodecsv.QUOTE_ALL, escapechar="\\", encoding='utf-8'))
	with open(catfile, 'rb') as csvfile:
		category = list(unicodecsv.reader(csvfile, delimiter=",", quoting=unicodecsv.QUOTE_ALL, escapechar="\\", encoding='utf-8'))

	# create a dictionary with the category names
	categories = create_categories_dict(category, True)
	# create a dictionary of german words for spelling correction
	if spelling_correction:
		german_dict = enchant.Dict("de_DE")
	# make a list with german stop-words
	stop_words = stopwords.words('german')
	stop_words = [i.decode('utf-8') for i in stop_words]
	stemmer = SnowballStemmer("german")
	questions = []
	featurenames = set()
	valid_questions = []
	# from all the questions remove the ones that don't have a category or whose length after the preprocessing is 0
	for i in question_train[1:]:
		for j in question_category_train[1:]:
			# check of the category of the question is contained in the question_category_train and if yes, do the preprocessing
			if i[0] == j[2]:
				category_text = i[4]
				# remove the punctuation
				category_text = re.sub(r'[.,?!-:;+*_=<>]',' ', category_text)
				# remove the stop words and split questions into words
				category_text = category_text.split()
				category_text = [w for w in category_text if w not in stop_words]
				for k in range(len(category_text)):
					# do the spelling correction, if specified
					if spelling_correction:
						if not german_dict.check(category_text[k]):
							try:
								category_text[k] = german_dict.suggest(category_text[k])[0]
							except:
								pass
					# convert words to lowercase
					category_text[k] = category_text[k].lower()
					# stem the words
					category_text[k] = stemmer.stem(category_text[k])
					#save the words as features
					if category_text[k]:
						featurenames.add(category_text[k])
				# if the texr of the question is not empty, append the question to the list of valid questions
				if category_text:
					i[4] = category_text
					valid_questions.append(i)
				break
	# transform featurenames from set to list (set was useful because it does not contain duplicates)
	featurenames = list(featurenames)
	categoryids = np.zeros((1, len(valid_questions)))
	# iterate over the valid questions and create the categoryids array with the question id-s and the list of questions
	for i in valid_questions:
		category_id = i[3]
		categoryids[:, valid_questions.index(i)] = category_id
		# save the questions in the list
		category_text = i[4]
		questions.append(category_text)
	# make the features matrix where each column is a data point and each row is a vector with a position for each word from the
	# featurenames list -> the column contains 1-s if the word appears in that question and 0-s if the word doesn't appear
	features = np.zeros((len(featurenames), len(valid_questions)))
	for feature_id in range(len(featurenames)):
		for q_id in range(len(questions)):
			if featurenames[feature_id] in questions[q_id]:
				features[feature_id, q_id] = 1.
	# print np.where(features[:,300]==1.)
	# print categoryids[:,300]
	# print questions[300]
	# save the arrays to the output file
	np.savez(outfile, features=features, featurenames=featurenames, categoryids=categoryids, categories=categories)

def create_categories_dict(category, subcats):
	"""
	create a dictionary with the category id as the key and the category description as the valid_questions
		category - list of entries from the category.csv file
		subcats - Boolean parameter which decides whether to perform the analysis for categories or subcategories

		return:
			categories - dictionary of (sub)categories
	"""
	categories = {}
	par_sub_relation = {}
	# iterate through all the categories
	for cat in range(1, len(category)):
		if int(category[cat][1]) != 0:
			subcat_id, subcat_description = int(category[cat][0]), category[cat][2]
			parent_id, parent_description = [(int(category[i][0]), category[i][2]) for i in range(1, len(category)) if category[i][0]==category[cat][1]][0]
			# create a dictionary par_sub_relation with categories and corresponding subcategories, might be useful later in the project...
			try:
				par_sub_relation[parent_id, parent_description].append((subcat_id, subcat_description))
			except:
				par_sub_relation[parent_id, parent_description]=[(subcat_id, subcat_description)]
			# two cases, when subcats is True or False, depending on that we extract subcategories (66) or categories(14) 
			if subcats:
				categories[subcat_id] = subcat_description
			else:
				categories[parent_id] = parent_description
	return categories

if __name__ == '__main__':
	# extract the features
	extract_features()
