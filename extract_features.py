import numpy as np
import unicodecsv
from nltk.corpus import stopwords
from nltk import SnowballStemmer
import re
import enchant

def extract_features(qfile='question_train.csv', qcatfile='question_category_train.csv', 
					catfile='category.csv', subcats=True, outfile='features.npz', spelling_correction = False,
					numbers_feature=False, stemming=False):
	"""
	extract the numerical features from text documents
		qfile - .csv file containing the SMS Guru questions
		qcatfile - .csv file containing the relation between questions and category
		catfile - .csv file containing the categories
		subcats - Boolean parameter which decides whether to perform the analysis for categories or subcategories
		outfile - .npz file to store the output arrays, dictionary and list
		spelling_correction - an optional Boolean parameter which decides whether to use spelling correction or not, by default False
		numbers_feature - an optional Boolean parameter whether to include an indication about a number in the datapoint, by default False
		stemming - an optional Boolean parameter whether to do the stemming or not, by default False
	"""
	# read from all the .csv files
	question_train, question_category_train, category = read_files(qfile, qcatfile, catfile)
	# create a dictionary with the category names
	categories, par_sub_relation = create_categories_dict(category, subcats)
	questions = []
	featurenames = set()
	valid_questions = []
	# from all the questions remove the ones that don't have a category or whose length after the preprocessing is 0
	valid_questions, featurenames = preprocess(question_train[1:], spelling_correction, stemming, featurenames, numbers_feature)
	# optionally add the feature which says if the question contained a number
	if numbers_feature:
		featurenames.append('contains_number')
	categoryids = np.zeros((1, len(valid_questions)))
	# iterate over the valid questions and create the categoryids array with the question id-s and the list of questions
	for i in valid_questions:
		if subcats:
			category_id = i[3]
		else:
			category_id = par_sub_relation[int(i[3])]
		categoryids[:, valid_questions.index(i)] = category_id
		# save the questions in the list
		category_text = i[4]
		questions.append(category_text)
	# make the features matrix where each column is a data point and each row is a vector with a position for each word from the
	# featurenames list -> the column contains 1-s if the word appears in that question and 0-s if the word doesn't appear
	features = np.zeros((len(featurenames), len(valid_questions)))
	for feature_id in range(len(featurenames)):
		for q_id in range(len(questions)):
			if featurenames[feature_id] in questions[q_id] or (featurenames[feature_id] == 'contains_num' and questions[q_id][-1]==1):
				features[feature_id, q_id] = 1.
	# save the arrays to the output file
	np.savez(outfile, features=features, featurenames=featurenames, categoryids=categoryids, categories=categories)
	


def read_files(qfile, qcatfile, catfile):
	"""
	read from .csv files
		qfile - .csv file containing the SMS Guru questions
		qcatfile - .csv file containing the relation between questions and category
		catfile - .csv file containing the categories
	"""
	with open(qfile, 'rb') as csvfile:
		question_train = list(unicodecsv.reader(csvfile, delimiter=",", quoting=unicodecsv.QUOTE_ALL, escapechar="\\", encoding='utf-8'))
	with open(qcatfile, 'rb') as csvfile:
		question_category_train = list(unicodecsv.reader(csvfile, delimiter=",", quoting=unicodecsv.QUOTE_ALL, escapechar="\\", encoding='utf-8'))
	with open(catfile, 'rb') as csvfile:
		category = list(unicodecsv.reader(csvfile, delimiter=",", quoting=unicodecsv.QUOTE_ALL, escapechar="\\", encoding='utf-8'))
	return question_train, question_category_train, category

def preprocess(question, spelling_correction, stemming, featurenames, numbers_feature):
	"""
	preprocess the questions
		question - string containing the text of the question
		spelling_correction - Boolean parameter which decides whether to use spelling correction or not, by default False
		stemming - Boolean parameter whether to do the stemming or not
		featurenames - set of feature names
		numbers_feature - Boolean parameter whether to include an indication about a number in the datapoint

		return:
			valid_questions - list of non-empty questions
			featurenames - list of feature names
	"""
	# make a list with german stop-words
	stop_words = stopwords.words('german')
	stop_words = [i.decode('utf-8') for i in stop_words]
	# create the stemmer
	stemmer = SnowballStemmer("german")
	# create a dictionary of german words for spelling correction
	if spelling_correction:
		german_dict = enchant.Dict("de_DE")
	valid_questions = []
	for i in question:
		# check if the question has a category
		if i[3] == 'N':
			continue
		contains_num = 0
		category_text = i[4]
		if re.search('\d+', category_text):
			contains_num = 1
		# remove the punctuation
		category_text = re.sub(r'[^a-zA-Z ]',' ', category_text)
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
			if stemming:
				category_text[k] = stemmer.stem(category_text[k])
			#save the words as features
			if category_text[k]:
				featurenames.add(category_text[k])
		# if the text of the question is not empty, append the question to the list of valid questions
		if category_text:
			i[4] = category_text 
			if numbers_feature:
				i[4] += [contains_num]
			valid_questions.append(i)
	return valid_questions, list(featurenames)


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
			# create a dictionary par_sub_relation with categories and corresponding subcategories
			par_sub_relation[subcat_id] = parent_id
			# two cases, when subcats is True or False, depending on that we extract subcategories (66) or categories(14) 
			if subcats:
				categories[subcat_id] = subcat_description
			else:
				categories[parent_id] = parent_description
	return categories, par_sub_relation

if __name__ == '__main__':
	# extract the features
	extract_features(subcats=True, spelling_correction=False, numbers_feature = False, stemming=False)
