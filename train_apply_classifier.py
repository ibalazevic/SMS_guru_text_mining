import numpy as np
from extract_features_train_test import extract_features_train_test
from sklearn.naive_bayes import MultinomialNB
from sklearn.lda import LDA
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

def train_apply_classifier(classifier = 'classifier_name', qfile_train = 'question_train.csv',
							qcatfile_train = 'question_category_train.csv', catfile = 'category.csv',
							qfile_test = 'question_test.csv', subcats=True):
	"""
	train and apply the specified classifier
		classifier - the classifier to use, either Naive Bayes or LDA
		qfile_train - .csv file containing the test data
		qcatfile_train - .csv file containing the relation between questions and category
		catfile - .csv file containing the categories
		qfile_test - .csv file containing the test data

		return:
			labels - predicted categories for the test data
	"""

	# extract numpy arrays, lists and dictionaries from the .npz file for the train and test data
	extract_features_train_test(subcats=subcats)
	features_file = np.load('features_train_test.npz')
	features_train, features_test, featurenames, categoryids_train, categories = \
												features_file['features_train'], features_file['features_test'], \
												features_file['featurenames'], features_file['categoryids_train'], \
												features_file['categories'].item()
	labels_train = categoryids_train[0,:]
	features_train = features_train.T
	features_test = features_test.T
	# train and apply the classifier
	if classifier == 'Naive Bayes':
		# select k best features
		features_train, features_test, featurenames = select_features(features_train, features_test, labels_train, featurenames, k=5000)
		# compute the labels of the test data
		labels = classify('nb', features_train, labels_train, features_test)
	else:
		# select k best features
		features_train, features_test, featurenames = select_features(features_train, features_test, labels_train, featurenames, k=3000)
		# compute the labels of the test data
		labels = classify('lda', features_train, labels_train, features_test)
	return labels

def select_features(features_train, features_test, labels_train, featurenames, k):
	"""
	select the k best features from the train data and choose only those features for the test data
		features_train - train data
		features_test - test data
		labels_train - the classes of the train data
		featurenames - a list with the names for each feature
		k - number of features to be selected

		return:
			features_train - train data reduced to k features
			features_test - test data reduced to k features
			featurenames - names of the k features
	"""
	# initialize the feature selector
	feature_selector = SelectKBest(score_func=chi2, k=k)
	features_scores = feature_selector.fit(features_train, labels_train).scores_
	# from feature scores, select k best ones
	features_kbest_scores = features_scores.argsort()[-k:][::-1]
	# reduce train and test features, as well as the featurenames to the k best features
	features_train = features_train[:, features_kbest_scores]
	features_test = features_test[:, features_kbest_scores]
	featurenames = [featurenames[i] for i in features_kbest_scores]
	return features_train, features_test, featurenames

def classify(classifier, features_train, labels_train, features_test):
	'''
	train and apply the classifier
		classifier - Naive Bayes or LDA
		features_train - train data
		labels_train - the classes of the train data
		features_test - test data

		return:
			labels_test_predicted - predicted categories for the test data
	'''
	# initialize the classifier
	if classifier == 'nb':
		clf = MultinomialNB(alpha=0.5)
	else:
		clf = LDA()
	# train the classifier
	clf.fit(features_train, labels_train)
	# predict the labels
	labels_test_predicted = clf.predict(features_test)
	return np.array([labels_test_predicted]).T

if __name__ == '__main__':
	train_apply_classifier(classifier='Naive Bayes')
