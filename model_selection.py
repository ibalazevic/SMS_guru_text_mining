import numpy as np
import pylab as pl
from extract_features import extract_features
from sklearn.naive_bayes import MultinomialNB
from sklearn.lda import LDA
from sklearn.cross_validation import KFold
from sklearn.cross_validation import StratifiedKFold
import sklearn.metrics as metrics
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import label_binarize
from sklearn.cross_validation import train_test_split
from sklearn.multiclass import OneVsRestClassifier



def test_classifier_parameters(classifier = 'classifier_name', infile = 'features.npz', subcats=True, norm = False):
	"""
	main function to determine number of features to be used for classification and classifier hyperparameters
		classifier - classifier to be used, Naive Bayes or LDA
		infile - .csv file with the extracted features
		norm - Boolean parameter whether to normalize the Naive Bayes for unbalanced class sizes
	"""
	# extract numpy arrays, lists and dictionaries from the features.npz
	extract_features(subcats=subcats)
	features_file = np.load(infile)
	features, featurenames, categoryids, categories = features_file['features'], features_file['featurenames'], \
													features_file['categoryids'], features_file['categories'].item()

	labels = categoryids[0,:]
	features = features.T
	categoryids = categoryids.T
	if classifier == 'Naive Bayes':
		# if True, do the normalization for unbalanced class sizes
		if norm:
			features = normalize(features, categoryids, categories)
		classify('nb', features, labels, categories)
	else:
		classify('lda', features, labels, categories)

def normalize(features, categoryids, categories):
	"""
	the normalization for unbalanced class sizes
		features - the feature matrix
		categoryids - category id-s for each fea vectors
		categories - dictionary of categories

		return:
			normalized features matrix
	"""
	alpha = 1.
	for i in categories.keys():
		indices_class = np.where(categoryids==i)[0]
		sum_all = np.sum(features[indices_class])
		features[indices_class]/=float(sum_all)*alpha
	return features


def classify(classifier, features, labels, categories, nfolds=10, strat_kf=True):
	'''
	apply the classifier using 1 fold of the data for testing and n-1 for training and compute different accuracy
	measures (precision, recall, f1 score)
		classifier - the classifier to be used, Naive Bayes or LDA
		features - input data to be classified, here the numpy matrix of the feature vectors
		labels - the classes of the data
		categories - different possible categories (66 for subcategories or 14 for categories)
		nfolds - number of folds for crossvalidation
		strat_kf - Boolean, if True perform the stratified k-fold crossvalidation, otherwise perform the normal crossvalidation
	'''
	# choose the classifier
	if classifier=='nb':
		clf = MultinomialNB(alpha=0.5)
		k_values = [500, 1000, 2000, 5000, 8000, 'all']
	else:
		clf = LDA()
		k_values = [1000, 1500, 2000, 2500, 3000, 5000]
	# decide whether to use normal or stratified cross-validation
	if strat_kf:
		kf = StratifiedKFold(labels, nfolds)
	else:
		kf = KFold(labels.shape[0], n_folds=nfolds, indices=False)
	# save the precision, recall and f1-score for plotting only
	accuracies = np.zeros((3, len(k_values)))
	accuracies_barplot = np.zeros((3, len(categories)))
	# run the cross-validation for different numbers of features selectes
	for k in range(len(k_values)):
		# select k best features
		features_trunc = select_features(features, labels, chi2, k_values[k])
		average_f1_score = 0
		average_precision = 0
		average_recall = 0
		# calculate and plot the roc curve
		roc(features_trunc, labels, categories, clf)
		# split the data into training and test set and do the crossvalidation
		for train, test in kf:
			features_train, features_test, categoryids_train, categoryids_test = features_trunc[train], features_trunc[test], \
																				labels[train], labels[test]
			# fit the classifier
			clf.fit(features_train, categoryids_train)
			# predict the labels
			predicted_labels = clf.predict(features_test)
			# compute the classification accuracy with the weighted average
			precision, recall, f1 = compute_accuracy_measures(categoryids_test, predicted_labels, categories.keys())
			average_precision += precision
			average_recall += recall
			average_f1_score += f1
			# compute the classification accuracy without averaging (for the bar plot)
			# precision, recall, f1 = compute_accuracy_measures(categoryids_test, predicted_labels, categories.keys(), average=None)
			# accuracies_barplot[0, :] += precision
			# accuracies_barplot[1, :] += recall
			# accuracies_barplot[2, :] += f1
		# compute the average accuracy scores for every cross-validation step
		average_precision /= nfolds
		average_recall /= nfolds
		average_f1_score /= nfolds
		accuracies[:, k] = average_precision, average_recall, average_f1_score
		# accuracies_barplot /= nfolds
	# bar_plot_score(accuracies_barplot, categories)
	plot_accuracy_measures(accuracies, k_values)

def select_features(X, y, score_func, k):
	"""
	select k best features for classification
		X - feature matrix
		y - labels
		score_func - scoring function for feature selection (chi2 or f_classif)
		k - the number of features to be selected

		return:
			feature matrix reduced to k best features
	"""
	feature_selector = SelectKBest(score_func=score_func, k=k)
	return feature_selector.fit_transform(X, y)


def compute_accuracy_measures(labels_true, labels_predicted, labels, average='weighted'):
	'''
	update the accuracy scores
		labels_true - true labels of the test data
		labels_predicted - labels predicted by the classifier
		average - an argument of the sklearn precision_score, recall_score and f1_score functions, 'weighted' by default

		return:
			precision, recall, f1 - the updated scores
	'''
	precision = metrics.precision_score(labels_true, labels_predicted, labels=labels, average=average)
	recall = metrics.recall_score(labels_true, labels_predicted, labels=labels, average=average)
	f1 = metrics.f1_score(labels_true, labels_predicted, labels=labels, average=average)
	return precision, recall, f1

def plot_accuracy_measures(accuracies, k_values):
	"""
	plot the precision, recall and f1-score for different number of features selected
	"""
	pl.figure()
	pl.plot(np.arange(len(k_values)), accuracies[0,:], 'r', marker='8', markersize=10, fillstyle='full')
	pl.plot(np.arange(len(k_values)), accuracies[1,:], 'g', marker='8', markersize=10, fillstyle='full')
	pl.plot(np.arange(len(k_values)), accuracies[2,:], 'b', marker='8', markersize=10, fillstyle='full')
	pl.legend(['precision', 'recall', 'f1-score'], loc=4)
	pl.ylim(np.min(accuracies)-0.05, np.max(accuracies)+0.05)
	pl.xticks(range(len(k_values)), k_values)
	pl.xlabel('number of features')
	pl.show()


def bar_plot_score(accuracies, categories):
	'''
	make the bar plot accross categories of the accuracy scores
	'''
	num_classes = len(categories)
	labels = categories.keys()
	class_names = categories.values()
	class_names1 = [class_name for (accuracy, class_name) in sorted(zip(accuracies[0, :],class_names), reverse=True)]
	class_names2 = [class_name for (accuracy, class_name) in sorted(zip(accuracies[1, :],class_names), reverse=True)]
	class_names3 = [class_name for (accuracy, class_name) in sorted(zip(accuracies[2, :],class_names), reverse=True)]
	width=0.6
	pl.subplot(311)
	pl.bar(range(num_classes), sorted(accuracies[0, :], reverse=True), color='b', width=width)
	pl.ylabel('precision')
	pl.ylim(0,1)
	pl.xticks(np.arange(num_classes)+width/2., class_names1)
	pl.tick_params(axis='both', labelsize=7.5)
	pl.subplot(312)
	pl.bar(range(num_classes), sorted(accuracies[1, :], reverse=True), color='r', width=width)
	pl.ylabel('recall')
	pl.ylim(0,1)
	pl.xticks(np.arange(num_classes)+width/2., class_names2)
	pl.tick_params(axis='both', labelsize=7.5)
	pl.subplot(313)
	pl.bar(range(num_classes), sorted(accuracies[2, :], reverse=True), color='g', width=width)
	pl.xlabel('category')
	pl.ylabel('f1-score')
	pl.ylim(0,1)
	pl.xticks(np.arange(num_classes)+width/2., class_names3)
	pl.tick_params(axis='both', labelsize=7.5)
	pl.show()

def roc(features_trunc, labels, categories, classifier):
	"""
	compute and plot the roc curve for the given classifier
		features_trunc - features matrix truncated to the k best features
		labels - the classes of the data
		categories - different possible categories (66 for subcategories or 14 for categories)
		classifier - MultinomialNB or lda
	"""
	# divide the data into training and test set
	features_train, features_test, categoryids_train, categoryids_test = train_test_split(features_trunc, labels, test_size=.1,random_state=0)
	# define the OneVsRestClassifier with the given classifier (LDA or Naive Bayes)
	clf = OneVsRestClassifier(classifier)
	# train the classifier and compute the probabilities for the test data labels
	clf_fit = clf.fit(features_train, categoryids_train)
	labels_score = clf_fit.predict_proba(features_test)
	# binarize the labels (necessary for the roc curve)
	categoryids_test = label_binarize(categoryids_test, classes=categories)
	# compute the false positive rate, true positive rate and the thresholds
	fpr, tpr, thresholds = metrics.roc_curve(categoryids_test.ravel(), labels_score.ravel())
	# compute the area under the curve
	roc_auc = metrics.auc(fpr, tpr)
	# plot the roc curve
	pl.clf()
	pl.plot(fpr, tpr, 'r',label='micro-average ROC curve (area = {0:0.2f})'''.format(roc_auc), linewidth=2)
	pl.plot([0, 1], [0, 1], 'k--', linewidth=2)
	pl.xlim([0.0, 1.0])
	pl.ylim([0.0, 1.05])
	pl.xlabel('false positive rate')
	pl.ylabel('true positive rate')
	pl.title('Receiver operating characteristic for micro-averaged classification scores')
	pl.legend(loc="lower right")
	pl.show()

if __name__ == '__main__':
	test_classifier_parameters(classifier='Naive Bayes')
