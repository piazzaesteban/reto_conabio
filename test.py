import numpy
from sklearn.model_selection import train_test_split,KFold
import utils as ut
import pickle
import argparse

MODEL_PATH = './Models/'

def get_arguments():
	parser = argparse.ArgumentParser(description='Reto Conabio')
	parser.add_argument('selector', type=int, 
	                   help="""an integer to select a model
	                   		 	0 - Naive Bayes
	                   		 	1 - SVM""")
	parser.add_argument('ndvi', type=int, 
	                   help='0 - Normal, 1 - NDVI')
	return parser.parse_args()

def main():
	args = get_arguments()
	image,gt = ut.read_raster_data(args.ndvi==1)
	values, ground_truth = ut.eliminate_lacking_data(image,gt)

	filenames = [['Naive.sav','SVM.sav'],['Naive_NDVI.sav','SVM_NDVI.sav']]

	print('ACCURACY TEST FOR {:s}'.format(filenames[args.ndvi][args.selector]))

	loaded_model = pickle.load(open(MODEL_PATH+filenames[args.ndvi][args.selector], 'rb'))

	#Test over 5 folds
	kf = KFold(n_splits=5)
	cross_score = [loaded_model.score(values[test], ground_truth[test])
		for train, test in kf.split(ground_truth)]
	accuracy = reduce(lambda x, y: x + y, cross_score) / len(cross_score)
	print('Fold result',cross_score)
	print('Mean model accuracy {:f}'.format(accuracy))

if __name__ == '__main__':
	main()
