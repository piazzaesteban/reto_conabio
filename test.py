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
	image, gt, gt_shape = ut.read_raster_data(args.ndvi==1)
	values, ground_truth, data_mask = ut.eliminate_lacking_data(image,gt)

	filenames = [['Naive.sav','SVM_FIXED.sav'],['Naive_NDVI.sav','SVM_NDVI.sav']]

	print('ACCURACY TEST FOR {:s}'.format(filenames[args.ndvi][args.selector]))

	loaded_model = pickle.load(open(filenames[args.ndvi][args.selector], 'rb'))
	'''
	We load back the scaler.
	'''
	filename_scaler = 'SCALER.sav'
	scaler = pickle.load(open(filename_scaler, 'rb'))

	#Test over 5 folds
	'''
	kf = KFold(n_splits=5)
	cross_score = [loaded_model.score(values[test], ground_truth[test])
		for train, test in kf.split(ground_truth)]
	accuracy = reduce(lambda x, y: x + y, cross_score) / len(cross_score)
	print('Fold result',cross_score)
	print('Mean model accuracy {:f}'.format(accuracy))
	'''

	'''
	Create a numpy array to hold the output.
	'''
	total_size = gt_shape[0] * gt_shape[1]
	output = numpy.zeros(total_size)

	'''
	Predict only on good pixels using the data mask from the eliminate_lacking_data
	method.
	'''
	X = scaler.transform(image.T[data_mask])
	y = loaded_model.predict(X)
	'''
	We put the prediction on the output array using the same mask.
	'''
	output[data_mask] = y
	output = output.reshape(gt_shape)
	
	'''
	Finally we use matplotlib to see our prediction, next to the training
	data and the first band of our dataset.
	'''
	print("Show")
	import matplotlib.pyplot as plt
	# Display the first band
	plt.subplot(131)
	image_to_show = image[:3,:]
	image_to_show = image_to_show.reshape((3, gt_shape[0], gt_shape[1]))
	plt.imshow(image_to_show.T[:,:,0], cmap=plt.cm.Greys_r) #Show the band 2 of the image
	plt.title('B_2')
	
	#Display the training data
	plt.subplot(132)
	plt.imshow(gt.reshape(gt_shape), cmap=plt.get_cmap('CMRmap'))
	plt.title('Ground truth')

	#Display the prediction
	plt.subplot(133)
	plt.imshow(output, cmap=plt.get_cmap('CMRmap'))
	plt.title('Prediction')
	plt.show()

if __name__ == '__main__':
	main()
