import rasterio
import numpy as np

DATA_PATH = './Data/Datasets/'

def read_raster_data(NDVI):
	r_bands = []
	r_bands.append(rasterio.open(DATA_PATH+'L8_B2.TIF'))
	r_bands.append(rasterio.open(DATA_PATH+'L8_B3.TIF'))
	r_bands.append(rasterio.open(DATA_PATH+'L8_B4.TIF'))
	r_bands.append(rasterio.open(DATA_PATH+'L8_B5.TIF'))

	r_gt = rasterio.open('./Data/Training/training.tif')
	gt = r_gt.read()

	bands = []
	for b in r_bands:
		a_band = np.zeros(shape=gt.shape).astype(np.uint16)
		bands.append(b.read(out=a_band))

	
	if (NDVI):
		diff = bands[3]-bands[0]
		add = bands[3]+bands[0]
		ndvi = diff/add
		bands.append(ndvi)
	
	image = np.array(bands)
	'''
	gt = np.reshape(gt,-1)
	image = np.reshape(image,(image.shape[2]*image.shape[3],image.shape[0]))
	'''
	gt = np.squeeze(gt)
	image = np.squeeze(image)

	print(gt.shape)
	print(image.shape)

	shape = image.shape
	return image.reshape((shape[0], shape[1] * shape[2])),gt.reshape((shape[1] * shape[2])), gt.shape

def eliminate_lacking_data(image,gt):
	'''
	We expect to have a (num_pixels, bands) shape, so we transpose the matrix.
	'''
	image = image.T
	'''
	We separate the mask creation, one for the gt one for the data, the data
	mask will be used at the end.
	'''
	test = np.any(image[:,:3],axis=1)
	gt_mask = gt!=0
	data_mask = np.any(image[:,:3],axis=1)
	cond = gt_mask & data_mask
	'''
	Remove the extra dimension.
	'''
	values = np.squeeze(image[cond])
	ground_truth = np.squeeze(gt[cond])
	'''
	Return the data mask to use it to visualize the prediction.
	'''
	return values, ground_truth, data_mask

