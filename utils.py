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
	gt = np.reshape(gt,-1)
	image = np.reshape(image,(image.shape[2]*image.shape[3],image.shape[0]))

	return image,gt

def eliminate_lacking_data(image,gt):
	cond = gt!=0 & np.any(image[:,:3],axis=1)
	values = image[cond] 
	ground_truth = gt[cond]
	return values, ground_truth

