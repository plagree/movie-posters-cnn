#!/usr/bin/env python
#-*-coding: utf-8 -*-

import keras
from keras.preprocessing import image
import numpy as np
from keras.applications.imagenet_utils import preprocess_input
import os, sys
from random import shuffle
import pickle
import argparse
from keras import applications
from operator import mul
from tensorflow import gfile
from tensorflow import logging
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def load_img2(img, target_size=None):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    if target_size:
        wh_tuple = (target_size[1], target_size[0])
        if img.size != wh_tuple:
            img = img.resize(wh_tuple)
            return img

def pred_vector(input_img):
    img = load_img2(input_img, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    y = model.predict(x)
    yb = y.reshape(reduce(mul, y.shape, 1))
    return yb

def feature_embedding():
    # Build the VGG16 network
    model = applications.VGG16(weights='imagenet', include_top=False)
    #file_weights = 'gs://play-dataset/dataset/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
    #model.load_weights(file_weights) # 'dataset/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')
    
    # N number of movies
    N = 0
    
    years_pattern = 'gs://play-dataset/dataset/pictures/'
    years = gfile.ListDirectory(years_pattern)
    
    with gfile.Open('gs://play-dataset/number_movies.csv', 'w') as fout:
        for year in years:
            N = 0
            files = gfile.Glob(years_pattern + year + '*.jpg')
            for f in files:
                N += 1
            fout.write('%d,%s' % (N, year))
            # break
    # print N
    sys.exit(1)
    
    logging.info("There is a total of %d movies." % N)
    i = 0
    # 25088 is the dimension of the feature space of the ConvNet
    A = np.zeros((N, 25088), dtype=np.float32)
    logging.info('Starting Predictions.')
    for year in sorted(years):
        files = gfile.Glob(years_pattern + year + '*.jpg')
        for movie in sorted(files):
            img_path = 'dataset/pictures/%s/%s' % (year, movie)
            with gfile.Open(f) as fin:
                img = Image.open(fin)
                y = pred_vector(img)
                A[i, :] += y
                i += 1
        logging.info("Number of training: %d.", i)
        break

# with gfile.Open('gs://play-dataset/data.pkl', 'w') as fout:
#     pickle.dump(A[:1000,:], fout)

def pca_reduction(A):
    # 2. PCA because T-SNE requires small dimension
    pca = PCA(n_components=250, svd_solver='randomized')
    pca.fit(A)
    A = pca.transform(A)
    logging.info("PCA done.")
    return A

def tsne(A):
    tsne = TSNE()
    y = tsne.fit_transform(Y)
    return y


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Input Arguments
    parser.add_argument(
      '--train-file',
      help='GCS or local paths to training data',
      required=True
    )

    parser.add_argument(
      '--job-dir',
      help='GCS location to write checkpoints and export models',
      required=True
    )
    args = parser.parse_args()
    arguments = args.__dict__
    job_dir = arguments.pop('job_dir')

    # 1. Feature embedding
    feature_embedding()

    # 2. PCA reduction
    X = pca_reduction(X)

    with gfile.Open('gs://play-dataset/data.pkl', 'wb') as fout:
        pickle.dump(X, fout)

    sys.exit(1)

    # 3. T-SNE reduction in 2D for visualization
    y = tsne(X)

    with gfile.Open('gs://play-dataset/final_data.pkl', 'wb') as fout:
        pickle.dump(y, fout)

#A = np.zeros((40000, 25088), dtype=np.float32)
#with gfile.Open('gs://play-dataset/data.pkl', 'w') as fout:
#    for i in xrange(A.shape[0]):
#       fout.write(','.join(map(str,A[i,:]))+'\n')
#sys.exit(1)


