# Image Description and Retrieval by building Deep Neural Network and Multimodal Embedding

**Objective**

We will be trying to create a model that generates natural language descriptions of images and their regions.We are using datasets of
images and their sentence descriptions to learn about the inter-modal correspondences between language and visual data.We will be
using a novel combination of Convolutional Neural Networks over image regions, bidirectional Recurrent Neural Networks over
sentences, and a structured objective that aligns the two modalities through a multimodal embedding.We will try to test our images on a
pretrained multimodal Recurrent Neural Network architecture that takes an input image and generates its description in text.

**Datasets**

We will use the Flickr8K, Flickr30K and MSCOCO datasets in our experiments. These datasets contain 8,000, 31,000 and 123,000
images respectively and each is annotated with 5 sentences using Amazon Mechanical Turk.

\
**Image Captioning Pseudocode**

import demo, tools, datasets

net = demo.build_convnet() ‘’’Building VGG 19 convolutional layers, pooling layers and fully connected
 Layers and loading the weights to find probabilities in softmax layer’’’
 
model = tools.load_model() ‘’’Load the embedding model and build image and sentence encoding into it’’’

train = datasets.load_dataset('f8k', load_train=True)[0] ‘’’Loading captions and numpy Image features of
 Flickr8k’’’
 
vectors = tools.encode_sentences(model, train[0], verbose=False) ’’Embed the sentences by creating h-dim vectors’’

\# Load the image

im = load_image("download (1).jpg") ‘’’Loading the image to be captioned’’’

\# Run image through convnet

feats = compute_features(net, im).flatten() ‘’’Pass the image through CNN and then get the features from last layer’’

feats /= norm(feats)’

feats = tools.encode_images(model, feats[None,:]) ‘’’Encode these features in the embedding’’’

captions=train[0]

scores = numpy.dot(feats, vectors.T).flatten() ‘’’Calculate dot similarity of the image from all all sentence vectors’’’

sorted_args = numpy.argsort(scores)[::-1] ‘’’Sort them in decreasing order’’’

sentences = [captions[a] for a in sorted_args[:5]] ‘’’Find sentences corresponding to top 5 dot products’’’

print sentences

\
**Image Retrieval Pseudocode**

images=[]

name="The boys play baseball"

caps=[] ‘’’Preprocessing for the sentence for which images have to be found out’’’

caps=name.splitlines()

train_caps=[]

with open('/home/chahak/f8k/f8k_train_caps.txt', 'rb') as f:

 for line in f:
 
 train_caps.append(line.strip())
 
feats_new=tools.encode_sentences(model, caps, verbose=False ) ‘’’Encode this sentence in the embedding’’’

imvecs=numpy.load('im_feats.npy') ‘’’’Load the numpy image features of Flickr 8K’’’

from tempfile import TemporaryFile

outfile = TemporaryFile() ‘’’Encode image features into the embedding’’’

numpy.save(outfile,imvecs)

numpy.save("im_feats.npy",imvecs)

numpy.savetxt("im_feats.txt",imvecs)

import numpy

import skimage.transform

score_new=numpy.dot(feats_new, imvecs.T).flatten() ‘’’Dot similarity of all images and the given sentence’’’

sorted_args1 = numpy.argsort(score_new)[::-1] ‘’’Sort them in decreasing order’’’

for a in sorted_args1[:5]:

print a

results = [images[a] for a in sorted_args1[:5]] ‘’’Get the images corresponding to it’’’

from IPython.display import Image,display

path_f8k="/home/chahak/Downloads/flickr_8k_data/Flicker8k_Dataset/"

for i in results:

imgname=path_f8k+i

display(Image(imgname)) ‘’’’Display the images’’’
