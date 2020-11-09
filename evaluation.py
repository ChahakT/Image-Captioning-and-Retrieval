"""
Evaluation code for multimodal-ranking
"""
import numpy

from datasets import load_dataset
from tools import encode_sentences, encode_images

def evalrank(model, data, split='dev'):
    """
    Evaluate a trained model on either dev or test
    data options: f8k, f30k, coco
    """
    print 'Loading dataset'
    if split == 'dev':
        X = load_dataset(data, load_train=False)[1]
    else:
        X = load_dataset(data, load_train=False)[2]

    print 'Computing results...'
    train = load_dataset('CAD', load_train=True)[0]
    vectors = encode_sentences(model, train[0], verbose=False)
    # demo.retrieve_captions(model, net, train[0], vectors, 'image.jpg', k=5)
    ls = encode_sentences(model, X[0])
    lim = encode_images(model, X[1])

    (r1, r5, r10) = i2t(lim, X[0], train[0], vectors)
    print "Image to text: %.1f, %.1f, %.1f" % (r1, r5, r10)
    # (r1i, r5i, r10i, medri) = t2i(lim, ls)
    # print "Text to image: %.1f, %.1f, %.1f, %.1f" % (r1i, r5i, r10i, medri)

def i2t(images, t_captions, captions, cvecs, npts=None):
    """
    Images->Text (Image Annotation)
    Images: (5N, K) matrix of images
    Captions: (5N, K) matrix of captions
    """
    dict_true={'Black':0,'Blue':0,'Brown':0,'Collar':0,'Cyan':0,'Female':0,
    'Floral_pattern':0,'Graphics_pattern':0,'Gray':0,'Green':0,'High_exposure':0,
    'Many_Colours':0,'Necktie':0,'Placket':0,'Plaid_pattern':0,'Purple':0,
    'Red':0,'Scarf':0,'Solid_pattern':0,'Spotted_pattern':0,'Striped_pattern':0,
    'White':0,'Yellow':0}

    dict_cp={'Black':0,'Blue':0,'Brown':0,'Collar':0,'Cyan':0,'Female':0,
    'Floral_pattern':0,'Graphics_pattern':0,'Gray':0,'Green':0,'High_exposure':0,
    'Many_Colours':0,'Necktie':0,'Placket':0,'Plaid_pattern':0,'Purple':0,
    'Red':0,'Scarf':0,'Solid_pattern':0,'Spotted_pattern':0,'Striped_pattern':0,
    'White':0,'Yellow':0}
    if npts == None:
        npts = images.shape[0] 
    index_list = []

    ranks1 = numpy.zeros(npts)
    ranks3 = numpy.zeros(npts)
    ranks5 = numpy.zeros(npts)
    for index in range(npts):

        # Get query image
        im = images[index].reshape(1, images.shape[1])

        # Compute scores
        d = numpy.dot(im, cvecs.T).flatten()
        inds = numpy.argsort(d)[::-1]
        index_list.append(inds[0])

        # Score
        # rank = 1e20
        # for i in range(5*index, 5*index + 5, 1):
        #     tmp = numpy.where(inds == i)[0][0]
        #     if tmp < rank:
        #         rank = tmp
        # ranks[index] = rank
        sentences = [captions[a] for a in inds[:3]]
        words=[]
        for s in sentences:
            li=s.split(' ')
            words.extend(li)
        words=set(words)
        t_sentences = t_captions[index]
        t_words=t_sentences.split(' ')
        t_words= set(t_words)
        if len(t_words & words) >2:
            ranks1[index]=1

        sentences = [captions[a] for a in inds[:50]]
        words=[]
        for s in sentences:
            li=s.split(' ')
            words.extend(li)
        words=set(words)
        t_sentences = t_captions[index]
        t_words=t_sentences.split(' ')
        
        for w in t_words:
            wset=set([w])
            
            dict_true[w]=dict_true[w]+1
            if wset.issubset(words):
                dict_cp[w]=dict_cp[w]+1
        t_words= set(t_words)
        if len(t_words & words) >2:
            ranks3[index]=1

        sentences = [captions[a] for a in inds[:10]]
        words=[]
        for s in sentences:
            li=s.split(' ')
            words.extend(li)
        words=set(words)
        t_sentences = t_captions[index]
        t_words=t_sentences.split(' ')
        t_words= set(t_words)
        if index in range(5):
            print words, t_words
        if len(t_words & words) >1:
            ranks5[index]=1

    # Compute metrics
    sum=0
    for q,a in zip(dict_cp, dict_true):
        ans=100
        if dict_true[a]!=0:
            ans=100.0*dict_cp[q]/dict_true[a]
        print q, ans
        sum+=ans
    sum=sum/23.0
    print sum
    r1 = 100.0 * sum(ranks1) / len(ranks1)
    r5 = 100.0 * sum(ranks3) / len(ranks3)
    r10 = 100.0 * sum(ranks5) / len(ranks5)
    # medr = numpy.floor(numpy.median(ranks)) + 1
    return (r1, r5, r10)

def t2i(images, captions, npts=None):
    """
    Text->Images (Image Search)
    Images: (5N, K) matrix of images
    Captions: (5N, K) matrix of captions
    """
    if npts == None:
        npts = images.shape[0] / 5
    ims = numpy.array([images[i] for i in range(0, len(images), 5)])

    ranks = numpy.zeros(5 * npts)
    for index in range(npts):

        # Get query captions
        queries = captions[5*index : 5*index + 5]

        # Compute scores
        d = numpy.dot(queries, ims.T)
        inds = numpy.zeros(d.shape)
        for i in range(len(inds)):
            inds[i] = numpy.argsort(d[i])[::-1]
            ranks[5 * index + i] = numpy.where(inds[i] == index)[0][0]

    # Compute metrics
    r1 = 100.0 * len(numpy.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(numpy.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(numpy.where(ranks < 10)[0]) / len(ranks)
    medr = numpy.floor(numpy.median(ranks)) + 1
    return (r1, r5, r10, medr)
