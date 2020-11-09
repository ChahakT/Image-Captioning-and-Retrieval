# -*- coding: utf-8 -*-
"""
Created on Sun Mar 19 09:22:17 2017

@author: chahak
"""

import demo, tools, datasets
net = demo.build_convnet()
model = tools.load_model()
train = datasets.load_dataset('f8k', load_train=True)[0]
vectors = tools.encode_sentences(model, train[0], verbose=False)
demo.retrieve_captions(model, net, train[0], vectors, 'child.jpg', k=5)