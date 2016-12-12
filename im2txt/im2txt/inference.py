from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os

import tensorflow as tf

from skimage import io
import pylab
import urllib

from im2txt import configuration
from im2txt import inference_wrapper
from im2txt.inference_utils import caption_generator
from im2txt.inference_utils import vocabulary

FLAGS = tf.flags.FLAGS

def inference_one(sess, restore_fn, model, vocab, filename):
    # Prepare the caption generator. Here we are implicitly using the default
    # beam search parameters. See caption_generator.py for a description of the
    # available beam search parameters.
    generator = caption_generator.CaptionGenerator(model, vocab)

    with tf.gfile.GFile(filename, "r") as f:
        image = f.read()
        captions = generator.beam_search(sess, image)
        print("Captions for image %s:" % os.path.basename(filename))
        pylab.imshow(io.imread(filename))
        pylab.show()
        for i, caption in enumerate(captions):
            # Ignore begin and end words.
            sentence = [vocab.id_to_word(w) for w in caption.sentence[1:-1]]
            sentence = " ".join(sentence)
            print("  %d) %s (p=%f)" % (i, sentence, math.exp(caption.logprob)))
            
def inference_from_url(sess, restore_fn, model, vocab, url):
    tmp_file = "/home1/im2txt/data/mytest/temp.jpg"
    urllib.urlretrieve(url,tmp_file)
    inference_one(sess, restore_fn, model, vocab, tmp_file)
   
