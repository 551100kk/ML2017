from argparse import ArgumentParser


import word2vec
import numpy as np
import nltk


from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from adjustText import adjust_text
    

parser = ArgumentParser()
parser.add_argument('--train', action='store_true',
                    help='Set this flag to train word2vec model')
parser.add_argument('--corpus-path', type=str, default='hp/all',
                    help='Text file for training')
parser.add_argument('--model-path', type=str, default='hp/model.bin',
                    help='Path to save word2vec model')
parser.add_argument('--plot-num', type=int, default=1000,
                    help='Number of words to perform dimensionality reduction')
args = parser.parse_args()


if args.train:
    # DEFINE your parameters for training
    WORDVEC_DIM = 87
    
    print (args.model_path)
    print (args.plot_num)

    # train model
    word2vec.word2vec(
        train=args.corpus_path,
        output=args.model_path,
        size=WORDVEC_DIM,
        verbose=True)

    print ('end')

else:
    # load model for plotting
    model = word2vec.load(args.model_path)

    vocabs = []                 
    vecs = []                   
    for vocab in model.vocab:
        vocabs.append(vocab)
        vecs.append(model[vocab])
    vecs = np.array(vecs)[:args.plot_num]
    vocabs = vocabs[:args.plot_num]

    '''
    Dimensionality Reduction
    '''
    tsne = TSNE(n_components=2)
    reduced = tsne.fit_transform(vecs)


    '''
    Plotting
    '''
    

    # filtering
    use_tags = set(['JJ', 'NNP', 'NN', 'NNS'])
    puncts = ["'", '.', ':', ";", ',', "?", "!", u"’"]
    
    
    plt.figure()
    texts = []
    for i, label in enumerate(vocabs):
        pos = nltk.pos_tag([label])
        if (label[0].isupper() and len(label) > 1 and pos[0][1] in use_tags
                and all(c not in label for c in puncts)):
            x, y = reduced[i, :]
            texts.append(plt.text(x, y, label))
            plt.scatter(x, y)

    adjust_text(texts, arrowprops=dict(arrowstyle='-', color='k', lw=0.5))
    plt.savefig('hp.png', dpi=1200)