"""Neighbors, t-SNE, original-space K-means, and a fixed retrieval comparison."""
import argparse
import json
from collections import Counter
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sgns import load_vectors, normalize, tokenize, positive_pairs
from train import save_json

ROOT = Path(__file__).resolve().parent
DOMAIN_QUERIES = ['gene','protein','cancer','immune','brain','dna','rna','neurons']
GENERIC = set('study studies results result using used use showed show shown suggest suggests revealed demonstrate demonstrated role roles novel significantly significant including however associated different analysis expression cells cell also furthermore therefore mechanism mechanisms function functions regulation regulates regulated based data identified increased decrease decreased increase effect effects human mouse mice levels level compared observed response responses important specific treatment activity dependent mediated signaling pathway pathways'.split())


def bm25_matrix(token_docs, k1=1.5, b=.75):
    words = sorted({w for doc in token_docs for w in doc})
    lookup = {w:i for i,w in enumerate(words)}
    rows, cols, values = [], [], []
    for i, doc in enumerate(token_docs):
        for word, count in Counter(doc).items():
            rows.append(i);cols.append(lookup[word]);values.append(count)
    tf = sp.csr_matrix((values,(rows,cols)),shape=(len(token_docs),len(words)),dtype=float)
    df = np.asarray((tf>0).sum(0)).ravel()
    idf = np.log(1+(len(token_docs)-df+.5)/(df+.5))
    lengths = np.array(list(map(len,token_docs)))
    coo = tf.tocoo()
    weights = idf[coo.col]*coo.data*(k1+1)/(coo.data+k1*(1-b+b*lengths[coo.row]/lengths.mean()))
    return sp.csr_matrix((weights,(coo.row,coo.col)),shape=tf.shape), lookup


def document_vectors(token_docs, words, vectors):
    lookup = {w:i for i,w in enumerate(words)}
    df = Counter(w for doc in token_docs for w in set(doc) if w in lookup)
    idf = np.array([np.log((1+len(token_docs))/(1+df[w]))+1 for w in words])
    def encode(tokens):
        counts = Counter(w for w in tokens if w in lookup)
        if not counts:
            return np.zeros(vectors.shape[1],dtype=np.float32)
        ids = np.array([lookup[w] for w in counts])
        weights = np.array([counts[w] for w in counts])*idf[ids]
        return normalize(np.average(vectors[ids],axis=0,weights=weights))
    return np.stack([encode(doc) for doc in token_docs]), encode, lookup


def top_indices(scores, k=5):
    return np.argsort(-scores,kind='stable')[:k]


def retrieval_metrics(rankings, targets, k=5):
    ranks = [(list(r[:k]).index(t)+1 if t in r[:k] else None) for r,t in zip(rankings,targets)]
    return {'target_recall@5':float(np.mean([r is not None for r in ranks])),
            'MRR@5':float(np.mean([1/r if r else 0 for r in ranks]))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output',type=Path,default=ROOT/'outputs')
    p.add_argument('--queries',type=Path,default=ROOT/'queries.json')
    args = p.parse_args()
    out = args.output
    words, vectors = load_vectors(out)
    unit = normalize(vectors)
    lookup = {w:i for i,w in enumerate(words)}
    neighbors = {}
    for word in DOMAIN_QUERIES:
        if word not in lookup:
            continue
        sim = unit @ unit[lookup[word]]
        sim[lookup[word]] = -np.inf
        neighbors[word] = [{'word':words[i],'cosine':float(sim[i])} for i in top_indices(sim)]
    save_json(out/'neighbors.json',neighbors)
    history = json.loads((out/'history.json').read_text())
    plt.figure(figsize=(7,3.4))
    plt.plot([x['epoch'] for x in history],[x['probe_loss'] for x in history],'-o',label='Fixed training probe')
    plt.plot([x['epoch'] for x in history[1:]],[x['train_loss'] for x in history[1:]],'-o',label='Epoch mean')
    plt.xlabel('Epoch');plt.ylabel('SGNS loss (sum over negatives)');plt.legend();plt.tight_layout()
    plt.savefig(out/'loss.png',dpi=180);plt.close()
    # Fixed frequency-based selection, independent of learned vectors.
    candidates = [i for i,w in enumerate(words) if w not in ENGLISH_STOP_WORDS and w not in GENERIC and len(w)>2][:300]
    if len(candidates)<100:
        raise ValueError('Need at least 100 non-special words for clustering.')
    X = unit[candidates]
    clustering = KMeans(n_clusters=8,random_state=42,n_init=20).fit(X)
    clusters = []
    for label in range(8):
        positions = np.flatnonzero(clustering.labels_==label)
        center = normalize(clustering.cluster_centers_[label])
        order = positions[np.argsort(-(X[positions]@center))]
        clusters.append({'cluster':label,'size':len(positions),
                         'representatives':[words[candidates[i]] for i in order[:10]],
                         'words':[words[candidates[i]] for i in positions]})
    save_json(out/'clusters.json',{'algorithm':'KMeans on L2-normalized original 100D vectors',
              'selection':'300 most frequent eligible words; English stopwords and documented generic terms removed',
              'k':8,'seed':42,'silhouette_cosine':float(silhouette_score(X,clustering.labels_,metric='cosine')),
              'clusters':clusters})
    # Select ten highest-frequency words from each cluster: at most 80 annotations.
    selected = np.array([i for c in range(8) for i in np.flatnonzero(clustering.labels_==c)[:10]])
    coords = TSNE(n_components=2,perplexity=20,init='pca',learning_rate='auto',
                  random_state=42,max_iter=1500).fit_transform(X[selected])
    fig, ax = plt.subplots(figsize=(12,8))
    palette = plt.get_cmap('tab10')
    ax.scatter(coords[:,0],coords[:,1],color=palette(clustering.labels_[selected]),s=35)
    for pos,xy in zip(selected,coords):
        ax.annotate(words[candidates[pos]],xy,xytext=(3,3),textcoords='offset points',fontsize=7)
    ax.margins(.10)
    for c in range(8):
        ax.scatter([],[],color=palette(c),s=25,label=str(c))
    ax.legend(title='Cluster',ncol=8,loc='upper center',fontsize=8)
    ax.set_title('t-SNE of trained word vectors; colors = K-means in original space')
    ax.set_xticks([]);ax.set_yticks([]);fig.tight_layout()
    fig.savefig(out/'tsne_clusters.png',dpi=200);plt.close(fig)
    save_json(out/'tsne_points.json',[{'word':words[candidates[pos]],'cluster':int(clustering.labels_[pos]),
              'x':float(xy[0]),'y':float(xy[1])} for pos,xy in zip(selected,coords)])
    docs = json.loads((out/'documents.json').read_text())
    token_docs = [tokenize(d['text']) for d in docs]
    sparse, sparse_vocab = bm25_matrix(token_docs)
    dense, encode, dense_vocab = document_vectors(token_docs,words,vectors)
    query_data = json.loads(args.queries.read_text())
    records = []
    for query in query_data['queries']:
        if query['target'] not in {d['id'] for d in docs}:
            raise ValueError('Target missing from corpus')
        tokens = tokenize(query['text'])
        qcols = [sparse_vocab[w] for w in set(tokens) if w in sparse_vocab]
        sparse_scores = np.asarray(sparse[:,qcols].sum(1)).ravel()
        qvec = encode(tokens)
        dense_scores = dense@qvec
        # A zero/OOV query produces no results, rather than arbitrary ties.
        scores = {'BM25':sparse_scores,'Word2Vec-TFIDF':dense_scores}
        record = dict(query)
        record['oov_tokens'] = sorted(set(tokens)-dense_vocab.keys())
        for name, values in scores.items():
            valid = bool(qcols) if name=='BM25' else bool(np.linalg.norm(qvec)>0)
            indices = top_indices(values) if valid else []
            record[name] = [{'id':docs[i]['id'],'title':docs[i]['title'],'score':float(values[i])} for i in indices]
        records.append(record)
    metrics = {}
    for group in ['all','lexical','paraphrase']:
        subset = [r for r in records if group=='all' or r['type']==group]
        metrics[group] = {name:retrieval_metrics([[d['id'] for d in r[name]] for r in subset],
                          [r['target'] for r in subset]) for name in ['BM25','Word2Vec-TFIDF']}
    save_json(out/'retrieval.json',{'task':query_data['task'],'metrics':metrics,'queries':records,
              'notes':'Same title+abstract corpus and tokenizer; BM25 keeps rare words; SGNS min_count=5. No supervised query training. One known target per query; target recall equals Hit@5.'})
    # Export a reproducible small sentence example.
    example = tokenize('Gene expression controls cell growth.')
    save_json(out/'sample_pairs.json',{'sentence':'Gene expression controls cell growth.', 'tokens':example,
              'window':2,'pairs':[[example[i],example[j]] for i,j in positive_pairs(list(range(len(example))),2)]})
    print(json.dumps({'neighbors':neighbors,'retrieval':metrics,'clusters':clusters},ensure_ascii=False),flush=True)

if __name__=='__main__':
    main()
