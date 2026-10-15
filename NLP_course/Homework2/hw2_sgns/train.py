"""Train on titles and abstracts; paths default relative to this script."""
import argparse
import hashlib
import json
import os
import platform
import time
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
from sgns import SGNS, NegativeSampler, tokenize, build_vocab, pair_array

ROOT = Path(__file__).resolve().parent


def save_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--corpus', type=Path, default=ROOT.parent.parent/'Homework1/datasets/gene_related_papers_1000')
    p.add_argument('--output', type=Path, default=ROOT/'outputs')
    p.add_argument('--epochs', type=int, default=12)
    p.add_argument('--dim', type=int, default=100)
    p.add_argument('--window', type=int, default=5)
    p.add_argument('--negatives', type=int, default=5)
    p.add_argument('--min-count', type=int, default=5)
    p.add_argument('--batch-size', type=int, default=4096)
    p.add_argument('--lr', type=float, default=.003)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()
    if min(args.epochs,args.dim,args.window,args.negatives,args.min_count,args.batch_size) < 1:
        p.error('Integer hyperparameters must be positive.')
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.use_deterministic_algorithms(True)
    rng = np.random.default_rng(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    files = sorted(args.corpus.glob('*.json'))
    if not files:
        raise FileNotFoundError(args.corpus)
    documents, token_docs, manifest = [], [], []
    for path in files:
        raw = path.read_bytes()
        data = json.loads(raw)
        text = '\n'.join(data.get(k, '') or '' for k in ('article_title','article_abstract'))
        documents.append({'id':path.stem, 'title':data['article_title'], 'text':text})
        token_docs.append(tokenize(text))
        manifest.append({'file':path.name, 'sha256':hashlib.sha256(raw).hexdigest()})
    words, counts = build_vocab(token_docs, args.min_count)
    word_to_id = {w:i for i,w in enumerate(words)}
    encoded = [np.array([word_to_id.get(w,-1) for w in doc],dtype=np.int64) for doc in token_docs]
    pairs = pair_array(encoded, args.window)
    model = SGNS(len(words), args.dim).to(args.device)
    sampler = NegativeSampler(counts, args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # Same small probe pairs and negatives each epoch: directly comparable loss.
    probe_indices = rng.choice(len(pairs), min(4096,len(pairs)), replace=False)
    probe = torch.tensor(pairs[probe_indices], device=args.device)
    probe_neg = sampler.sample(probe[:,1], args.negatives)
    def probe_loss():
        with torch.no_grad():
            return model(probe[:,0], probe[:,1], probe_neg).item()
    history = [{'epoch':0, 'probe_loss':probe_loss()}]
    train_started = time.perf_counter()
    for epoch in range(1,args.epochs+1):
        epoch_started = time.perf_counter()
        total = 0.
        for start in range(0,len(pairs),args.batch_size):
            if start == 0:
                order = rng.permutation(len(pairs))
            batch = torch.tensor(pairs[order[start:start+args.batch_size]], device=args.device)
            neg = sampler.sample(batch[:,1], args.negatives)
            optimizer.zero_grad(set_to_none=True)
            loss = model(batch[:,0],batch[:,1],neg)
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite loss')
            loss.backward()
            optimizer.step()
            total += loss.item()*len(batch)
        row = {'epoch':epoch, 'train_loss':total/len(pairs), 'probe_loss':probe_loss(),
               'seconds':time.perf_counter()-epoch_started}
        history.append(row)
        print(json.dumps(row),flush=True)
    training_seconds = time.perf_counter()-train_started
    model.cpu()
    torch.save(model.state_dict(), args.output/'model.pt')
    np.save(args.output/'vectors.npy',model.center.weight.detach().numpy())
    save_json(args.output/'vocab.json',{'words':words,'counts':counts,'word_to_id':word_to_id})
    config = {k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    config.update(fields=['article_title','article_abstract'],activation='sigmoid (logsigmoid in loss)',
                  exported_matrix='center',negative_exponent=.75,subsampling=False,
                  conflict_policy='redraw positive context; duplicates and center word allowed')
    save_json(args.output/'config.json',config)
    save_json(args.output/'history.json',history)
    save_json(args.output/'documents.json',documents)
    save_json(args.output/'manifest.json',manifest)
    stats = {'documents':len(files), 'tokens':sum(map(len,token_docs)),
             'vocab_size':len(words),'retained_tokens':sum(counts),'pairs_per_epoch':len(pairs),
             'training_seconds':training_seconds,'total_seconds':time.perf_counter()-started,
             'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,
             'device':torch.cuda.get_device_name(0) if args.device.startswith('cuda') else platform.processor(),
             'missing_abstracts':sum(not json.loads(f.read_text()).get('article_abstract') for f in files)}
    save_json(args.output/'stats.json',stats)
    print(json.dumps(stats,ensure_ascii=False),flush=True)

if __name__ == '__main__':
    main()
