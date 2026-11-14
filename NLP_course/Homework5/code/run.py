"""BERT sentiment fine-tuning, frozen-encoder baseline, evaluation and plots."""
import argparse
import json
import os
from pathlib import Path
import random
import subprocess
import threading
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[1]
TEMP = ROOT / 'temp' / 'results'
MODEL = 'google-bert/bert-base-chinese'
REV = '8f23c25b06e129b6c986331a13d8d025a92cf0ea'
DATA = 'lansinuote/ChnSentiCorp'
DATA_REV = 'b0c4c119c3fb33b8e735969202ef9ad13d717e5a'
SEED = 42
CUSTOM = ['这家酒店很干净，服务也很好，下次还会来。', '这家酒店不干净，服务也不好，下次不会来了。',
          '这本书很好，内容清楚，值得购买。', '这本书并不好，内容混乱，不值得购买。',
          '电脑外观漂亮，但是经常死机，实在失望。', '电脑外观普通，但是运行稳定，用起来很满意。',
          '不能说这家酒店不好，只是价格有点高。', '本来以为会很差，没想到用起来这么好。']

def save(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')

def seed():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

def setup():
    torch.set_num_threads(4)
    seed()
    if not torch.cuda.is_available(): raise RuntimeError('This measured experiment requires CUDA.')
    torch.backends.cuda.matmul.allow_tf32 = True

class MemoryMonitor:
    """Poll this PID only; allocator peaks are also measured separately."""
    def __init__(self):
        self.stop_event = threading.Event(); self.samples = []; self.start = time.perf_counter()
    def poll(self):
        while not self.stop_event.is_set():
            try:
                out = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,used_gpu_memory',
                    '--format=csv,noheader,nounits'], text=True, timeout=3)
                values = [int(s.split(',')[1].strip()) for s in out.splitlines()
                          if s.split(',')[0].strip() == str(os.getpid())]
                if values: self.samples.append({'seconds': time.perf_counter()-self.start, 'MiB': sum(values)})
            except (OSError, ValueError, subprocess.SubprocessError): pass
            self.stop_event.wait(1)
    def __enter__(self):
        self.thread = threading.Thread(target=self.poll, daemon=True); self.thread.start(); return self
    def __exit__(self, *args):
        self.stop_event.set(); self.thread.join()
    def peak(self): return max((s['MiB'] for s in self.samples), default=None)

def tokenizer(): return AutoTokenizer.from_pretrained(MODEL, revision=REV)

def dataset(): return load_dataset(DATA, revision=DATA_REV)

def encode(ds, tok):
    return ds.map(lambda x: tok(x['text'], truncation=True, max_length=128), batched=True,
                  remove_columns=['text'], desc='Tokenizing').rename_column('label', 'labels')

def loader(ds, tok, shuffle=False):
    return DataLoader(ds, batch_size=8, shuffle=shuffle, collate_fn=DataCollatorWithPadding(tok),
                      generator=torch.Generator().manual_seed(SEED))

def amp(): return torch.autocast('cuda', dtype=torch.bfloat16)

def evaluate(model, dl, features=False):
    model.eval(); predictions=[]; targets=[]; probabilities=[]; latent=[]; loss_sum=0; count=0
    with torch.inference_mode():
        for batch in dl:
            batch={k:v.cuda() for k,v in batch.items()}
            with amp():
                # Pooler uses CLS -> Linear -> tanh, matching BertForSequenceClassification.
                enc=model.bert(**{k:v for k,v in batch.items() if k!='labels'}, return_dict=True)
                logits=model.classifier(model.dropout(enc.pooler_output))
                loss=torch.nn.functional.cross_entropy(logits, batch['labels'])
            n=len(batch['labels']); count+=n; loss_sum+=loss.item()*n
            predictions.extend(logits.argmax(-1).cpu().tolist()); targets.extend(batch['labels'].cpu().tolist())
            probabilities.extend(logits.float().softmax(-1).cpu().tolist())
            if features: latent.append(enc.pooler_output.float().cpu().numpy())
    return {'loss':loss_sum/count, 'accuracy':accuracy_score(targets,predictions),
            'macro_f1':f1_score(targets,predictions,average='macro'),
            'report':classification_report(targets,predictions,labels=[0,1],target_names=['negative','positive'],
                                           output_dict=True,zero_division=0),
            'confusion_matrix':confusion_matrix(targets,predictions,labels=[0,1]).tolist()}, predictions, probabilities, (np.concatenate(latent) if features else None)

def prepare():
    tok=tokenizer(); ds=dataset(); stats={}
    fig,ax=plt.subplots(figsize=(7,4))
    for name, split in ds.items():
        lengths=np.array([len(x) for x in tok(list(split['text']), truncation=False, verbose=False)['input_ids']])
        stats[name]={'count':len(split),'negative':split['label'].count(0),'positive':split['label'].count(1),
            'token_length_includes_special_tokens':True,'mean_length':float(lengths.mean()),
            'min':int(lengths.min()),'median':float(np.median(lengths)),'p90':float(np.percentile(lengths,90)),
            'p95':float(np.percentile(lengths,95)),'max':int(lengths.max()),
            'truncated_count':int((lengths>128).sum()),'truncated_fraction':float((lengths>128).mean())}
        np.save(TEMP/f'{name}_lengths.npy',lengths)
        ax.hist(lengths,bins=np.arange(0,2101,50),alpha=.4,label=name,density=True)
    ax.set(xlabel='Tokens including CLS and SEP',ylabel='Density',title='Token length distribution')
    ax.axvline(128,color='black',linestyle='--',label='max length 128');ax.legend();fig.tight_layout()
    fig.savefig(ROOT/'results/length_distribution.png',dpi=180);plt.close(fig)
    sample='酒店很干净，服务不错。'; e=tok(sample,max_length=24,padding='max_length',truncation=True)
    save(ROOT/'results/data_statistics.json',stats)
    save(ROOT/'results/tokenization_example.json',{'text':sample,**e,'tokens':tok.convert_ids_to_tokens(e['input_ids'])})
    # Download the base weights once. Classifier is initialized anew by each training process.
    AutoModelForSequenceClassification.from_pretrained(MODEL,revision=REV,num_labels=2)
    print(json.dumps(stats,ensure_ascii=False),flush=True)

def train(mode):
    setup(); directory=ROOT/'results'/mode;directory.mkdir(parents=True,exist_ok=True)
    work=TEMP/mode;work.mkdir(parents=True,exist_ok=True)
    tok=tokenizer(); ds=dataset(); encoded=encode(ds['train'],tok); valid=encode(ds['validation'],tok)
    model=AutoModelForSequenceClassification.from_pretrained(MODEL,revision=REV,num_labels=2,
            id2label={0:'negative',1:'positive'},label2id={'negative':0,'positive':1}).cuda()
    if mode=='frozen':
        for param in model.bert.parameters():param.requires_grad=False
    # Same hyperparameters; baseline isolates encoder updates, not individually optimized performance.
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=2e-5,weight_decay=.01)
    dl=loader(encoded,tok,True);vd=loader(valid,tok)
    config={'model':MODEL,'model_revision':REV,'dataset':DATA,'dataset_revision':DATA_REV,'seed':SEED,
        'mode':mode,'batch_size':8,'max_length':128,'epochs':3,'lr':2e-5,'weight_decay':.01,
        'optimizer':'AdamW','precision':'bf16 autocast, fp32 parameters','gradient_clip_norm':1.,
        'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,
        'parameters':sum(p.numel() for p in model.parameters()),
        'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
        'frozen_encoder_dropout':'disabled' if mode=='frozen' else 'training dropout enabled'}
    save(directory/'config.json',config)
    history=[];best=-1;global_step=0; torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
    overall=time.perf_counter(); train_seconds=0.;val_seconds=0.;checkpoint_seconds=0.
    with MemoryMonitor() as monitor, (work/'steps.jsonl').open('w') as log:
        for epoch in range(1,4):
            model.train()
            if mode=='frozen':model.bert.eval()
            torch.cuda.synchronize();start=time.perf_counter();loss_sum=0;count=0
            for step,batch in enumerate(dl,1):
                batch={k:v.cuda() for k,v in batch.items()};optimizer.zero_grad(set_to_none=True)
                with amp(): output=model(**batch);loss=output.loss
                loss.backward();torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.)
                optimizer.step();n=len(batch['labels']);loss_sum+=loss.item()*n;count+=n;global_step+=1
                if step%100==0 or step==len(dl):
                    row={'epoch':epoch,'step':global_step,'mean_train_loss':loss_sum/count,
                         'epoch_elapsed_seconds':time.perf_counter()-start,
                         'max_allocated_MiB':torch.cuda.max_memory_allocated()/2**20,
                         'max_reserved_MiB':torch.cuda.max_memory_reserved()/2**20,
                         'nvidia_process_peak_MiB':monitor.peak()}
                    log.write(json.dumps(row)+'\n');log.flush();print(json.dumps(row),flush=True)
            torch.cuda.synchronize();duration=time.perf_counter()-start;train_seconds+=duration
            start=time.perf_counter();metrics,*_=evaluate(model,vd);torch.cuda.synchronize()
            val_duration=time.perf_counter()-start;val_seconds+=val_duration
            row={'epoch':epoch,'train_loss':loss_sum/count,'validation_loss':metrics['loss'],
                'validation_macro_f1':metrics['macro_f1'],'validation_accuracy':metrics['accuracy'],
                'train_seconds':duration,'validation_seconds':val_duration}
            history.append(row);save(directory/'history.json',history)
            if metrics['macro_f1']>best:
                best=metrics['macro_f1'];best_epoch=epoch;start=time.perf_counter()
                model.save_pretrained(TEMP/mode/'checkpoint');tok.save_pretrained(TEMP/mode/'checkpoint')
                checkpoint_seconds+=time.perf_counter()-start
            print('EPOCH_RESULT '+json.dumps(row),flush=True)
    summary={'train_seconds':train_seconds,'validation_seconds':val_seconds,'checkpoint_seconds':checkpoint_seconds,
        'train_loop_wall_seconds':time.perf_counter()-overall,'best_epoch':best_epoch,'best_validation_macro_f1':best,
        'max_allocated_MiB':torch.cuda.max_memory_allocated()/2**20,
        'max_reserved_MiB':torch.cuda.max_memory_reserved()/2**20,'nvidia_process_peak_MiB':monitor.peak(),
        'memory_scope':'training and validation; nvidia-smi per-process sampled every 1s',
        'timing_scope':'data/model loading excluded; shared GPU load affects elapsed time'}
    save(directory/'resources.json',summary);save(work/'gpu_samples.json',monitor.samples)
    print('FINISHED '+json.dumps(summary),flush=True)

def assess():
    setup();tok=tokenizer();ds=dataset();test=encode(ds['test'],tok)
    for mode in ['full','frozen']:
        directory=ROOT/'results'/mode
        model=AutoModelForSequenceClassification.from_pretrained(TEMP/mode/'checkpoint').cuda().eval()
        start=time.perf_counter();metrics,pred,probs,latent=evaluate(model,loader(test,tok),features=True)
        torch.cuda.synchronize();metrics['test_seconds']=time.perf_counter()-start
        save(directory/'test_metrics.json',metrics)
        rows=[{'id':i,'text':r['text'],'label':r['label'],'prediction':pred[i],
               'probabilities':probs[i]} for i,r in enumerate(ds['test'])]
        with (directory/'test_predictions.jsonl').open('w') as f:
            for row in rows:f.write(json.dumps(row,ensure_ascii=False)+'\n')
        sample=random.Random(SEED).sample(range(len(rows)),20)
        save(directory/'random20.json',[rows[i] for i in sample])
        save(directory/'errors.json',[r for r in rows if r['label']!=r['prediction']])
        custom=tok(CUSTOM,padding=True,truncation=True,max_length=128,return_tensors='pt').to('cuda')
        with torch.inference_mode(),amp():logits=model(**custom).logits
        cprobs=logits.float().softmax(-1).cpu().tolist()
        save(directory/'custom_predictions.json',[{'text':t,'prediction':int(np.argmax(pr)),
             'probabilities':pr} for t,pr in zip(CUSTOM,cprobs)])
        # A separately loaded model must give the same labels/probabilities for a fixed batch.
        reference=logits.float().cpu();del model;torch.cuda.empty_cache()
        reloaded=AutoModelForSequenceClassification.from_pretrained(TEMP/mode/'checkpoint').cuda().eval()
        with torch.inference_mode(),amp():check=reloaded(**custom).logits.float().cpu()
        save(directory/'reload_check.json',{'same_predictions':bool(torch.equal(reference.argmax(-1),check.argmax(-1))),
            'max_logit_difference':float((reference-check).abs().max())})
        del reloaded;torch.cuda.empty_cache()
        fig,ax=plt.subplots(figsize=(4,4));cm=np.array(metrics['confusion_matrix']);ax.imshow(cm,cmap='Blues')
        for (i,j),v in np.ndenumerate(cm):ax.text(j,i,str(v),ha='center',va='center')
        ax.set(xticks=[0,1],yticks=[0,1],xticklabels=['Negative','Positive'],yticklabels=['Negative','Positive'],
               xlabel='Predicted label',ylabel='True label',title=mode+' confusion matrix')
        fig.tight_layout();fig.savefig(directory/'confusion_matrix.png',dpi=180);plt.close(fig)
        np.save(TEMP/mode/'pooled_representations.npy',latent)
        pca=PCA(n_components=2,random_state=SEED);xy=pca.fit_transform(latent);labels=np.array(ds['test']['label'])
        fig,ax=plt.subplots(figsize=(6,4))
        for label,color in [(0,'#3574b9'),(1,'#df8627')]:
            mask=labels==label;ax.scatter(xy[mask,0],xy[mask,1],s=10,alpha=.45,c=color,label=['Negative','Positive'][label])
        wrong=labels!=np.array(pred);ax.scatter(xy[wrong,0],xy[wrong,1],s=24,c='black',marker='x',label='Misclassified')
        ax.set(xlabel='PC1',ylabel='PC2',title=mode+' pooled BERT representations');ax.legend(fontsize=8)
        fig.tight_layout();fig.savefig(directory/'latent_pca.png',dpi=180);plt.close(fig)
        save(directory/'pca.json',{'representation':'BERT pooler output: tanh(W * CLS + b)',
                                 'explained_variance_ratio':pca.explained_variance_ratio_.tolist()})
        h=json.loads((directory/'history.json').read_text());fig,ax=plt.subplots(figsize=(5,3))
        ax.plot([x['epoch'] for x in h],[x['train_loss'] for x in h],'o-',label='Train')
        ax.plot([x['epoch'] for x in h],[x['validation_loss'] for x in h],'o-',label='Validation')
        ax.set(xlabel='Epoch',ylabel='Cross-entropy',xticks=[1,2,3],title=mode);ax.legend();fig.tight_layout()
        fig.savefig(directory/'loss_curve.png',dpi=180);plt.close(fig)
        print(mode,json.dumps(metrics),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','full','frozen','evaluate'])
    args=parser.parse_args();(ROOT/'results').mkdir(exist_ok=True);TEMP.mkdir(parents=True,exist_ok=True)
    if args.stage=='prepare':prepare()
    elif args.stage=='evaluate':assess()
    else:train(args.stage)
