"""Load a saved sentiment classifier and predict supplied Chinese comments."""
import argparse
import json
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

parser=argparse.ArgumentParser()
parser.add_argument('texts',nargs='+',help='One or more quoted comments')
parser.add_argument('--checkpoint',default=str(Path(__file__).resolve().parents[1]/'temp/results/full/checkpoint'))
parser.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
args=parser.parse_args()
tokenizer=AutoTokenizer.from_pretrained(args.checkpoint)
model=AutoModelForSequenceClassification.from_pretrained(args.checkpoint).to(args.device).eval()
inputs=tokenizer(args.texts,padding=True,truncation=True,max_length=128,return_tensors='pt').to(args.device)
with torch.inference_mode():
    probs=model(**inputs).logits.softmax(-1).cpu().tolist()
for text,prob in zip(args.texts,probs):
    print(json.dumps({'text':text,'prediction':int(prob[1]>prob[0]),'probabilities':prob},ensure_ascii=False))
