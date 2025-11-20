import json
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig

MODEL_NAME = "/data/kunfeng/models/Llama-3.1-8B-Instruct"
train_file = "/home/kfeng/python_code/Hello-World/P_DiseaseOntology/do_terms_train_23.json"
test_file = "/home/kfeng/python_code/Hello-World/P_DiseaseOntology/do_terms_test_24.json"

def build_prompt(example):
    """Convert JSON entry to LLM training prompt."""
    prompt = f"""Generate a precise and ontology-style disease definition.

ID: {example['id']}
Name: {example['name']}
Synonyms: {', '.join(example['synonyms']) if example['synonyms'] else 'None'}
Parents: {', '.join(example['parents']) if example['parents'] else 'None'}
Subsets: {', '.join(example['subsets']) if example['subsets'] else 'None'}
Xrefs: {', '.join([x['val'] for x in example['xrefs']]) if example['xrefs'] else 'None'}

Definition:"""

    return {"text": prompt + " " + example["definition"]}

train_data = Dataset.from_json(train_file).map(build_prompt)
test_data = Dataset.from_json(test_file).map(build_prompt)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    load_in_8bit=True,
    device_map="auto",
)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
)

training_config = SFTConfig(
    output_dir="llama-do-definition",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=2e-5,
    num_train_epochs=3,
    logging_steps=20,
    save_steps=200,
    fp16=True,
    dataset_text_field="text",
    max_length=1024,
    packing=False,
)

trainer = SFTTrainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=test_data,
    processing_class=tokenizer,
    peft_config=lora_config,
    args=training_config,
)

trainer.train()
trainer.save_model("llama-do-definition-lora")
