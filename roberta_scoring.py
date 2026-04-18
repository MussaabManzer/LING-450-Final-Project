from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

import transformers
print(transformers.__version__)

MODEL_NAME = "coastalcph/roberta-large-ft-trump-populism"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval()

# Check 1: are the classifier weights non-trivial?
print("Classifier weight sample:")
print(model.classifier.out_proj.weight.data[:3, :5])
print("Classifier bias:")
print(model.classifier.out_proj.bias.data)

# Check 2: do two very different sentences produce different logits?
sents = [
    "The corrupt elite have betrayed the American people.",
    "The clerk will report the amendment."
]
for s in sents:
    enc = tokenizer(s, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        out = model(**enc)
    print(f"\nText: {s[:60]}")
    print(f"Logits: {out.logits}")
    print(f"Probs:  {F.softmax(out.logits, dim=-1)}")
