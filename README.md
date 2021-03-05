# TOEIC_Mask_Filling
Use GPT-2 to solve filling mask questions.

## Idea:
- Substitute 4 options into a question to make 4 complete sentences
- Estimate the perplexity of the 4 sentences 
- Choose the option that make up the sentence with the lowest perplexity

## Code

Install transformer:
```
pip install transformers
```

Initialize the model and the tokenizer:

```
model_name = 'gpt2-medium'
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
```

Create function to calculate perplexity:

```
import numpy as np
def score(sentence):
    tokens_tensor = tokenizer(sentence, return_tensors="pt")
    loss = model(**tokens_tensor, labels=tokens_tensor['input_ids'])[0]
    return loss.detach().numpy().item()
```

Create function to answer the question:

```
def use_gpt2(question,options):
  scores = [score(tokenizer(question.replace("[MASK]", o), return_tensors="pt")) for o in options]
  return options[np.argmin(scores)]
```

Usage examples:

- Filling words

```
option = ['suffer', 'suffers', 'suffering', 'suffered']
question = "The assets of Marble Faun Publishing Company [MASK] last quarter when one of their main local distributors went out of business."
answer = use_gpt2(question,option)
```

- Filling a sentence

```
def use_gpt2(question,options):
  scores = [score(tokenizer(question.replace("[MASK]", o), return_tensors="pt")) for o in options]
  return options[np.argmin(scores)]
```

## Results:



## References:

Solution:

- Using GPT-2 to output perplexity (https://stackoverflow.com/questions/63543006/how-can-i-find-the-probability-of-a-sentence-using-gpt-2)


Datasets:

- Filling words (https://www.kaggle.com/tientd95/bert-model-for-anwsering-toeic-reading-test) 

- Filling an entire sentence (https://toeic24.vn/)
