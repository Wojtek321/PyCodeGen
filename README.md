# PyCodeGen 350M

This repository contains the script used to finetune [codegen-350M-mono](https://huggingface.co/Salesforce/codegen-350M-mono) on python code [dataset](https://huggingface.co/datasets/iamtarun/python_code_instructions_18k_alpaca) using QLORA method.
Finetuned model can be found [here](https://huggingface.co/chincyk/PyCodeGen).

## Pretrained model description

[codegen-350M-mono](https://huggingface.co/Salesforce/codegen-350M-mono)

Codegen-350M-mono comes from the family of autoregressive models for program synthesis developed by Salesforce. 
This model was first trained on ThePile dataset which is 825.18 GiB English text corpus.
It was then adapted to generate code by training on a set of GitQuery with source codes.
Finally model has been adapted to the Python language by training on the BigPython dataset.

## Training Data

[python_code_instructions_18k_alpaca](https://huggingface.co/datasets/iamtarun/python_code_instructions_18k_alpaca)

The dataset contains problem descriptions and code in python language. 
This dataset is taken from [iamtarun/code_instructions_120k_alpaca](https://huggingface.co/datasets/iamtarun/code_instructions_120k_alpaca).

## Intended uses

The model can be used to generate python code that solves task with optionally given input data.

## Example of usage

```py
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained('chincyk/PyCodeGen')
tokenizer = AutoTokenizer.from_pretrained('chincyk/PyCodeGen')

instruction = "Write a python class that represents a calculator, then use it to add two numbers."
input = "a = 5, b = 2"

prompt = f"""
    ### Instruction:
    Use the Task below and the Input given to write the Response, which is a programming code that can solve the Task.

    ### Task:
    {instruction}

    ### Input:
    {input}
    
    ### Response:
    """

input_ids = tokenizer(prompt, truncation=True, return_tensors="pt")['input_ids']
output = model.generate(input_ids=input_ids, max_length=200)

print(tokenizer.decode(output[0], skip_special_tokens=True))
```

## Training parameters

BitsAndBytes:
- load_in_4bit: True,
- bnb_4bit_quant_type: nf4,
- bnb_4bit_use_double_quant: True,
- bnb_4bit_compute_dtype: torch.bfloat16

LoraConfig:
- r: 32,
- lora_alpha: 16,
- target_modules: all-linear,
- lora_dropout: 0.1,
- bias: none,
- task_type: CASUAL_LM

Finetuning:
- num_epochs: 15
- train_batch_size: 4
- eval_batch_size: 8
- gradient_accumulation_steps: 8
- learning_rate: 3e-4
- weight_decay: 0.01
- lr_scheduler_name: cosine
- num_warmup_steps: 190
