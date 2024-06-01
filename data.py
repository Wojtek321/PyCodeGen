from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from transformers import AutoTokenizer, DataCollatorWithPadding
import matplotlib.pyplot as plt
from datasets import load_dataset
from arguments import args
import seaborn as sns
import numpy as np


class PreparedDataset(Dataset):
    def __init__(self, dataset, tokenizer):
        self.tokenizer = tokenizer
        self.dataset = dataset

        self.dataset = self.dataset.map(self.tokenize_function, batched=True)
        self.dataset = self.dataset.remove_columns(['instruction', 'input', 'output', 'prompt'])
        self.dataset.set_format("torch")

    def tokenize_function(self, sample):
        return self.tokenizer(sample['prompt'], truncation=True)

    def __len__(self):
        return self.dataset.num_rows

    def __getitem__(self, idx):
        return self.dataset[idx]


def create_prompt(samples):
    prompts = []
    for instruction, input_text, output in zip(samples['instruction'], samples['input'], samples['output']):
        prompt = f"""
        ### Instruction:
        Use the Task below and the Input given to write the Response, which is a programming code that can solve the Task.,

        ### Task:
        {instruction}

        ### Input:
        {input_text}

        ### Response:
        {output}
        """
        prompts.append(prompt)

    return {'prompt': prompts}


tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
tokenizer.pad_token = tokenizer.eos_token


dataset = load_dataset(args.dataset_name, split='train')

dataset = dataset.map(create_prompt, batched=True)
dataset = dataset.train_test_split(test_size=0.15)

train_dataset = dataset['train']
valid_dataset = dataset['test']

prepared_train_dataset = PreparedDataset(train_dataset, tokenizer)
prepared_eval_dataset = PreparedDataset(valid_dataset, tokenizer)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

train_dataloader = DataLoader(prepared_train_dataset, batch_size=args.train_batch_size, collate_fn=data_collator, pin_memory=True, shuffle=True)
valid_dataloader = DataLoader(prepared_eval_dataset, batch_size=args.eval_batch_size, collate_fn=data_collator, pin_memory=True)


if __name__ == '__main__':
    lengths = []

    for input in train_dataset:
        tokens = tokenizer(input['prompt'], truncation=False)['input_ids']
        lengths.append(len(tokens))

    for input in valid_dataset:
        tokens = tokenizer(input['prompt'], truncation=False)['input_ids']
        lengths.append(len(tokens))

    lengths = np.array(lengths)

    plt.figure(figsize=(9, 7))
    ax = sns.histplot(lengths, bins=75)
    ax.set(xlabel='Token quantity', title='Input data length distribution')
    plt.savefig('assets/Data_length_distribution.png')
