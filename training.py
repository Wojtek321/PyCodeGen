from transformers import AutoModelForCausalLM, get_scheduler, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.tensorboard import SummaryWriter
from torch.optim import AdamW
import torch
from data import train_dataloader, valid_dataloader
from arguments import args
from tqdm import tqdm
import time


def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for param in model.parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    print(f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}")


nf4_config = BitsAndBytesConfig(
   load_in_4bit=True,
   bnb_4bit_quant_type="nf4",
   bnb_4bit_use_double_quant=True,
   bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(args.pretrained_model_name, quantization_config=nf4_config)
model = prepare_model_for_kbit_training(model)

for name, param in model.named_parameters():
    param.requires_grad = False


config = LoraConfig(
    r=32,
    lora_alpha=16,
    target_modules='all-linear',
    lora_dropout=0.1,
    bias='none',
    task_type='CASUAL_LM'
)

model = get_peft_model(model, config)
print_trainable_parameters(model)


optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

num_training_steps = int(args.num_epochs * len(train_dataloader) / args.gradient_accumulation_steps)
lr_scheduler = get_scheduler(args.lr_scheduler_name, optimizer=optimizer, num_warmup_steps=args.num_warmup_steps, num_training_steps=num_training_steps)


writer = SummaryWriter()

for epoch in range(args.num_epochs):

    # training
    optimizer.zero_grad()
    bar = tqdm(total=len(train_dataloader), desc=f'Epoch {epoch + 1}/{args.num_epochs}', unit='batch')
    avg_loss = 0
    model.train()
    start_time = time.time()


    for step, batch in enumerate(train_dataloader, start=1):
        input_ids = batch['input_ids'].to(args.device)
        attention_mask = batch['attention_mask'].to(args.device)

        output = model(input_ids, attention_mask=attention_mask, labels=input_ids, use_cache=False)
        loss = output.loss
        loss = loss / args.gradient_accumulation_steps
        avg_loss += loss.item()
        loss.backward()

        if step % args.gradient_accumulation_steps == 0:
            writer.add_scalar("Train/loss", avg_loss, (epoch*len(train_dataloader) + step))
            writer.add_scalar("Train/lr", optimizer.param_groups[0]['lr'], (epoch*len(train_dataloader) + step))

            optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step()

            bar.set_postfix({'loss': avg_loss, 'lr': optimizer.param_groups[0]['lr']})
            avg_loss = 0
            start_time = time.time()

        bar.update(1)

    bar.close()


    # evaluating
    bar = tqdm(total=len(valid_dataloader), desc=f'Evaluating', unit='batch')
    model.eval()

    for step, batch in enumerate(valid_dataloader, start=1):
        input_ids = batch['input_ids'].to(args.device)
        attention_mask = batch['attention_mask'].to(args.device)

        with torch.no_grad():
            output = model(input_ids, attention_mask=attention_mask, labels=input_ids, use_cache=False)

        loss = output.loss
        ppl = torch.exp(loss)

        loss = loss.item()
        ppl = ppl.item()

        writer.add_scalar("Test/loss", loss, (epoch * len(valid_dataloader) + step))
        writer.add_scalar("Test/ppl", ppl, (epoch * len(valid_dataloader) + step))

        bar.set_postfix({'loss': loss, 'perplexity': ppl})
        bar.update(1)

    bar.close()

    model.save_pretrained(args.model_name)
