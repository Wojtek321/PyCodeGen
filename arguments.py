from dataclasses import dataclass
import torch


@dataclass
class Arguments:
    pretrained_model_name = "Salesforce/codegen-350M-mono"
    tokenizer = "Salesforce/codegen-350M-mono"
    dataset_name = "iamtarun/python_code_instructions_18k_alpaca"

    model_name = "PyCodeGen"

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    num_epochs = 15
    train_batch_size = 4
    eval_batch_size = 8
    gradient_accumulation_steps = 8

    learning_rate = 3e-4
    weight_decay = 0.01
    lr_scheduler_name = "cosine"
    num_warmup_steps = 190


args = Arguments()