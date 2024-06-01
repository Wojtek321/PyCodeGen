from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from arguments import args


tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

base_model = AutoModelForCausalLM.from_pretrained(args.pretrained_model_name)
model = PeftModel.from_pretrained(base_model, args.model_name)

merged_model = model.merge_and_unload()

merged_model.save_pretrained(args.model_name, push_to_hub=True)
tokenizer.save_pretrained(args.model_name, push_to_hub=True)
