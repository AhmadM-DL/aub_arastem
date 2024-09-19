import argparse, os, json
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils import PromptFactory
import torch
import numpy as np
from accelerate import Accelerator


CHECKPOINTS_DIR = "./checkpoints/"

def setup():
    # Check Huggingface Token
    if os.path.exists(".env"):
        token = [l.split("=")[1] for l in open(".env", "r").readlines()][0]
        os.environ["HF_TOKEN"] = token
    else:
        raise Exception("Please add an .env file that includes huggingface HF_TOKEN")
    # Check checkpoint directory
    if not os.path.exists(CHECKPOINTS_DIR):
        os.mkdir(CHECKPOINTS_DIR)
    

def load_data(data_path):
    data = json.load(open(data_path))
    return data

def load_model(model_identifier, accelerator):
    tokenizer = AutoTokenizer.from_pretrained(model_identifier)
    model = AutoModelForCausalLM.from_pretrained(model_identifier, trust_remote_code=True, device_map="auto")
    model = accelerator.prepare(model)
    return model, tokenizer

def load_checkpoint(model_identifier):
    model_identifier = model_identifier.replace("/", "-")
    filepath = os.path.join(CHECKPOINTS_DIR, f"{model_identifier}.output")
    if os.path.exists(filepath):
        output = json.load(open(filepath))
    else:
        output = []
    return output

def save_checkpoint(model_identifier, output):
    model_identifier = model_identifier.replace("/", "-")
    filepath = os.path.join(CHECKPOINTS_DIR, f"{model_identifier}.output")
    json.dump(output, open(filepath, "w"), ensure_ascii=False, indent=4)

def softmax(x):
    z = x - max(x)
    numerator = np.exp(z)
    denominator = np.sum(numerator)
    softmax = numerator/denominator
    return softmax

def main():
    parser = argparse.ArgumentParser(description="A script that loads a model and compute inference on a dataset.")
    parser.add_argument('-m', '--model', type=str, required=True, help="Huggingface model identifier")
    parser.add_argument('-d', '--data', type=str, required=True, help="MCQ data to infere on")
    parser.add_argument('-s', '--max_input_token', type=int, required=True, help="Max input tokens a model can consume")
    parser.add_argument('-v', '--verbose', action='store_true', help="Verbose")
    args = parser.parse_args()

    accelerator = Accelerator()
    device = accelerator.device

 
    if args.verbose: print("Setup ...")
    setup()

    if args.verbose: print("Load data ...")
    data = load_data(args.data)
    
    if args.verbose: print("Load model ...")
    model, tokenizer = load_model(args.model, accelerator)
    
    prompt_factory = PromptFactory()
    prompt_generator = prompt_factory.get_prompt_function(n_shots=0)

    if args.verbose: print("Load Checkpoint ...")
    output = load_checkpoint(args.model)
    checkpoint = output[-1]["id"] if output else -1

    if args.verbose: print("Running Inference ...")
    for i, q in enumerate(data[checkpoint+1:]):
        id = i + checkpoint + 1
        question = q["question"]
        options = [q[f"option_{i}"] for i in [0, 1, 2, 3] if f"option_{i}" in q and q[f"option_{i}"] != "" ]
        prompt = prompt_generator(question, options, subject=q["subject"], level=q["level"])
        options_tokens = [tokenizer.encode(choice)[-1] for choice in ["0", "1", "2", "3", "4"]]
        with torch.no_grad():
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            input_size = inputs['input_ids'].size(1)
            if input_size > args.max_input_token:
                raise ValueError(f"Input {id} is too long! The tokenized input has {input_size} tokens, which exceeds the maximum allowed size of {args.max_input_token} tokens.")
            #input_ids = inputs["input_ids"]
            outputs = model(**inputs)#, labels=input_ids)
            last_token_logits = outputs.logits[:, -1, :]
            options_tokens_logits = last_token_logits[:, options_tokens].detach().cpu().numpy()
            conf = softmax(options_tokens_logits[0])
            pred = np.argmax(options_tokens_logits[0])
        output.append({"id": id, "prediction": pred, "confidence": conf[pred]})
        save_checkpoint(args.model, output)
        if id>0 and id%100==0:
            if args.verbose: 
                print(f"Processed {id} questions")

if __name__ == '__main__':
    main()
