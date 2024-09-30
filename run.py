import argparse, os, json
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from utils_prompt import PromptFactory
import torch
import numpy as np
from accelerate import Accelerator


CHECKPOINTS_DIR = "checkpoints"

def setup():
    # Check Huggingface Token
    if os.path.exists(".env"):
        token = [l.split("=")[1] for l in open(".env", "r").readlines()][0]
        os.environ["HF_TOKEN"] = token
    else:
        raise Exception("Please add an .env file that includes huggingface HF_TOKEN as HF_TOKEN=...")
    # Check checkpoint directory
    if not os.path.exists(CHECKPOINTS_DIR):
        os.mkdir(CHECKPOINTS_DIR)
    
def load_data(data_path):
    data = json.load(open(data_path))
    return data

def load_model(model_identifier, accelerator, is_seq2seq):
    tokenizer = AutoTokenizer.from_pretrained(model_identifier)
    if is_seq2seq:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_identifier, trust_remote_code=True, device_map="auto")
    else:
        model = AutoModelForCausalLM.from_pretrained(model_identifier, trust_remote_code=True, device_map="auto")
    model = accelerator.prepare(model)
    return model, tokenizer

def infere_from_model(model, tokenizer, prompt, device, is_seq2seq, has_token_types, max_input_token, verbose):
    
    if is_seq2seq:
        options_tokens = [tokenizer.encode(choice)[0] for choice in ["0", "1", "2", "3", "4"]]
    else:
        options_tokens = [tokenizer.encode(choice)[-1] for choice in ["0", "1", "2", "3", "4"]]
    
    with torch.no_grad():
        if max_input_token:
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_input_token).to(device)
        else:
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
        input_size = inputs['input_ids'].size(1)
        if verbose and input_size > max_input_token:
            print(f"Input is too long! The tokenized input has {input_size} tokens, which exceeds the maximum allowed size of {max_input_token} tokens. Was truncized.")
        input_ids = inputs["input_ids"].to(device)
        if has_token_types:
            inputs.pop("token_type_ids")     
        if is_seq2seq:
            start_token = tokenizer('<pad>', return_tensors="pt").to(device)
            outputs = model(**inputs, decoder_input_ids=start_token['input_ids'])
        else:
            outputs = model(**inputs)
        last_token_logits = outputs.logits[:, -1, :]
        options_tokens_logits = last_token_logits[:, options_tokens].detach().cpu().numpy()
        conf = softmax(options_tokens_logits[0])
        pred = np.argmax(options_tokens_logits[0])
    
    return pred, conf

def load_checkpoint(destination, model_identifier):
    model_identifier = model_identifier.replace("/", "-")
    filepath = os.path.join(destination, CHECKPOINTS_DIR, f"{model_identifier}.output")
    if os.path.exists(filepath):
        output = json.load(open(filepath))
    else:
        output = []
    return output

def save_checkpoint(destination, model_identifier, output):
    model_identifier = model_identifier.replace("/", "-")
    filepath = os.path.join(destination, CHECKPOINTS_DIR, f"{model_identifier}.output")
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
    parser.add_argument('-r', '--root', type=str, help="The ouput destination")
    parser.add_argument('-s', '--max_input_token', type=int, help="Max input tokens a model can consume")
    parser.add_argument('--has_token_types', action='store_true', default=False, help="Does the tokenizer output token types")
    parser.add_argument('--is_seq2seq', action='store_true', default=False, help="Is the model sequence to sequence")
    parser.add_argument('-v', '--verbose', action='store_true', help="Verbose")
    args = parser.parse_args()

    accelerator = Accelerator()
    device = accelerator.device

 
    if args.verbose: print("Setup ...")
    setup()

    if args.verbose: print("Load data ...")
    data = load_data(args.data)
    
    if args.verbose: print("Load model ...")
    model, tokenizer = load_model(args.model, accelerator, args.is_seq2seq)
    
    prompt_factory = PromptFactory()
    prompt_generator = prompt_factory.get_prompt_function(n_shots=0)

    if args.verbose: print("Load Checkpoint ...")
    if args.root:
        destination = args.root
    else:
        destination = ""
    output = load_checkpoint(destination, args.model)
    checkpoint = output[-1]["id"] if output else -1

    if args.verbose: print("Running Inference ...")
    for i, q in enumerate(data[checkpoint+1:]):
        id = i + checkpoint + 1
        question = q["question"]
        options = [q[f"option_{i}"] for i in [0, 1, 2, 3] if f"option_{i}" in q and q[f"option_{i}"] != "" ]
        prompt = prompt_generator(question, options, subject=q["subject"], level=q["level"])
        pred, conf = infere_from_model(model, tokenizer, prompt, device, args.is_seq2seq, args.has_token_types, args.max_input_token, args.verbose)
        output.append({"id": id, "prediction": int(pred), "confidence": float(conf[pred])})
        save_checkpoint(destination, args.model, output)
        if id>0 and id%100==0:
            if args.verbose: 
                print(f"Processed {id} questions")

if __name__ == '__main__':
    main()
