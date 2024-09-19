import argparse, os, json
import transformers
from transformers import AutoTokenizer
from utils import PromptFactory

CHECKPOINTS_DIR = "./checkpoints/"

def setup():
    os.environ["HF_TOKEN"] = "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"
    if os.path.exists(CHECKPOINTS_DIR):
        pass
    else:
        os.mkdir(CHECKPOINTS_DIR)
    

def load_data(data_path):
    data = json.load(open(data_path))
    return data

def load_model(model_identifier):
    tokenizer = AutoTokenizer.from_pretrained(model_identifier)
    generator = transformers.pipeline(
    "text-generation",
    model=model_identifier,
    tokenizer=tokenizer,
    #torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
    return_full_text=False)
    return generator, tokenizer

def load_checkpoint(model_identifier):
    filepath = os.path.join(CHECKPOINTS_DIR, f"{model_identifier}.output")
    if os.path.exists(filepath):
        output = json.load(open(filepath))
    else:
        output = []
    return output

def save_checkpoint(model_identifier, output):
    filepath = os.path.join(CHECKPOINTS_DIR, f"{model_identifier}.output")
    json.dump(output, open(filepath, "w"), ensure_ascii=False, indent=4)

def main():
    parser = argparse.ArgumentParser(description="A script that loads a model and compute inference on a dataset.")
    parser.add_argument('-m', '--model', type=str, required=True, help="Huggingface model identifier")
    parser.add_argument('-d', '--data', type=str, required=True, help="MCQ data to infere on")
    parser.add_argument('-s', '--maxsize', type=int, required=True, help="Generated text max. size")
    # parser.add_argument('-x', '--xxx', type=str, required=True, help="xxx")
    # parser.add_argument('-x', '--xxx', type=int, required=True, help="xxx")
    # parser.add_argument('-x', '--xxx', action='store_true', help="xxx")
    args = parser.parse_args()

    setup()
    
    data = load_data(args.data)
    
    generator, tokenizer = load_model(args.model)
    
    prompt_factory = PromptFactory()
    prompt_generator = prompt_factory.get_prompt_function(n_shots=0)

    output = load_checkpoint(args.model)
    checkpoint = output[-1].id

    for i, q in enumerate(data[checkpoint+1:]):
        id = i + checkpoint + 1
        question = q["question"]
        options = [q[f"option_{i}"] for i in [0, 1, 2, 3] if f"option_{i}" in q and q[f"option_{i}"] != "" ]
        prompt = prompt_generator(question, options, subject=q["subject"], level=q["level"])
        generated_outputs = generator(question, pad_token_id= tokenizer.eos_token_id, max_new_tokens= args.maxsize)
        generated_output = generated_outputs[0]['generated_text']
        output.append({"id": id, "generated_text": generated_output})
        save_checkpoint(args.model, output)

if __name__ == '__main__':
    main()
