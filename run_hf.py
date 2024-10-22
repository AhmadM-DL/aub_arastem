import argparse, os, json, requests
from utils_prompt import PromptFactory

CHECKPOINTS_DIR = "checkpoints"

def load_data(data_path):
    data = json.load(open(data_path))
    return data

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

def infere_from_model(endpoint, prompt):
    payload = {"inputs":"", "prompt": prompt}
    headers = {
        "Accept" : "application/json",
        "Authorization": "Bearer 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",
        "Content-Type": "application/json" 
    }
    response = requests.post(endpoint, headers=headers, json=payload)
    if response.status_code==200:
        output = response.json()
        return output["pred"], output["conf"]
    else:
        raise Exception(f"Error {response.status_code} requesting the model")

def main():
    parser = argparse.ArgumentParser(description="A script that requests a model and compute inference on a dataset.")
    parser.add_argument('-m', '--model', type=str, required=True, help="Model Identifier")
    parser.add_argument('-e', '--endpoint', type=str, required=True, help="Model Endpoint")
    parser.add_argument('-d', '--data', type=str, required=True, help="MCQ data to infere on")
    parser.add_argument('-r', '--root', type=str, default=".", help="The ouput destination")
    parser.add_argument('-v', '--verbose', action='store_true', help="Verbose")
    args = parser.parse_args()

    if args.verbose: print("Load data ...")
    data = load_data(args.data)

    prompt_factory = PromptFactory()
    prompt_generator = prompt_factory.get_prompt_function(n_shots=0)


    if args.verbose: print("Load Checkpoint ...")
    output = load_checkpoint(args.root, args.model)
    checkpoint = output[-1]["id"] if output else -1

    if args.verbose: print("Running Inference ...")
    for i, q in enumerate(data[checkpoint+1:]):
        id = i + checkpoint + 1
        question = q["question"]
        options = [q[f"option_{i}"] for i in [0, 1, 2, 3] if f"option_{i}" in q and q[f"option_{i}"] != "" ]
        prompt = prompt_generator(question, options, subject=q["subject"], level=q["level"])
        pred, conf = infere_from_model(args.endpoint, args.model)
        output.append({"id": id, "prediction": int(pred), "confidence": float(conf[pred])})
        save_checkpoint(args.root, args.model, output)
        if id>0 and id%100==0:
            if args.verbose: 
                print(f"Processed {id} questions")

if __name__ == '__main__':
    main()
