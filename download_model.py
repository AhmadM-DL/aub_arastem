import argparse
from huggingface_hub import snapshot_download

def download_model(model_identifier, hf_token):
    snapshot_download(repo_id=model_identifier, token= hf_token)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download a model from Hugging Face Hub.")
    parser.add_argument("-m", "--model", type=str, help="Name of the model to download from Hugging Face Hub.")
    parser.add_argument("-token", "--token", type=str, help="Hugging Face token")
    args = parser.parse_args()
    
    download_model(args.model, args.token)