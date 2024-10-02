import argparse
from huggingface_hub import snapshot_download

def download_model(model_identifier):
    snapshot_download(repo_id=model_identifier)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download a model from Hugging Face Hub.")
    parser.add_argument("-m", "--model", type=str, help="Name of the model to download from Hugging Face Hub.")
    args = parser.parse_args()
    
    download_model(args.model)