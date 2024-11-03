from typing import Dict, List, Any
import torch
from accelerate import Accelerator
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import numpy as np


def softmax(x):
    z = x - max(x)
    numerator = np.exp(z)
    denominator = np.sum(numerator)
    softmax = numerator/denominator
    return softmax

class EndpointHandler():
    def __init__(self, path=""):
        self.accelerator = Accelerator()
        self.device = self.accelerator.device
        self.model = AutoModelForSeq2SeqLM.from_pretrained(path, trust_remote_code=True, device_map="auto")
        self.model = self.accelerator.prepare(self.model)
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.options_tokens = [self.tokenizer.encode(choice)[0] for choice in ["A", "B", "C", "D"]]

    def __call__(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
       data args:
            inputs (:obj: `str` | `PIL.Image` | `np.array`)
            kwargss
      Return:
            A :obj:`list` | `dict`: will be serialized and returned
        """
        with torch.no_grad():
            prompt = data.pop("prompt")
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            input_size = inputs['input_ids'].size(1)
            input_ids = inputs["input_ids"].to(self.device)
            start_token = self.tokenizer('<pad>', return_tensors="pt").to(self.device)
            outputs = self.model(**inputs, decoder_input_ids=start_token['input_ids'])
            last_token_logits = outputs.logits[:, -1, :]
            options_tokens_logits = last_token_logits[:, self.options_tokens].detach().cpu().numpy()
            conf = softmax(options_tokens_logits[0])
            pred = np.argmax(options_tokens_logits[0])
        return [{"pred": pred, "conf":conf}]