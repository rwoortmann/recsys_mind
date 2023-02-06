from transformers import AutoTokenizer, AutoModel
import torch.nn as nn
import numpy as np
import torch

class SentenceTransformer(nn.Module):
    def __init__(self, st_model, device):
        super().__init__()
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/' + st_model)
        self.model = AutoModel.from_pretrained('sentence-transformers/' + st_model).to(device)

    def batch_to_device(self, batch):
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(self.device)
        return batch   
        
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def encode(self, sentences, normalize_embeddings=False, train=False):
        batch_size = 32
        all_embeddings = []
        length_sorted_idx = np.argsort([-len(sen) for sen in sentences])
        sentences_sorted = [sentences[idx] for idx in length_sorted_idx]
        
        for start_index in range(0, len(sentences), batch_size):
            sentences_batch = sentences_sorted[start_index:start_index+batch_size]
            encoded_input = self.tokenizer(sentences_batch, padding=True, truncation=True, return_tensors='pt')
            encoded_input = self.batch_to_device(encoded_input)
            
            if train: 
                out = self.model(**encoded_input, return_dict=True)
            else: 
                with torch.no_grad(): 
                    out = self.model(**encoded_input, return_dict=True)

            embeddings = self.mean_pooling(out, encoded_input['attention_mask'])
            
            if normalize_embeddings:
                embeddings = F.normalize(embeddings, p=2, dim=1)
            
            all_embeddings.extend(embeddings)
            
        all_embeddings = [all_embeddings[idx] for idx in np.argsort(length_sorted_idx)]
        all_embeddings = torch.stack(all_embeddings)
        
        return all_embeddings


    