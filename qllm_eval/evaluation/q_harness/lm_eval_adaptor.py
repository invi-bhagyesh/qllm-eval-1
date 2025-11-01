import transformers
import torch
from lm_eval.api.model import LM
from lm_eval.api.instance import Instance
from typing import List, Tuple, Optional

class LMEvalAdaptor(LM):
    def __init__(self, model_name, model, tokenizer, batch_size=1, max_length=-1):
        super().__init__()
        
        assert isinstance(batch_size, int)
        
        self.model_name = model_name
        self.model = model
        self.model.eval()
        self.tokenizer = tokenizer
        self.vocab_size = self.tokenizer.vocab_size
        self._batch_size = batch_size
        self._max_length = max_length
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        if self._max_length != -1:
            return self._max_length
        if hasattr(self.model.config, 'n_ctx'):
            return self.model.config.n_ctx
        elif hasattr(self.model.config, 'max_position_embeddings'):
            return self.model.config.max_position_embeddings
        elif hasattr(self.model.config, 'n_positions'):
            return self.model.config.n_positions
        elif 'llama' in self.model_name.lower():
            return 2048
        else:
            return 2048

    @property
    def max_gen_toks(self):
        return 256

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def device(self):
        return self._device

    def tok_encode(self, string: str, **kwargs):
        return self.tokenizer.encode(string, add_special_tokens=False)

    def tok_decode(self, tokens, **kwargs):
        return self.tokenizer.decode(tokens)

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        """Required method for the new API"""
        results = []
        for request in requests:
            context, continuation = request.args
            # Implement your loglikelihood calculation here
            # This is a simplified version - adjust based on your needs
            context_enc = self.tok_encode(context)
            continuation_enc = self.tok_encode(continuation)
            full_enc = context_enc + continuation_enc
            
            input_ids = torch.tensor([full_enc]).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(input_ids)
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
            
            # Calculate log probability for continuation tokens
            logprobs = torch.nn.functional.log_softmax(logits, dim=-1)
            continuation_logprobs = []
            for i, token_id in enumerate(continuation_enc):
                pos = len(context_enc) + i - 1
                if pos >= 0 and pos < logprobs.size(1):
                    continuation_logprobs.append(
                        logprobs[0, pos, token_id].item()
                    )
            
            loglikelihood = sum(continuation_logprobs)
            is_greedy = True  # Implement greedy check if needed
            results.append((loglikelihood, is_greedy))
        
        return results

    def generate_until(self, requests: List[Instance]) -> List[str]:
        """Required method for the new API"""
        results = []
        for request in requests:
            context = request.args[0]
            gen_kwargs = request.args[1] if len(request.args) > 1 else {}
            
            input_ids = torch.tensor([self.tok_encode(context)]).to(self.device)
            
            with torch.no_grad():
                output_ids = self.model.generate(
                    input_ids,
                    max_new_tokens=gen_kwargs.get('max_gen_toks', self.max_gen_toks),
                    do_sample=False,
                    eos_token_id=self.eot_token_id,
                )
            
            generated_text = self.tok_decode(output_ids[0][input_ids.shape[1]:].tolist())
            results.append(generated_text)
        
        return results

    def loglikelihood_rolling(self, requests: List[Instance]) -> List[float]:
        """Required method for the new API"""
        results = []
        for request in requests:
            string = request.args[0]
            tokens = self.tok_encode(string)
            
            input_ids = torch.tensor([tokens]).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(input_ids)
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
            
            logprobs = torch.nn.functional.log_softmax(logits, dim=-1)
            total_logprob = 0.0
            
            for i in range(1, len(tokens)):
                total_logprob += logprobs[0, i-1, tokens[i]].item()
            
            results.append(total_logprob)
        
        return results
