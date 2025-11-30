# complete the beam expansion logic in the provided code

import torch
import torch.nn.functional as F

class TransformerInference:
    def __init__(self, model, src_vocab, tgt_vocab):
        self.model = model
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.model.eval()
    
    def greedy_decode(self, src_sentence, max_len=50):
        """Greedy decoding - select most likely token at each step"""
        # Encode source sentence with consistent max_len
        src_tokens = torch.tensor([self.src_vocab.encode(src_sentence, max_len=15)], dtype=torch.long)
        src_mask = self.model.create_padding_mask(src_tokens)
        
        # Initialize decoder input with SOS token
        tgt_tokens = torch.tensor([[self.tgt_vocab.token2idx['<SOS>']]], dtype=torch.long)
        
        for _ in range(max_len):
            # Create both causal and padding masks for target
            tgt_causal_mask = self.model.create_causal_mask(tgt_tokens.size(1))
            tgt_padding_mask = self.model.create_padding_mask(tgt_tokens)
            tgt_mask = tgt_causal_mask & tgt_padding_mask
            
            # Forward pass with no_grad for better performance
            with torch.no_grad():
                output = self.model(src_tokens, tgt_tokens, src_mask, tgt_mask)
            
            # Get next token
            next_token_logits = output[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(1)
            
            # Stop if EOS token generated
            if next_token.item() == self.tgt_vocab.token2idx['<EOS>']:
                break
            
            # Append to sequence
            tgt_tokens = torch.cat([tgt_tokens, next_token], dim=1)
        
        return self.tgt_vocab.decode(tgt_tokens[0].tolist())
    
    def beam_search(self, src_sentence, beam_width=3, max_len=50):
        """Beam search decoding - maintain multiple hypotheses"""
        # Encode source with consistent max_len
        src_tokens = torch.tensor([self.src_vocab.encode(src_sentence, max_len=15)], dtype=torch.long)
        src_mask = self.model.create_padding_mask(src_tokens)
        
        # Initialize beam with SOS token
        beams = [{'tokens': [self.tgt_vocab.token2idx['<SOS>']], 'score': 0.0}]
        
        for step in range(max_len):
            candidates = []
            
            for beam in beams:
                if beam['tokens'][-1] == self.tgt_vocab.token2idx['<EOS>']:
                    candidates.append(beam)
                    continue
                
                # Get current sequence
                tgt_tokens = torch.tensor([beam['tokens']], dtype=torch.long)
                # Create both causal and padding masks for target
                tgt_causal_mask = self.model.create_causal_mask(tgt_tokens.size(1))
                tgt_padding_mask = self.model.create_padding_mask(tgt_tokens)
                tgt_mask = tgt_causal_mask & tgt_padding_mask
                
                # Forward pass with no_grad for better performance
                with torch.no_grad():
                    output = self.model(src_tokens, tgt_tokens, src_mask, tgt_mask)
                
                # Get probabilities for next token
                logits = output[:, -1, :]
                
                # TODO: Calculate log probabilities for the next token
                probs = F.log_softmax(logits, dim=-1)
                
                # TODO: Get the top 'beam_width' candidates (probabilities and indices)
                top_probs, top_indices = torch.topk(probs, beam_width)
                
                # TODO: Create new candidate beams by extending the current beam
                # with each of the top indices and their corresponding scores.
                # Append each new candidate as a dictionary to the 'candidates' list.
                for prob, idx in zip(top_probs[0], top_indices[0]):
                    new_tokens = beam['tokens'] + [idx.item()]
                    new_score = beam['score'] + prob.item()
                    candidates.append({'tokens': new_tokens, 'score': new_score})
            
            # Select top beam_width candidates
            candidates.sort(key=lambda x: x['score'], reverse=True)
            beams = candidates[:beam_width]
            
            # Check if all beams ended
            if all(beam['tokens'][-1] == self.tgt_vocab.token2idx['<EOS>'] for beam in beams):
                break
        
        # Return best sequence
        best_beam = max(beams, key=lambda x: x['score'])
        return self.tgt_vocab.decode(best_beam['tokens'])