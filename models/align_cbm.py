import torch
from torch import nn
import pandas as pd
import numpy as np
import torch.nn.functional as F

class ConceptBottleneckModel(nn.Module):
    def __init__(self, attribute_predictor, classifier, visual_xai=None):
        super().__init__()
        self.attribute_predictor = attribute_predictor.cuda()
        self.classifier = classifier.cuda()
        self.visual_xai = visual_xai.cuda() if visual_xai is not None else None

    def set_visual_xai(self, visual_xai):
        self.visual_xai = visual_xai
    
    def forward(self, x, apply_softmax=False, get_delta_y=True):
        c_logits = self.attribute_predictor(x)
        c_probs_list = [torch.nn.functional.softmax(output, dim=1) for output in c_logits]
        c_probs = torch.cat(c_probs_list, dim=1)
        
        y_logits = self.classifier(c_probs) #y before softmax
        y_probs = torch.softmax(y_logits, dim=1) #y after softmax

        if not get_delta_y:
            if apply_softmax:
                return c_probs_list, y_probs   
            else:
                return c_logits, y_logits
            
        # initalize delta_y
        actual_batch_size = c_probs.size(0)
        num_attributes = len(c_probs_list)
        delta_y = torch.zeros(actual_batch_size, y_probs.size(1), num_attributes).to(y_probs.device) # [B, K, L]

        for i in range(num_attributes):  # For each concept probability
            c_i_list = [c.clone() for c in c_probs_list] 
            # Set all sub-concept probabilities to 0 in the i-th concept
            c_i_list[i] = torch.zeros_like(c_probs_list[i]) 
            c_i = torch.cat(c_i_list, dim=1)
            y_i_logits = self.classifier(c_i) # Predict with modified concept  
            y_i_probs = torch.softmax(y_i_logits, dim=1)
            delta_y[:, :, i] = torch.abs(y_probs - y_i_probs)
            
        if not apply_softmax:
            return c_logits, y_logits, delta_y
        else:   # for testing, return model output after softmax 
            return c_probs_list, y_probs, delta_y 
