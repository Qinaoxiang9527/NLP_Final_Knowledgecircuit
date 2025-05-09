from functools import partial
from typing import List, Optional, Union
import sys
sys.path.append('..')
import einops
import numpy as np
import plotly.express as px
import plotly.io as pio
import torch
from fancy_einsum import einsum
from IPython.display import HTML, IFrame
from jaxtyping import Float
import sys
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import transformers
import seaborn as sns
import transformer_lens.utils as utils
from transformer_lens import ActivationCache, HookedTransformer, SVDInterpreter

# 禁用自动微分
torch.set_grad_enabled(False)
print("Disabled automatic differentiation")
plt.rcParams['font.sans-serif']=['Times New Roman']

def decode_tokens(tokenizer, token_array):
    if hasattr(token_array, "shape") and len(token_array.shape) > 1:
        return [decode_tokens(tokenizer, row) for row in token_array]
    return [tokenizer.decode([t]) for t in token_array]

def find_sublist_index(main_list, sub_list):
    main_list = main_list.tolist()
    sub_list = sub_list.tolist()
    index = main_list.index(sub_list[0])
    return (index, index+len(sub_list))

def check_subrange(tokens, sublist, range):
    subtokens = tokens[range[0]:range[1]]
    assert len(sublist) == len(subtokens)
    for x, y in zip(sublist, subtokens):
        if x == y:
            continue
        else:
            return False
    return True

def find_token_range(tokenizer, token_array, substring):
    toks = decode_tokens(tokenizer, token_array)
    whole_string = "".join(toks)
    char_loc = whole_string.index(substring)
    loc = 0
    tok_start, tok_end = None, None
    for i, t in enumerate(toks):
        loc += len(t)
        if tok_start is None and loc > char_loc:
            tok_start = i
        if tok_end is None and loc >= char_loc + len(substring):
            tok_end = i + 1
            break
    return (tok_start, tok_end)

class ComponentAnalyzer:
    def __init__(self, model, prompt, answer, subject) -> None:
        # List of prompts
        self.prompts = [prompt]
        # print(prompt)
        tokens = model.to_tokens(self.prompts, prepend_bos=True)
        str_tokens = model.to_str_tokens(self.prompts, prepend_bos=True)
        self.tokens = tokens
        self.str_tokens = str_tokens
        if 'llama' in model.cfg.model_name:
            subject_range = find_sublist_index(tokens[0], model.to_tokens(subject, prepend_bos=False)[0])
            assert check_subrange(tokens[0], model.to_tokens(subject, prepend_bos=False)[0], subject_range)
        else:
            subject_range = find_token_range(model.tokenizer, tokens[0], subject)
        # List of answers, in the format (correct, incorrect)
        self.answer_token = model.to_tokens(answer, prepend_bos=False)[0][0]
        self.subject_range = subject_range
        # self.subject_pos = subject_range[-1]-1
        self.subject_last_token = tokens[0][subject_range[-1]-1]
        self.logits, self.cache = model.run_with_cache(tokens)
        self.accum_resid, self.labels = self.cache.accumulated_resid(incl_mid=False, mlp_input=True, return_labels=True, apply_ln=True)
        self.heads_out = self.get_heads_out(model)
        self.mlp_out = self.get_mlp_out(model)
        
    def get_min_rank_at_subject(self, W_U, token_num):
        last_token_accum = self.accum_resid[:, 0, self.subject_range[0]:self.subject_range[1], :] 
        layers_unembedded = einsum(
                "layer ... d_model, d_model d_vocab -> layer ... d_vocab",
                last_token_accum,
                W_U,
            )
        sorted_indices = torch.argsort(layers_unembedded, dim=2, descending=True)
        rank_answer = (sorted_indices == token_num).nonzero(as_tuple=True)[2].view(layers_unembedded.size(0), -1)
        return rank_answer.min(dim=-1)[0]
    
    def get_token_rank(self, W_U, token_num, pos=-1):
        last_token_accum = self.accum_resid[:, 0, pos, :] 
        layers_unembedded = einsum(
                "layer d_model, d_model d_vocab -> layer d_vocab",
                last_token_accum,
                W_U,
            )
        sorted_indices = torch.argsort(layers_unembedded, dim=1, descending=True)
        rank_answer = (sorted_indices == token_num).nonzero(as_tuple=True)[1]
        return rank_answer
    
    def get_token_logits(self, model, tokens, pos=-1):
        answer_residual_directions = model.tokens_to_residual_directions(tokens)
        # print("Answer residual directions shape:", answer_residual_directions.shape)
        logit_diff_directions = (
            answer_residual_directions[:, ]
        )
        if len(logit_diff_directions.shape) == 1:
            logit_diff_directions = logit_diff_directions.unsqueeze(0)
        scaled_residual_stack = self.cache.apply_ln_to_stack(
                self.accum_resid, layer=-1, pos_slice=-1
            )
        # print(scaled_residual_stack.shape)
        return einsum(
                "... batch d_model, batch d_model -> ...",
                scaled_residual_stack,
                logit_diff_directions,
            )
            
    def get_token_probability(self, model, tokens, pos=-1):
        last_token_accum = self.accum_resid[:, 0, pos, :] 
        layers_unembedded = einsum(
                "layer d_model, d_model d_vocab -> layer d_vocab",
                last_token_accum,
                model.W_U,
            )
        probs = layers_unembedded.softmax(dim=-1)
        # print(scaled_residual_stack.shape)
        return probs[:, tokens]
        
    def get_heads_out(self, model, pos_slice=-1):
        per_head_residual, labels = self.cache.stack_head_results(
            layer=-1, pos_slice=pos_slice, return_labels=True, apply_ln=True
        )
        heads_out = {}
        for index, label in enumerate(labels):
            # Set the label
            layer = index // model.cfg.n_heads
            head_index = index % model.cfg.n_heads
            assert f"L{layer}H{head_index}" == label
            heads_out[label] = per_head_residual[index, :]
        return heads_out
        
    def get_mlp_out(self, model, pos_slice=-1):
        per_layer_residual, labels = self.cache.decompose_resid(
            mode='mlp', layer=-1, pos_slice=-1, return_labels=True, apply_ln=True
        )
        mlp_out = {}
        for x, y in zip(per_layer_residual, labels):
            mlp_out[y] = x
        return mlp_out
        
    def get_component_logits(self, output, model):
        # print(heads_out[head_name].shape)
        layers_unembedded = einsum(
                " ... d_model, d_model d_vocab -> ... d_vocab",
                output,
                model.W_U,
            )
        sorted_indices = torch.argsort(layers_unembedded, dim=1, descending=True)
        temp_logits = layers_unembedded[0]
        tmp_sorted_indices = sorted_indices[0]
        for i in range(10):
            print(
                f"Top {i}th token. Logit: {temp_logits[tmp_sorted_indices[i]].item():5.2f} Token: |{model.to_string(tmp_sorted_indices[i])}|"
            )

def draw_output_pattern_with_text(component, model, top_k=10):
    layers_unembedded = einsum(
            " ... d_model, d_model d_vocab -> ... d_vocab",
            component,
            model.W_U,
        )
    sorted_indices = torch.argsort(layers_unembedded, dim=1, descending=True)
    temp_logits = layers_unembedded[0]
    tmp_sorted_indices = sorted_indices[0]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(4.5, 18))
    
    # 绘制前top_k个token的logits
    tokens = []
    logits = []
    for i in range(top_k):
        token = model.to_string(tmp_sorted_indices[i])
        logit = temp_logits[tmp_sorted_indices[i]].item()
        tokens.append(token)
        logits.append(logit)
    
    ax1.barh(range(top_k), logits, align='center')
    ax1.set_yticks(range(top_k))
    ax1.set_yticklabels(tokens)
    ax1.set_xlabel('Logits')
    ax1.set_title('Top Tokens')
    
    # 绘制所有token的logits分布
    ax2.hist(temp_logits.cpu().numpy(), bins=50)
    ax2.set_xlabel('Logits')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Logits Distribution')
    
    plt.tight_layout()
    plt.show()

# 以下是使用示例代码（可以根据需要保留或删除）
if __name__ == "__main__":
    # 加载模型示例
    model = HookedTransformer.from_pretrained("gpt2-medium")
    
    # 创建ComponentAnalyzer实例示例
    prompt = "The capital of France is"
    answer = "Paris"
    subject = "France"
    analyzer = ComponentAnalyzer(model, prompt, answer, subject)
    
    # 分析特定头的输出示例
    head_name = "L10H5"  # 第10层第5个注意力头
    if head_name in analyzer.heads_out:
        draw_output_pattern_with_text(analyzer.heads_out[head_name], model, 10)