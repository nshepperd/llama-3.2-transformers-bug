from typing import Tuple
import torch
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor
import transformers
import argparse

def fixed_forward(
    self: transformers.models.mllama.modeling_mllama.MllamaCrossAttentionDecoderLayer,
    hidden_states: torch.Tensor,
    cross_attention_states: torch.Tensor,
    cross_attention_mask: torch.Tensor,
    attention_mask: torch.Tensor,
    full_text_row_masked_out_mask: torch.Tensor,
    position_ids = None,
    past_key_value = None,
    output_attentions = False,
    use_cache = False,
    cache_position = None,
    position_embeddings = None,
    **kwargs,
) -> Tuple[torch.Tensor] | Tuple[torch.Tensor,torch.Tensor]:
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)

    hidden_states, attn_weights = self.cross_attn(
        hidden_states=hidden_states,
        attention_mask=cross_attention_mask,
        cross_attention_states=cross_attention_states,
        past_key_value=past_key_value,
        output_attentions=output_attentions,
        cache_position=cache_position,
        **kwargs,
    )
    if full_text_row_masked_out_mask is not None:
        hidden_states = full_text_row_masked_out_mask[:, 0] * hidden_states  # type: ignore
    hidden_states = residual + self.cross_attn_attn_gate.tanh() * hidden_states

    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    if full_text_row_masked_out_mask is not None:
        hidden_states = full_text_row_masked_out_mask[:, 0] * hidden_states  # type: ignore
    hidden_states = residual + self.cross_attn_mlp_gate.tanh() * hidden_states

    outputs = (hidden_states,)

    if output_attentions:
        outputs += (attn_weights,)

    return outputs

parser = argparse.ArgumentParser()
parser.add_argument('--fixed', action='store_true', help='Use the fixed implementation')
args = parser.parse_args()
if args.fixed:
    transformers.models.mllama.modeling_mllama.MllamaCrossAttentionDecoderLayer.forward = fixed_forward # type: ignore


model_id = "meta-llama/Llama-3.2-11B-Vision"

model = MllamaForConditionalGeneration.from_pretrained(
    model_id,
    # torch_dtype=torch.bfloat16,
    device_map="auto",
)

processor = AutoProcessor.from_pretrained(model_id)

image = Image.open("llama-models/llama_models/scripts/resources/dog.jpg")

# Note: <|begin_of_text|> *before* <|image|> results in 1 token that can't see the image,
# which is what triggers the issue here.(the same would occur in any dialogue that had an 
# image added halfway through).
input_text = "<|begin_of_text|>Let's write a poem.<|image|>If I had to write a haiku for this one"
inputs = processor(
    text=[input_text], images=[[image]], add_special_tokens=False, return_tensors="pt"
).to(model.device)
prompt_len = inputs.input_ids.shape[1]

with torch.no_grad():
    print(processor.decode(inputs.input_ids[0]))
    output = model.generate(**inputs, max_new_tokens=100, do_sample=False)
    print(processor.decode(output[0, prompt_len:]))
