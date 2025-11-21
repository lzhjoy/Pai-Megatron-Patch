# Copyright (c) 2025 Alibaba PAI and Nvidia Megatron-LM Team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from functools import partial
from contextlib import nullcontext
import torch
import torch._dynamo

from model_provider import model_provider as mcore_model_provider  # YuLan-Pretrain/model_provider.py

from megatron.core.models.gpt import GPTModel
from megatron.training import print_rank_0

from megatron_patch.model.qwen2_moe.layer_specs import (
    get_gpt_decoder_block_spec,
    get_gpt_layer_local_spec,
    get_gpt_layer_with_transformer_engine_spec,
)
from megatron_patch.model.qwen2_moe.transformer_config import core_transformer_config_from_args

torch._dynamo.config.suppress_errors = True


def qwen2_builder(args, pre_process, post_process, vp_stage=None):
    """Build Qwen2 model for conversion.
    
    Args:
        args: Arguments containing model configuration
        pre_process: Whether to compute embeddings
        post_process: Whether to compute output logits/loss
        vp_stage: Virtual pipeline stage (optional)
    
    Returns:
        GPTModel: The built Qwen2 model
    """
    use_te = args.transformer_impl == "transformer_engine"
    assert use_te, "Only transformer_engine is supported"
    
    print_rank_0('building Qwen2-Megatron model ...')
    
    config = core_transformer_config_from_args(args)
    
    if args.num_experts:
        # Define the decoder block spec
        transformer_layer_spec = get_gpt_decoder_block_spec(config, use_transformer_engine=use_te)
    else:
        # Define the decoder layer spec
        if use_te:
            transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(
                args.num_experts, args.moe_grouped_gemm,
                args.qk_layernorm, moe_use_legacy_grouped_gemm=args.moe_use_legacy_grouped_gemm)
        else:
            transformer_layer_spec = get_gpt_layer_local_spec(
                args.num_experts, args.moe_grouped_gemm,
                args.qk_layernorm, moe_use_legacy_grouped_gemm=args.moe_use_legacy_grouped_gemm)
    
    build_model_context = nullcontext
    build_model_context_args = {}
    
    with build_model_context(**build_model_context_args):
        model = GPTModel(
            config=config,
            transformer_layer_spec=transformer_layer_spec,
            vocab_size=args.padded_vocab_size,
            max_sequence_length=args.max_position_embeddings,
            pre_process=pre_process,
            post_process=post_process,
            fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
            parallel_output=True,
            share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
            position_embedding_type=args.position_embedding_type,
            rotary_percent=args.rotary_percent,
            rotary_base=args.rotary_base,
            rope_scaling=args.use_rope_scaling
        )
    
    return model


# Bind qwen2_builder to mcore_model_provider using partial
model_provider = partial(mcore_model_provider, qwen2_builder)

