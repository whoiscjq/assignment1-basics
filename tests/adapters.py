from __future__ import annotations

import os
from collections.abc import Iterable
from typing import IO, Any, BinaryIO

import numpy.typing as npt
import numpy as np
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor
from .layers import Linear, Embedding, Rmsnorm, SwiGLU, RotaryPositionalEmbedding, softmax, scaled_dot_product_attention, Multihead_self_attention, Multihead_self_attention_with_rope
from .layers import Transformer_block, Transformer_lm, silu
from einops import rearrange, einsum
import math


def run_linear(
    d_in: int,
    d_out: int,
    weights: Float[Tensor, " d_out d_in"],
    in_features: Float[Tensor, " ... d_in"],
) -> Float[Tensor, " ... d_out"]:
    """
    Given the weights of a Linear layer, compute the transformation of a batched input.

    Args:
        in_dim (int): The size of the input dimension
        out_dim (int): The size of the output dimension
        weights (Float[Tensor, "d_out d_in"]): The linear weights to use
        in_features (Float[Tensor, "... d_in"]): The output tensor to apply the function to

    Returns:
        Float[Tensor, "... d_out"]: The transformed output of your linear module.
    """
    layer=Linear(d_in,d_out)
    layer.load_state_dict({"weight":weights})
    return layer(in_features)


def run_embedding(
    vocab_size: int,
    d_model: int,
    weights: Float[Tensor, " vocab_size d_model"],
    token_ids: Int[Tensor, " ..."],
) -> Float[Tensor, " ... d_model"]:
    """
    Given the weights of an Embedding layer, get the embeddings for a batch of token ids.

    Args:
        vocab_size (int): The number of embeddings in the vocabulary
        d_model (int): The size of the embedding dimension
        weights (Float[Tensor, "vocab_size d_model"]): The embedding vectors to fetch from
        token_ids (Int[Tensor, "..."]): The set of token ids to fetch from the Embedding layer

    Returns:
        Float[Tensor, "... d_model"]: Batch of embeddings returned by your Embedding layer.
    """

    layer=Embedding(vocab_size,d_model)
    layer.load_state_dict({"weight":weights})
    return layer(token_ids)



def run_swiglu(
    d_model: int,
    d_ff: int,
    w1_weight: Float[Tensor, " d_ff d_model"],
    w2_weight: Float[Tensor, " d_model d_ff"],
    w3_weight: Float[Tensor, " d_ff d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """Given the weights of a SwiGLU network, return
    the output of your implementation with these weights.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        d_ff (int): Dimensionality of the up-project happening internally to your swiglu.
        w1_weight (Float[Tensor, "d_ff d_model"]): Stored weights for W1
        w2_weight (Float[Tensor, "d_model d_ff"]): Stored weights for W2
        w3_weight (Float[Tensor, "d_ff d_model"]): Stored weights for W3
        in_features (Float[Tensor, "... d_model"]): Input embeddings to the feed-forward layer.

    Returns:
        Float[Tensor, "... d_model"]: Output embeddings of the same shape as the input embeddings.
    """
    # Example:
    # If your state dict keys match, you can use `load_state_dict()`
    # swiglu.load_state_dict(weights)
    # You can also manually assign the weights
    # swiglu.w1.weight.data = w1_weight
    # swiglu.w2.weight.data = w2_weight
    # swiglu.w3.weight.data = w3_weight
    swiglu=SwiGLU(d_model,d_ff)
    swiglu.w1.weight.data = w1_weight
    swiglu.w2.weight.data = w2_weight
    swiglu.w3.weight.data = w3_weight
    return swiglu(in_features)


def run_scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    """
    Given key (K), query (Q), and value (V) tensors, return
    the output of your scaled dot product attention implementation.

    Args:
        Q (Float[Tensor, " ... queries d_k"]): Query tensor
        K (Float[Tensor, " ... keys d_k"]): Key tensor
        V (Float[Tensor, " ... values d_v"]): Values tensor
        mask (Bool[Tensor, " ... queries keys"] | None): Mask tensor
    Returns:
        Float[Tensor, " ... queries d_v"]: Output of SDPA
    """
    return scaled_dot_product_attention(Q,K,V,mask)


def run_multihead_self_attention(
    d_model: int,
    num_heads: int,
    q_proj_weight: Float[Tensor, " d_k d_in"],
    k_proj_weight: Float[Tensor, " d_k d_in"],
    v_proj_weight: Float[Tensor, " d_v d_in"],
    o_proj_weight: Float[Tensor, " d_model d_v"],
    in_features: Float[Tensor, " ... sequence_length d_in"],
) -> Float[Tensor, " ... sequence_length d_out"]:
    """
    Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This implementation should handle the key, query, and value projections
    for all heads in a single matrix multiply.
    This function should not use RoPE.
    See section 3.2.2 of Vaswani et al., 2017.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        q_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the Q projection
        k_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the K projection
        v_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the V projection
        o_proj_weight (Float[Tensor, "d_model d_v"]): Weights for the output projection
        in_features (Float[Tensor, "... sequence_length d_in"]): Tensor to run your implementation on.

    Returns:
        Float[Tensor, " ... sequence_length d_out"]: Tensor with the output of running your optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
    """

    layer=Multihead_self_attention(d_model,num_heads)
    layer.q_proj.weight.data=q_proj_weight
    layer.k_proj.weight.data=k_proj_weight
    layer.v_proj.weight.data=v_proj_weight
    layer.output_proj.weight.data=o_proj_weight
    return layer(in_features)


def run_multihead_self_attention_with_rope(
    d_model: int,
    num_heads: int,
    max_seq_len: int,
    theta: float,
    q_proj_weight: Float[Tensor, " d_k d_in"],
    k_proj_weight: Float[Tensor, " d_k d_in"],
    v_proj_weight: Float[Tensor, " d_v d_in"],
    o_proj_weight: Float[Tensor, " d_model d_v"],
    in_features: Float[Tensor, " ... sequence_length d_in"],
    token_positions: Int[Tensor, " ... sequence_length"] | None = None,
) -> Float[Tensor, " ... sequence_length d_out"]:
    """
    Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This implementation should handle the key, query, and value projections
    for all heads in a single matrix multiply.
    This version of MHA should include RoPE.
    In this case, the RoPE embedding dimension must be the head embedding dimension (d_model // num_heads).
    See section 3.2.2 of Vaswani et al., 2017.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        theta (float): RoPE parameter.
        q_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the Q projection
        k_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the K projection
        v_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the V projection
        o_proj_weight (Float[Tensor, "d_model d_v"]): Weights for the output projection
        in_features (Float[Tensor, "... sequence_length d_in"]): Tensor to run your implementation on.
        token_positions (Int[Tensor, " ... sequence_length"] | None): Optional tensor with the positions of the tokens

    Returns:
        Float[Tensor, " ... sequence_length d_out"]: Tensor with the output of running your optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
    """
    layer=Multihead_self_attention_with_rope(d_model,num_heads,theta,max_seq_len)
    layer.q_proj.weight.data=q_proj_weight
    layer.k_proj.weight.data=k_proj_weight
    layer.v_proj.weight.data=v_proj_weight
    layer.output_proj.weight.data=o_proj_weight
    return layer(in_features,token_positions)


def run_rope(
    d_k: int,
    theta: float,
    max_seq_len: int,
    in_query_or_key: Float[Tensor, " ... sequence_length d_k"],
    token_positions: Int[Tensor, " ... sequence_length"],
) -> Float[Tensor, " ... sequence_length d_k"]:
    """
    Run RoPE for a given input tensor.

    Args:
        d_k (int): Embedding dimension size for the query or key tensor.
        theta (float): RoPE parameter.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        in_query_or_key (Float[Tensor, "... sequence_length d_k"]): Input tensor to run RoPE on.
        token_positions (Int[Tensor, "... sequence_length"]): Tensor of shape (batch_size, sequence_length) with the token positions
    Returns:
        Float[Tensor, " ... sequence_length d_k"]: Tensor with RoPEd input.
    """
    layer=RotaryPositionalEmbedding(theta=theta,d_k=d_k,max_seq_len=max_seq_len)
    return layer(in_query_or_key,token_positions)

    


def run_transformer_block(
    d_model: int,
    num_heads: int,
    d_ff: int,
    max_seq_len: int,
    theta: float,
    weights: dict[str, Tensor],
    in_features: Float[Tensor, " batch sequence_length d_model"],
) -> Float[Tensor, " batch sequence_length d_model"]:
    """
    Given the weights of a pre-norm Transformer block and input features,
    return the output of running the Transformer block on the input features.

    This function should use RoPE.
    Depending on your implementation, you may simply need to pass the relevant args
    to your TransformerBlock constructor, or you may need to initialize your own RoPE
    class and pass that instead.

    Args:
        d_model (int): The dimensionality of the Transformer block input.
        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        theta (float): RoPE parameter.
        weights (dict[str, Tensor]):
            State dict of our reference implementation.
            The keys of this dictionary are:
            - `attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is (d_model, d_model).
            - `ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `ffn.w3.weight`
                Weight of the third linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
        in_features (Float[Tensor, "batch sequence_length d_model"]):
            Tensor to run your implementation on.

    Returns:
        Float[Tensor, "batch sequence_length d_model"] Tensor with the output of
        running the Transformer block on the input features while using RoPE.
    """
    layer=Transformer_block(d_model, num_heads, d_ff, theta, max_seq_len)
    layer.load_state_dict(weights)
    return layer(in_features)


def run_transformer_lm(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    weights: dict[str, Tensor],
    in_indices: Int[Tensor, " batch_size sequence_length"],
) -> Float[Tensor, " batch_size sequence_length vocab_size"]:
    """Given the weights of a Transformer language model and input indices,
    return the output of running a forward pass on the input indices.

    This function should use RoPE.

    Args:
        vocab_size (int): The number of unique items in the output vocabulary to be predicted.
        context_length (int): The maximum number of tokens to process at once.
        d_model (int): The dimensionality of the model embeddings and sublayer outputs.
        num_layers (int): The number of Transformer layers to use.
        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer (section 3.3).
        rope_theta (float): The RoPE $\Theta$ parameter.
        weights (dict[str, Tensor]):
            State dict of our reference implementation. {num_layers} refers to an
            integer between `0` and `num_layers - 1` (the layer index).
            The keys of this dictionary are:
            - `token_embeddings.weight`
                Token embedding matrix. Shape is (vocab_size, d_model).
            - `layers.{num_layers}.attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is ((d_model / num_heads) * num_heads, d_model).
            - `layers.{num_layers}.ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `layers.{num_layers}.ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `layers.{num_layers}.ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `layers.{num_layers}.ffn.w3.weight`
                Weight of the third linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `layers.{num_layers}.ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ln_final.weight`
                Weights of affine transform for RMSNorm applied to the output of the final transformer block.
                Shape is (d_model, ).
            - `lm_head.weight`
                Weights of the language model output embedding.
                Shape is (vocab_size, d_model).
        in_indices (Int[Tensor, "batch_size sequence_length"]) Tensor with input indices to run the language model on. Shape is (batch_size, sequence_length), where
            `sequence_length` is at most `context_length`.

    Returns:
        Float[Tensor, "batch_size sequence_length vocab_size"]: Tensor with the predicted unnormalized
        next-word distribution for each token.
    """
    layer=Transformer_lm(
        vocab_size = vocab_size,
        context_length = context_length,
        d_model= d_model,
        num_layers = num_layers,
        num_heads = num_heads,
        d_ff = d_ff,
        rope_theta = rope_theta,
        device=None,
        dtype=None,
    )
    layer.load_state_dict(weights)
    return layer(in_indices)


def run_rmsnorm(
    d_model: int,
    eps: float,
    weights: Float[Tensor, " d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """Given the weights of a RMSNorm affine transform,
    return the output of running RMSNorm on the input features.

    Args:
        d_model (int): The dimensionality of the RMSNorm input.
        eps: (float): A value added to the denominator for numerical stability.
        weights (Float[Tensor, "d_model"]): RMSNorm weights.
        in_features (Float[Tensor, "... d_model"]): Input features to run RMSNorm on. Can have arbitrary leading
            dimensions.

    Returns:
        Float[Tensor,"... d_model"]: Tensor of with the same shape as `in_features` with the output of running
        RMSNorm of the `in_features`.
    """
    layer=Rmsnorm(d_model,eps)
    layer.load_state_dict({"weight":weights})
    return layer(in_features)


def run_silu(in_features: Float[Tensor, " ..."]) -> Float[Tensor, " ..."]:
    """Given a tensor of inputs, return the output of applying SiLU
    to each element.

    Args:
        in_features(Float[Tensor, "..."]): Input features to run SiLU on. Shape is arbitrary.

    Returns:
        Float[Tensor,"..."]: of with the same shape as `in_features` with the output of applying
        SiLU to each element.
    """
    return silu(in_features)

import random
def run_get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given a dataset (a 1D numpy array of integers) and a desired batch size and
    context length, sample language modeling input sequences and their corresponding
    labels from the dataset.

    Args:
        dataset (np.array): 1D numpy array of integer token IDs in the dataset.
        batch_size (int): Desired batch size to sample.
        context_length (int): Desired context length of each sampled example.
        device (str): PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the device
            to place the sampled input sequences and labels on.

    Returns:
        Tuple of torch.LongTensors of shape (batch_size, context_length). The first tuple item
        is the sampled input sequences, and the second tuple item is the corresponding
        language modeling labels.
    """
    first_list=[]
    next_list=[]
    length=dataset.shape[0]
    for _ in range(batch_size):
        idx=random.randint(0,length-context_length-1)
        first_list.append(dataset[idx:idx+context_length]) 
        next_list.append(dataset[idx+1:idx+context_length+1])
    return torch.from_numpy(np.stack(first_list)).long().to(device),torch.from_numpy(np.stack(next_list)).long().to(device)
    


def run_softmax(in_features: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    """
    Given a tensor of inputs, return the output of softmaxing the given `dim`
    of the input.

    Args:
        in_features (Float[Tensor, "..."]): Input features to softmax. Shape is arbitrary.
        dim (int): Dimension of the `in_features` to apply softmax to.

    Returns:
        Float[Tensor, "..."]: Tensor of with the same shape as `in_features` with the output of
        softmax normalizing the specified `dim`.
    """
    return softmax(in_features,dim=dim)


def run_cross_entropy(
    inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]
) -> Float[Tensor, ""]:
    """Given a tensor of inputs and targets, compute the average cross-entropy
    loss across examples.

    Args:
        inputs (Float[Tensor, "batch_size vocab_size"]): inputs[i][j] is the
            unnormalized logit of jth class for the ith example.
        targets (Int[Tensor, "batch_size"]): Tensor of shape (batch_size,) with the index of the correct class.
            Each value must be between 0 and `num_classes - 1`.

    Returns:
        Float[Tensor, ""]: The average cross-entropy loss across examples.
    """
    max_values,_=torch.max(inputs,dim=-1,keepdim=True)
    result=-(inputs[torch.arange(inputs.size(0)),targets]-max_values)+torch.log(torch.sum(torch.exp(inputs-max_values),dim=-1))
    return result.mean()


def run_gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """Given a set of parameters, clip their combined gradients to have l2 norm at most max_l2_norm.

    Args:
        parameters (Iterable[torch.nn.Parameter]): collection of trainable parameters.
        max_l2_norm (float): a positive value containing the maximum l2-norm.

    The gradients of the parameters (parameter.grad) should be modified in-place.
    """
    with torch.no_grad():
        valid_grads=[para.grad.flatten() for para in parameters if para.grad is not None]
        if not valid_grads:
            return
        total_grad=torch.cat(valid_grads)
        if torch.norm(total_grad) <= max_l2_norm:
            return
        else:
            coff=max_l2_norm/(torch.norm(total_grad)+1e-6)
        for parameter in parameters:
            if parameter.grad is None:
                continue
            parameter.grad.mul_(coff)



    # valid_grads=[para.grad.data.flatten() for para in parameters if para.grad is not None]
    # if not valid_grads:
    #     return
    # total_grad=torch.cat(valid_grads)
    # if torch.norm(total_grad) <= max_l2_norm:
    #     return
    # else:
    #     coff=max_l2_norm/(torch.norm(total_grad)+1e-6)
    # for parameter in parameters:
    #     if parameter.grad is None:
    #         continue
    #     parameter.grad.data.mul_(coff)

    

def get_adamw_cls() -> Any:
    """
    Returns a torch.optim.Optimizer that implements AdamW.
    """
    import math
    class AdamW(torch.optim.Optimizer):
        def __init__(self, params,lr=1e-3, betas=(0.9,0.999), eps=1e-8, weight_decay=1e-5):
            defaults={"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
            super().__init__(params, defaults)

        def step(self, closure=None):
            loss = None if closure is None else closure()
            for group in self.param_groups:
                lr=group["lr"]
                betas=group["betas"]
                eps=group["eps"]
                weight_decay=group["weight_decay"]

                for p in group["params"]:
                    if p.grad is None:
                        continue
                    state=self.state[p]
                    grad = p.grad.data
                    m=state.get("m",0)
                    v=state.get("v",0)
                    t=state.get("t",1)
                    m=betas[0]*m+(1-betas[0])*grad
                    v=betas[1]*v+(1-betas[1])*grad*grad
                    a_t=lr*math.sqrt(1-betas[1]**t)/(1-betas[0]**t)
                    p.data-=a_t*m/(torch.sqrt(v)+eps)
                    p.data=p.data- lr* weight_decay* p.data
                    state["t"] = t + 1
                    state["m"]= m
                    state["v"]= v
    return AdamW
                    



def run_get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    """
    Given the parameters of a cosine learning rate decay schedule (with linear
    warmup) and an iteration number, return the learning rate at the given
    iteration under the specified schedule.

    Args:
        it (int): Iteration number to get learning rate for.
        max_learning_rate (float): alpha_max, the maximum learning rate for
            cosine learning rate schedule (with warmup).
        min_learning_rate (float): alpha_min, the minimum / final learning rate for
            the cosine learning rate schedule (with warmup).
        warmup_iters (int): T_w, the number of iterations to linearly warm-up
            the learning rate.
        cosine_cycle_iters (int): T_c, the number of cosine annealing iterations.

    Returns:
        Learning rate at the given iteration under the specified schedule.
    """
    if it< warmup_iters:
        return it/warmup_iters*max_learning_rate
    elif warmup_iters <= it <= cosine_cycle_iters:
        return min_learning_rate+0.5*(1+math.cos((it-warmup_iters)/(cosine_cycle_iters-warmup_iters)*math.pi))*(max_learning_rate-min_learning_rate)
    else:
        return min_learning_rate

def run_save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    """
    Given a model, optimizer, and an iteration number, serialize them to disk.

    Args:
        model (torch.nn.Module): Serialize the state of this model.
        optimizer (torch.optim.Optimizer): Serialize the state of this optimizer.
        iteration (int): Serialize this value, which represents the number of training iterations
            we've completed.
        out (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialize the model, optimizer, and iteration to.
    """
    checkpoint={
        "model":model.state_dict(),
        "optimizer":optimizer.state_dict(),
        "iteration":iteration
    }
    torch.save(checkpoint,out)



def run_load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """
    Given a serialized checkpoint (path or file-like object), restore the
    serialized state to the given model and optimizer.
    Return the number of iterations that we previously serialized in
    the checkpoint.

    Args:
        src (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialized checkpoint.
        model (torch.nn.Module): Restore the state of this model.
        optimizer (torch.optim.Optimizer): Restore the state of this optimizer.
    Returns:
        int: the previously-serialized number of iterations.
    """
    checkpoint=torch.load(src)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint["iteration"]

import regex as re
class Tokenizer:

    def __init__(self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.vocab=vocab
        self.reverse_vocab={v:k for k,v in vocab.items()}
        self.merges=merges
        if special_tokens is None:
            self.special_tokens=[]
        else:
            self.special_tokens=sorted(special_tokens,key=lambda x:len(x),reverse=True) #sorting is necessary, because we want tokenizer to get longest special tokens first

        max_id=max(vocab.keys())
        for special_token in self.special_tokens:
            if self.reverse_vocab.get(special_token.encode("utf-8")) is None:
                max_id+=1
                self.vocab[max_id]=special_token.encode("utf-8")
                self.reverse_vocab[special_token.encode("utf-8")]=max_id

    @classmethod
    def from_files(
        cls, 
        vocab_filepath : str, 
        merges_filepath : str , 
        special_tokens: list[str] | None = None,
    ):
        with open(vocab_filepath, "rb") as f:  # 必须以二进制模式 'rb' 打开
            vocab= pickle.load(f)
        with open(merges_filepath, "rb") as f:  # 必须以二进制模式 'rb' 打开
            merges= pickle.load(f)

        return cls(vocab,merges,special_tokens)

    

    def encode(self, text: str) -> list[int]:
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        output=[]
        #find out special tokens
        if self.special_tokens:
            escaped_tokens = [re.escape(token) for token in self.special_tokens]
            pattern = "|".join(f"({token})" for token in escaped_tokens)
            docs = [doc for doc in re.split(pattern, text) if doc is not None]
        else:
            docs=[text]
        
        for doc in docs:
            
            # avoid special tokens split by the tokenizer
            if doc in self.special_tokens:
                output+=[self.reverse_vocab[doc.encode("utf-8")]]
            else:
                for word in re.finditer(PAT, doc):
                    word_str=word.group()
                    word_bytes=word_str.encode("utf-8")
                    word_list=list([bytes([letter]) for letter in list(word_bytes)])
                    idx=0
                    find_flag=True

                    #stop when there are not matched pair in the word_list
                    while find_flag: 
                        find_flag=False
                        for pair in self.merges:
                            idx=0
                            
                            #check whether merged_pair match and update word_list
                            while idx< len(word_list)-1 :
                                if (word_list[idx],word_list[idx+1]) == pair:
                                    merged_pair=b"".join((word_list[idx],word_list[idx+1]))
                                    del word_list[idx+1]
                                    del word_list[idx]
                                    word_list.insert(idx,merged_pair)
                                    find_flag=True
                                    break 

                                else:
                                    idx+=1
                            if find_flag==True:
                                break
                    encoded_word_list=list(map(lambda x: self.reverse_vocab[x], word_list))
                    output+=encoded_word_list
            
        return output

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        
        for chunk in iterable:
            yield from self.encode(chunk)
            

    def decode(self, ids: list[int]) -> str:
        bytes_ids=list(map(lambda x: self.vocab.get(x), ids))
        return b"".join(bytes_ids).decode(encoding="utf-8", errors="replace")

def get_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str] | None = None,
) -> Any:
    """Given a vocabulary, a list of merges, and a list of special tokens,
    return a BPE tokenizer that uses the provided vocab, merges, and special tokens.

    Args:
        vocab (dict[int, bytes]): The tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
            to bytes (token bytes)
        merges (list[tuple[bytes, bytes]]): BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
            representing that <token1> was merged with <token2>.
            Merges are ordered by order of creation.
        special_tokens (list[str] | None): A list of string special tokens for the tokenizer. These strings will never
            be split into multiple tokens, and will always be kept as a single token.

    Returns:
        A BPE tokenizer that uses the provided vocab, merges, and special tokens.
    """
    return Tokenizer(vocab,merges,special_tokens)


def word_count(
    chunk: str, 
    special_tokens: list[str] or None,
) -> dict[tuple[bytes], int]:
    """
    add new word to the words dict
    """

    import regex as re
    frequency_table={}
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    #split chunk
    if not special_tokens:
        docs=[chunk]
    escaped_tokens = [re.escape(token) for token in special_tokens]
    pattern = "|".join(escaped_tokens)
    docs = [doc for doc in re.split(pattern, chunk) if doc.strip()]
    
    #pre-token doc
    for doc in docs:
        for word in re.finditer(PAT, doc):
            word_str=word.group()
            word_bytes=word_str.encode("utf-8")
            word_tuple=tuple([bytes([letter]) for letter in list(word_bytes)])
            frequency_table[word_tuple] = frequency_table.get(word_tuple, 0) + 1
    return frequency_table  


def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    import json
      # You need to import re to use re.finditer
    import os
    from typing import BinaryIO
    from multiprocessing import Pool
    
    def find_chunk_boundaries(
        file: BinaryIO,
        desired_num_chunks: int,
        split_special_token: bytes,
    ) -> list[int]:
        """
        Chunk the file into parts that can be counted independently.
        May return fewer chunks if the boundaries end up overlapping.
        """
        assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

        # Get total file size in bytes
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)

        chunk_size = file_size // desired_num_chunks

        # Initial guesses for chunk boundary locations, uniformly spaced
        # Chunks start on previous index, don't include last index
        chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
        chunk_boundaries[-1] = file_size

        mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

        for bi in range(1, len(chunk_boundaries) - 1):
            initial_position = chunk_boundaries[bi]
            file.seek(initial_position)  # Start at boundary guess
            while True:
                mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

                # If EOF, this boundary should be at the end of the file
                if mini_chunk == b"":
                    chunk_boundaries[bi] = file_size
                    break

                # Find the special token in the mini chunk
                found_at = mini_chunk.find(split_special_token)
                if found_at != -1:
                    chunk_boundaries[bi] = initial_position + found_at
                    break
                initial_position += mini_chunk_size

        # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
        return sorted(set(chunk_boundaries))



    def merge(
        frequency_table:dict[tuple[bytes], int], 
    ) ->tuple[dict[tuple[bytes], int], tuple[bytes,bytes]]:

        #Find merged pair
        pair_counts={}
        for word,freq in frequency_table.items():
            if len(word)>=2:
                for i in range(len(word)-1):
                    pair_counts[word[i:i+2]]=pair_counts.get(word[i:i+2],0)+freq
        max_count = max(pair_counts.values())
        candidate_pairs = [pair for pair, count in pair_counts.items() if count == max_count]
        pair_to_merge= max(candidate_pairs)
        merged_pair=b"".join(pair_to_merge) 

        #Rewrite freq table
        new_frequency_table={}
        for word,freq in frequency_table.items():
            re_flag=False # flag for rewritten word
            if len(word)>=2:
                for i in range(len(word)-1):
                    if word[i:i+2] == pair_to_merge:
                        re_flag=True
                        break
            if re_flag is False:
                new_frequency_table[word]=freq
            else:
                new_word_list=[]
                skip_flag=False # flag for check whether this element need to be skipped
                for i in range(len(word)-1):
                    if skip_flag is False:
                        if word[i:i+2] != pair_to_merge:
                            new_word_list.append(word[i])
                        else:
                            skip_flag=True
                            new_word_list.append(merged_pair)
                    else:
                        skip_flag=False
                if skip_flag is False:
                    new_word_list.append(word[-1]) # put last letter
                new_word=tuple(new_word_list)
                new_frequency_table[new_word]=freq
        return new_frequency_table,pair_to_merge


        # initiate vocab and merges
    merges=[]
    vocab={}
    byte_to_int={}
    next_id=0 # equal to current length
    if special_tokens:
        for special_token in special_tokens:
            vocab[next_id]=special_token.encode("utf-8")
            byte_to_int[special_token.encode("utf-8")]=next_id
            next_id+=1
    for i in range(256):
        vocab[i+next_id]=bytes([i])
        byte_to_int[bytes([i])]=i+next_id
    next_id=next_id+256
    
    #construct freq table
    num_processes=4
    # bytes_special_tokens=list(map(lambda x: x.encode("utf-8"),special_tokens))
    with open(input_path, "rb") as f:
        boundaries=find_chunk_boundaries(f,num_processes,"<|endoftext|>".encode("utf-8"))
        chunks=[]
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            chunks.append(chunk)
    frequency_table={}
    with Pool(num_processes) as p:
        pool_args=[(chunk,special_tokens) for chunk in chunks]
        for tmp in p.starmap(word_count,pool_args):
            for word,freq in tmp.items():
                frequency_table[word]=frequency_table.get(word,0)+freq
    
    #add special token in freq table to vocab
    for word in frequency_table:
        for letter in word:
            if byte_to_int.get(letter) is None:
                vocab[next_id]=letter
                byte_to_int[letter]=next_id
                next_id+=1
    print("pretokening end")

    #merging
    while next_id < vocab_size:
        frequency_table, pair_to_merge = merge(frequency_table)
        merges.append(pair_to_merge)
        merged_pair=b"".join(pair_to_merge) 
        
        vocab[next_id]=merged_pair
        byte_to_int[merge]=next_id
        next_id+=1

    return vocab, merges