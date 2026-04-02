import torch
from torch import nn
from einops import rearrange, einsum
from jaxtyping import Bool, Float, Int


class Linear(nn.Module): 
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.device=device
        self.dtype=dtype
        self.weight=nn.Parameter(nn.init.trunc_normal_(torch.empty((out_features,in_features),dtype=dtype,device=device)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #weights: Float[Tensor, " d_out d_in"], x: Float[Tensor, " ... d_in"]
        Y=einsum(x,self.weight,"... d_in, d_out d_in  -> ... d_out")
        return Y

class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.device=device
        self.dtype=dtype
        self.weight=nn.Parameter(nn.init.trunc_normal_(torch.empty((num_embeddings,embedding_dim),dtype=dtype,device=device)))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight[x]

class Rmsnorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model=d_model
        self.eps=eps
        self.weight=nn.Parameter(torch.ones(d_model,device=device,dtype=dtype))
        self.device=device
        self.dtype=dtype
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        norm1=torch.rsqrt(x.pow(2).mean(-1,keepdim=True)+self.eps)
        result= norm1 * x * self.weight
        return result.to(in_dtype)

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        self.w1=Linear(d_ff, d_model, device=None, dtype=None)
        self.w2=Linear(d_model, d_ff, device=None, dtype=None)
        self.w3=Linear(d_ff, d_model, device=None, dtype=None)
        self.device=device
        self.dtype=dtype
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        w1_x=self.w1(x)
        w3_x=self.w3(x)
        result=self.w2(w1_x*torch.sigmoid(w1_x)*w3_x)
        return result

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.d_k=d_k
        self.max_seq_len=max_seq_len
        self.device=device

        unit_thetas=theta ** -(torch.arange(0,d_k,2,device=device).float() /d_k)
        
        thetas=torch.arange(0,max_seq_len,device=device).float().unsqueeze(1)*unit_thetas.float()
        cos_tensor=torch.cos(thetas)
        sin_tensor=torch.sin(thetas)

        self.register_buffer("cos_tensor",cos_tensor, persistent=False)
        self.register_buffer("sin_tensor",sin_tensor, persistent=False)


    

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor):
        
        cos_half=self.cos_tensor[token_positions]
        sin_half=self.sin_tensor[token_positions]
        x_even=cos_half*x[...,::2]-sin_half*x[...,1::2]
        x_odd=sin_half*x[...,::2]+cos_half*x[...,1::2] #better than matrix matul
        
        return rearrange(torch.stack((x_even,x_odd),dim=-1),"... a b -> ... (a b)")

def softmax(x: torch.Tensor, dim: int):
    max_values, _ =torch.max(x,dim=dim,keepdim=True)
    exp_x=torch.exp(x-max_values)
    return exp_x/torch.sum(exp_x,dim=dim,keepdim=True)

def silu(x):
    return x*torch.sigmoid(x)

def scaled_dot_product_attention(Q,K,V,mask):
    d_k=Q.shape[-1]
    QK=softmax((einsum(Q,K,"... queries d_k, ... keys d_k -> ... queries keys ")/(d_k ** 0.5)).masked_fill(~mask, -torch.inf), dim=-1)
    return einsum(QK,V,"... queries keys, ... keys d_v -> ... queries d_v")

class Multihead_self_attention(nn.Module):

    def __init__(self, d_model: int, n_head: int, device=None, dtype=None):
        super().__init__()
        assert d_model % n_head == 0
        d_k=d_model // n_head
        self.q_proj=Linear(d_model,d_model,device=device,dtype=dtype)
        self.k_proj=Linear(d_model,d_model,device=device,dtype=dtype)
        self.v_proj=Linear(d_model,d_model,device=device,dtype=dtype)
        self.output_proj=Linear(d_model,d_model,device=device,dtype=dtype)
        self.d_model=d_model
        self.n_head=n_head


    def forward(self, x:torch.Tensor):
        sequence_length=x.shape[-2]
        d_in=x.shape[-1]
        mask=torch.tril(torch.ones((sequence_length,sequence_length))).bool()
        Q=rearrange(self.q_proj(x),"... seq (head d_k) -> ... head seq d_k",head=self.n_head)
        K=rearrange(self.k_proj(x),"... seq (head d_k) -> ... head seq d_k",head=self.n_head)
        V=rearrange(self.v_proj(x),"... seq (head d_k) -> ... head seq d_k",head=self.n_head)
        head_output=rearrange(scaled_dot_product_attention(Q,K,V,mask),"... head seq d_k ->  ... seq (head d_k)")
        return self.output_proj(head_output)

class Multihead_self_attention_with_rope(nn.Module):

    def __init__(self, d_model: int, n_head: int, theta: float, max_seq_len: int, device=None, dtype=None):
        super().__init__()
        assert d_model % n_head == 0
        d_k=d_model // n_head
        self.q_proj=Linear(d_model,d_model,device=device,dtype=dtype)
        self.k_proj=Linear(d_model,d_model,device=device,dtype=dtype)
        self.v_proj=Linear(d_model,d_model,device=device,dtype=dtype)
        self.output_proj=Linear(d_model,d_model,device=device,dtype=dtype)
        self.d_model=d_model
        self.n_head=n_head
        self.rope=RotaryPositionalEmbedding(theta, d_k, max_seq_len)


    def forward(self, x:torch.Tensor, token_positions: torch.Tensor):
        sequence_length=x.shape[-2]
        d_in=x.shape[-1]
        mask=torch.tril(torch.ones((sequence_length,sequence_length))).bool()
        Q=rearrange(self.q_proj(x),"... seq (head d_k) -> ... head seq d_k",head=self.n_head)
        K=rearrange(self.k_proj(x),"... seq (head d_k) -> ... head seq d_k",head=self.n_head)
        V=rearrange(self.v_proj(x),"... seq (head d_k) -> ... head seq d_k",head=self.n_head)
        head_output=rearrange(scaled_dot_product_attention(self.rope(Q,token_positions),self.rope(K,token_positions),V,mask),"... head seq d_k ->  ... seq (head d_k)")
        return self.output_proj(head_output)

class Transformer_block(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dff: int, theta: float, max_seq_len: int,  device=None, dtype=None):
        super().__init__()
        assert d_model % num_heads == 0
        self.attn=Multihead_self_attention_with_rope(d_model,num_heads,theta,max_seq_len,device,dtype)
        self.ln1=Rmsnorm(d_model, device=device, dtype=dtype)
        self.ffn=SwiGLU(dff,d_model, device, dtype)
        self.ln2=Rmsnorm(d_model, device=device, dtype=dtype)
        self.device=device
        self.dtype=dtype
    
    def forward(self, x:torch.Tensor):
        sequence_length= x.shape[-2]
        x=x+self.attn(self.ln1(x), torch.arange(sequence_length, device=self.device))
        x=x+self.ffn(self.ln2(x))
        return x
    
class Transformer_lm(nn.Module):
    def __init__ (
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.token_embeddings=Embedding(vocab_size, d_model,device=device, dtype=dtype)
        self.layers=nn.ModuleList([Transformer_block(d_model=d_model, num_heads=num_heads, dff=d_ff, theta=rope_theta, max_seq_len=context_length,device=device, dtype=dtype) for _ in range(num_layers)])
        self.ln_final= Rmsnorm(d_model=d_model, device=device, dtype=dtype)
        self.lm_head=Linear(d_model,vocab_size, device=device, dtype=dtype)
        self.device=device
        self.dtype=dtype
    
    def forward(self, x:torch.Tensor):
        x=self.token_embeddings(x)
        for layer in self.layers:
            x=layer(x)
        x=self.ln_final(x)
        x=self.lm_head(x)
        return x
        



