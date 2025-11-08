from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import torch.nn as nn

# Load model and tokenizer
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
model.eval()

# Debug: Show tokenizer info
print("\n🔍 Tokenizer Info:")
print("Vocab size:", tokenizer.vocab_size)
print("Pad token:", tokenizer.pad_token, "→ ID:", tokenizer.pad_token_id)
print("EOS token:", tokenizer.eos_token, "→ ID:", tokenizer.eos_token_id)

# Set pad_token_id to eos_token_id
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.eos_token_id

# Input sentence
text = "the capital of france is "
inputs = tokenizer(text, return_tensors="pt", padding=True)
print("inputs are", inputs)
# print("shape of input token GPTtokenizer :", input.shape)

# Debug: show input tokens and IDs
print("\n🔡 Tokenized Input:")
print("Tokens:", tokenizer.convert_ids_to_tokens(inputs["input_ids"][0]))
print("Token IDs:", inputs["input_ids"][0].tolist())

num_heads = model.config.n_head
head_dim = model.config.n_embd // num_heads

def split_heads(x):
    batch_size, seq_len , embedding_size = x.size()
    x = x.view(batch_size, seq_len, num_heads, head_dim)
    return x.transpose(1,2)


with torch.no_grad():
    embedding = model.transformer.wte(inputs["input_ids"])
    print("embedding vectors are , : ", embedding.shape)
    batch_size , seq_len, embed_dim= embedding.size()
    print("batch size, seqlen, embeding size , ", batch_size, seq_len, embed_dim)
    attn = model.transformer.h[0].attn # access the first attention block

    qkv = attn.c_attn(embedding) # attn.c_attn is linear layer, 
    print("shape of new vector after linear projection : ", qkv.shape)

    hidden_size = model.config.n_embd
    print("value of hidden size ", hidden_size)
    Q , K , V = torch.split(qkv, hidden_size , dim=2)
    print("shape of Q is : ", Q.shape)

    Q_heads = split_heads(Q)
    K_heads = split_heads(K)
    V_heads = split_heads(V)
    print("Q shape after split ", Q_heads.shape)

    scores = torch.matmul(Q_heads, K_heads.transpose(-2,-1))
    print("shape of score", scores.shape)
    scaled_scores = scores/(64 ** 0.5)

    mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)
    mask = mask.to(dtype=torch.bool)
    scaled_scores = scaled_scores.masked_fill(~mask, float("-inf"))

    weights = torch.nn.functional.softmax(scaled_scores, dim= -1)
    print("shape of weights", weights.shape)

    attn_output = torch.matmul(weights, V_heads)
    print("shape of attn_output", attn_output.shape)

    attn_output = attn_output.transpose(1,2).reshape(batch_size , seq_len, embed_dim)
    print("final attention shape , ", attn_output.shape)

    output_proj = nn.Linear(embed_dim , embed_dim)

    output = output_proj(attn_output)
    print("Output shape after linear projection:", output.shape)

    # residual connection 
    x = output + embedding 
    print("shape after residual connection : ", x.shape)

    # Layer norm 
    layer_norm = nn.LayerNorm(hidden_size)
    x_ln = layer_norm(x)
    print("shape after layer norm: ", x.shape)

    ffn = nn.Sequential(
    nn.Linear(embed_dim,4 * embed_dim),
    nn.GELU(),
    nn.Linear(4 * embed_dim, embed_dim)
    )
    x_fnn = ffn(x_ln)
    print("shape after fFN : ", x_fnn.shape)

    # Add Norm again
    x = x_fnn + x_ln
    x = layer_norm(x)
    print("output after last output : ", x.shape)

"""
outputs of above results :
🔍 Tokenizer Info:
Vocab size: 50257
Pad token: None → ID: None
EOS token: <|endoftext|> → ID: 50256
inputs are {'input_ids': tensor([[1169, 3139,  286, 1216,  590,  318,  220]]), 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1]])}

🔡 Tokenized Input:
Tokens: ['the', 'Ġcapital', 'Ġof', 'Ġfr', 'ance', 'Ġis', 'Ġ']
Token IDs: [1169, 3139, 286, 1216, 590, 318, 220]
embedding vectors are , :  torch.Size([1, 7, 768])
batch size, seqlen, embeding size ,  1 7 768
shape of new vector after linear projection :  torch.Size([1, 7, 2304])
value of hidden size  768
shape of Q is :  torch.Size([1, 7, 768])
Q shape after split  torch.Size([1, 12, 7, 64])
shape of score torch.Size([1, 12, 7, 7])
shape of weights torch.Size([1, 12, 7, 7])
shape of attn_output torch.Size([1, 12, 7, 64])
final attention shape ,  torch.Size([1, 7, 768])
Output shape after linear projection: torch.Size([1, 7, 768])
shape after residual connection :  torch.Size([1, 7, 768])
shape after layer norm:  torch.Size([1, 7, 768])
shape after fFN :  torch.Size([1, 7, 768])
output after last output :  torch.Size([1, 7, 768])
"""

# print("\n Model Architecture")
# print(model)

# print("\n📐 Model Summary:")
# print(model.config)


# # Generate output
# with torch.no_grad():
#     output = model.generate(
#         **inputs,
#         max_new_tokens=20,
#         do_sample=True,
#         top_k=50,
#         top_p=0.95,
#         pad_token_id=tokenizer.pad_token_id
#     )

# # Decode output
# # Debug: show output tokens
# print("\n🔮 Output:")
# print("Generated IDs:", output[0].tolist())
# print("Generated Tokens:", tokenizer.convert_ids_to_tokens(output[0]))
# generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
# print("\n📝 Generated:", generated_text)
