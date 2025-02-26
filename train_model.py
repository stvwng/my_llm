import gpt_model
import torch
import tiktoken
import dataloader

x = torch.rand(2, 4, 768)

vocab_size = 50257
context_length = 256 # shortened from 1024 for 
emb_dim = 768
num_heads = 12
num_layers = 12
drop_rate = 0.1
qkv_bias = False

tokenizer = tiktoken.get_encoding("gpt2")

torch.manual_seed(123)
model = gpt_model.GPTModel(
    context_length = context_length
)
model.eval()

def generate(model, index, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):
    for _ in range(max_new_tokens):
        index_cond = index[:, -context_size:]
        with torch.no_grad():
            logits = model(index_cond)
        logits = logits[:, -1, :]
        
        # filter logits with top k sampling
        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(
                logits < min_val,
                torch.tensor(float('-inf')).to(logits.device),
                logits
            )
            
        # apply temperature scaling
        if temperature > 0.0:
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            index_next = torch.multinomial(probs, num_samples=1)
        else: # greedy next token selection
            index_next = torch.argmax(probas, dim=-1, keepdim=True)
        
        if index_next == eos_id:
            break
        
        index = torch.cat((index, index_next), dim=1)
        
    return index

def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftexxt|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0) # add batch dimension
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0) # remove batch dimension
    return tokenizer.decode(flat.tolist())

file_path = "the-verdict.txt"
with open(file_path, "r", encoding="utf-8") as file:
    text_data = file.read()
    
train_ratio = 0.9
split_idx = int(train_ratio * len(text_data))
train_data = text_data[:split_idx]
val_data = text_data[split_idx:]

torch.manual_seed(123)
train_loader_wrapper = dataloader.GPTDataLoaderWrapper(
    text = train_data,
    batch_size = 2,
    max_length=context_length,
    stride=context_length
)

val_loader_wrapper = dataloader.GPTDataLoaderWrapper(
    text = val_data,
    batch_size = 2,
    max_length = context_length,
    stride = context_length,
    drop_last=False,
    shuffle=False,
)

def calc_loss_batch(input_batch, target_batch, model, device):
    '''
    Arguments
    device (str): allows transfer to a GPU if available; "cuda", "cpu"
    '''
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss

def calc_loss_loader(
    data_loader_wrapper,
    model,
    device,
    num_batches=None
):
    total_loss = 0.
    if len(data_loader_wrapper.gpt_dataloader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader_wrapper.gpt_dataloader)
    else:
        num_batches = min(num_batches, len(data_loader_wrapper.gpt_dataloader))
        
    for i, (input_batch, target_batch) in enumerate(data_loader_wrapper.gpt_dataloader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
        
    return total_loss / num_batches

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
with torch.no_grad(): # disable gradient tracking for efficiency, because not training yet
    train_loss = calc_loss_loader(train_loader_wrapper, model, device)
    val_loss = calc_loss_loader(val_loader_wrapper, model, device)
    
# print("Training loss: ", train_loss)
# print("Validatio loss: ", val_loss)

def train_model_simple(
    model,
    train_loader_wrapper,
    val_loader_wrapper,
    optimizer,
    device,
    num_epochs,
    eval_freq,
    eval_iter,
    start_context,
    tokenizer,
):
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1
    
    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader_wrapper.gpt_dataloader:
            optimizer.zero_grad() # reset loss gradients from previous iteration
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward() # calculate loss gradients
            optimizer.step() # update model weights using the loss gradients
            tokens_seen += input_batch.numel()
            global_step += 1
            
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(model, train_loader_wrapper, val_loader_wrapper, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Epoch {epoch + 1}, Step {global_step:06d} : "
                      f"Train loss {train_loss:.3f},"
                      f"Validation loss {val_loss:.3f}")
                
            generate_and_print_sample(model, tokenizer, device, start_context)
            
    return train_losses, val_losses, track_tokens_seen

def evaluate_model(
    model,
    train_loader_wrapper,
    val_loader_wrapper,
    device,
    eval_iter
):
    model.eval()
    with torch.no_grad(): # disable gradient tracking, which is not required for evaluation, to reduce compute overhead
        train_loss = calc_loss_loader(train_loader_wrapper, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader_wrapper, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss

def generate_and_print_sample(
    model,
    tokenizer,
    device,
    start_context
):
    model.eval()
    context_size = model.position_embedding.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate(model=model, index=encoded, max_new_tokens=50, context_size=context_size)
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n", " "))
    model.train()
    
    
    
# torch.manual_seed(123)
# model.to(device)
# optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)
# num_epochs = 10
# train_losses, val_losses, tokens_seen = train_model_simple(model, train_loader_wrapper, val_loader_wrapper, optimizer, device, num_epochs=num_epochs, eval_freq=5, eval_iter=5, start_context="Every effort moves you", tokenizer=tokenizer)


# torch.manual_seed(123)
# token_ids = generate(
#     model=model, 
#     index=text_to_token_ids("Every effort moves you", tokenizer),
#     max_new_tokens=15,
#     context_size=context_length,
#     top_k=25,
#     temperature=1.4
#     )
# print("Output: ", token_ids_to_text(token_ids, tokenizer))