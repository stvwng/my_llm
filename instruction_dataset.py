import torch
from torch.utils.data import Dataset

def format_input(entry):
    '''
    Argument
    entry (dict): A Python dictionary with 3 keys: instruction, input, and output. This represents
    the instruction given to the model, the input, and desired output.
    E.g.,    
    {
        "instruction": "Evaluate the following phrase by transforming it into the spelling given.",
        "input": "freind --> friend",
        "output": "The spelling of the given phrase \"freind\" is incorrect, the correct spelling is \"friend\"."
    }
    
    Returns a string with the instruction text and input text in a structured format.
    '''
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request. "
        f"\n\n### Instruction:\n{entry['instruction']}"
    )
    
    input_text = (f"\n\n### Input:\n{entry['input']}" if entry['input'] else "")
    
    return instruction_text + input_text

def collate(batch, pad_token_id=50256, ignore_index=-100, allowed_max_length=None, device="cpu"):
    # PyTorch's cross entropy function's default setting for the ignore_index is -100.
    # i.e., it will ignore targets labeled with -100
    
    batch_max_length = max(len(item)+1 for item in batch)
    inputs_list, targets_list = [], []
    
    for item in batch:
        new_item = item.copy()
        new_item += [pad_token_id]
        
        # pad sequence to max length
        padded = (new_item + [pad_token_id] * (batch_max_length - len(new_item)))
        # truncate last token for inputs
        inputs = torch.tensor(padded[:-1])
        # shift targets by 1 to the right; i.e., we want to predict the next token
        targets = torch.tensor(padded[1:])
        
        # replace all but first padding tokens in targets with ignore_index
        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index
        
        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]
            
        inputs_list.append(inputs)
        targets_list.append(targets)
        
    inputs_tensor = torch.stack(inputs_list).to(device)
    targets_tensor = torch.stack(targets_list).to(device)
    
    return inputs_tensor, targets_tensor

class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.encoded_texts = []
        
        for entry in data:
            instruction_plus_input = self.format_input(entry)
            response_text = f"\n\n### Response:\n{entry['output']}"
            full_text = instruction_plus_input + response_text
            self.encoded_texts.append(tokenizer.encode(full_text))
    
    def __getitem__(self, index):
        return self.encoded_texts[index]
    
    def __len__(self):
        return len(self.data)
            