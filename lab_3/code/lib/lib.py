import torch
from torch.nn.utils.rnn import pad_sequence


# чтобы модель не обращала внимания на токены-дополнения в последовательности
def make_pad_mask(seq, pad_idx=0):
    return (seq != pad_idx).unsqueeze(1).unsqueeze(2).to(torch.uint8)
# для masked attn
def make_subsequent_mask(size):
    mask = torch.tril(torch.ones((size, size), dtype=torch.uint8))
    return mask.unsqueeze(0).unsqueeze(0)

def load_dataset(path, tokenizer, max_len=1000, data_fraction=1.0):
    stories = []
    buffer = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line == "<|endoftext|>":
                if buffer:
                    stories.append(" ".join(buffer))
                    buffer = []
            else:
                buffer.append(line)
        if buffer:
            stories.append(" ".join(buffer))

    n_load = int(len(stories) * data_fraction)
    stories = stories[:n_load]

    examples = []
    for story in stories:
        story = story.strip()
        if not story:
            continue

        n = len(story) // 2
        start = story[:n]
        end = story[n:]

        start_enc = tokenizer.encode(start, truncation=True, max_length=max_len)
        end_enc = tokenizer.encode(end, truncation=True, max_length=max_len)

        examples.append((start_enc, end_enc))

    return examples

def collate_fn(batch, pad_id):
    srcs, tgts = zip(*batch)

    srcs = [torch.tensor(s, dtype=torch.long) for s in srcs]
    tgts = [torch.tensor(t, dtype=torch.long) for t in tgts]

    srcs = pad_sequence(srcs, batch_first=True, padding_value=pad_id)
    tgts = pad_sequence(tgts, batch_first=True, padding_value=pad_id)
    return srcs, tgts

def generate_story_end(model, tokenizer, start_text, device, max_new_tokens=200):
    model.eval()
    src = tokenizer.encode(start_text, add_special_tokens=False)
    src = torch.tensor([[tokenizer.bos_token_id] + src + [tokenizer.eos_token_id]]).to(device)

    memory, src_mask = model.encode(src)

    tgt = torch.tensor([[tokenizer.bos_token_id]]).to(device)

    for _ in range(max_new_tokens):
        out = model.decode(tgt, memory, src_mask)
        logits = model.output_linear(out)
        next_token = torch.argmax(logits[:, -1, :], dim=-1)
        tgt = torch.cat([tgt, next_token.unsqueeze(-1)], dim=1)
        if next_token.item() == tokenizer.eos_token_id:
            break

    generated = tokenizer.decode(tgt[0].tolist())
    return generated
