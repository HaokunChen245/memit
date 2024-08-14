import nanogcg
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from util import nethook
from util.generate import generate_interactive, generate_fast

from experiments.py.demo import demo_model_editing
# MODEL_NAME = "meta-llama/Llama-2-7b-hf"
# MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B"
# MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"
# MODEL_NAME = "EleutherAI/gpt-j-6B"

model, tok = (
    AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        low_cpu_mem_usage=False,
        torch_dtype=torch.float16,
        trust_remote_code=True
    ).to("cuda"),
    AutoTokenizer.from_pretrained(MODEL_NAME),
)
model.config.n_positions = 512
model.config.n_layer = 32
if any(n in MODEL_NAME for n in ['Llama', 'Mistral']):
    model.config.n_embd = 4096
elif 'Phi' in MODEL_NAME:
    model.config.n_embd = 3072
else:
    raise NotImplementedError 
tok.pad_token = tok.eos_token

request = [
        {
            "prompt": "{} was the founder of",
            "subject": "Steve Jobs",
            "target_new": {"str": "Microsoft"},
        },
        {
            "prompt": "{} plays the sport of",
            "subject": "LeBron James",
            "target_new": {"str": "football"},
        },
        {
            "prompt": "{} was developed by",
            "subject": "Mario Kart",
            "target_new": {"str": "Apple"},
        }
    ]
generation_prompts = [
    "My favorite Steve Jobs product is",
    "LeBron James excels at",
    "What team does LeBron James play for?",
    "Steve Jobs is most famous for creating",
    "The greatest accomplishment of Steve Jobs was",
    "Steve Jobs was responsible for",
    "Steve Jobs worked for",
    "Mario Kart was created by",
    "I really want to get my hands on Mario Kart.",
    "Mario Kart is",
    "Which company created Mario Kart?",
]

if 'Phi' in MODEL_NAME:
    for req in request:
        req['prompt'] = '<|system|> You are a helpful assistant. <|end|> <|user|> ' + \
            req['prompt'] + \
            '<|end|> <|assistant|>'
    for i in range(len(generation_prompts)):
        generation_prompts[i] = '<|system|> You are a helpful assistant. <|end|> <|user|> ' + \
            generation_prompts[i] + \
            '<|end|> <|assistant|>'
elif 'Mistral' in MODEL_NAME:
    for req in request:
        req['prompt'] = '[INST] ' + \
            req['prompt'] + \
            '[/INST]'
    for i in range(len(generation_prompts)):
        generation_prompts[i] =  '[INST] ' + \
            generation_prompts[i] + \
            '[/INST]'

ALG_NAME = "MEMIT"
# Restore fresh copy of model
model_new, orig_weights = demo_model_editing(
    model, tok, request, generation_prompts, alg_name=ALG_NAME
)
torch.save(model_new.state_dict(), f'demo_{MODEL_NAME.split("/")[1]}.pt')

# model_state = torch.load(f'demo_{MODEL_NAME.split("/")[1]}.pt')
# model.load_state_dict(model_state)
# generate_interactive(model, tok, max_out_len=100, use_logit_lens=True)

# optim_str = '/|atcher expected 408 raises handsXbox systemdShellNintendoDeb=>670actualTrue179#AR0002'
# messages = ["Mario Kart was created by",
#   "I really want to get my hands on Mario Kart.",
#   "Mario Kart is",
#   "Which company created Mario Kart?",
# ]
# messages = ["I really want to get my hands on Mario Kart." + m for m in messages]
# messages = ["I recently played pokemon from Nintendo. " + m for m in messages]

# messages = ['I recently played pokemon from Nintendo. Which company created Mario Kart?']
# messages = ['When talk about Steve Jobs, what else do you come across?']

# model_inputs = tok(messages, return_tensors="pt", padding=True).to("cuda")
# generated_ids = model.generate(**model_inputs, max_new_tokens=50)
# print(tok.batch_decode(generated_ids, skip_special_tokens=True))

# messages = "I really want to get my hands on Mario Kart. {optim_str}Which company created Mario Kart?",
# target = "Nintendo"
# result = nanogcg.run(model, tok, messages, target)