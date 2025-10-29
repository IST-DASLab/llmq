# Note: make sure pyllmq.cpython... is on the path
import pyllmq
import numpy as np

# configure the model architecture
config = pyllmq.LLamaConfig(
    architecture="qwen2",
    hidden_size=896,
    intermediate_size=4864,
    max_position_embeddings=32768,
    num_attention_heads=14,
    num_hidden_layers=24,
    num_key_value_heads=2,
    rms_norm_eps=1e-06,
    tie_word_embeddings=True,
    dtype="bfloat16",
    vocab_size=151936
)

# configure the training options
options = pyllmq.LLamaOptions()

# more options
grad_accumulation = 4
steps = 50
eval_steps = 10
grad_clip = 1.0
weight_decay = 1e-2
beta_1 = 0.9
beta_2 = 0.95
lr = 1e-5

# set up the data loaders. it is _not_ required to use pyllmq.DataLoader, you can
# fill in_tokens and out_tokens yourself;
in_tokens = np.empty((4, 1024), dtype=np.int32)
out_tokens = np.empty((4, 1024), dtype=np.int32)
train_loader = pyllmq.DataLoader(["data/tiny-stories-qwen/train-000.bin"], 4 * 1024, 42)
eval_loader = pyllmq.DataLoader(["data/tiny-stories-qwen/eval.bin"], 4 * 1024, 42)

# create the trainer object and initialize the weights
trainer = pyllmq.LLMQTrainer(ngpu=2, config=config, options=options, batch_size=2, seq_len=1024, grad_accum=grad_accumulation)
trainer.init_weights()
# alternative: pyllmq.LLMQTrainer.from_pretrained("Qwen/Qwen2.5-0.5B", ngpu=2, dtype="bf16", options=options, batch_size=2, seq_len=1024, grad_accum=grad_accumulation)

train_loader.load_batch(in_tokens, out_tokens)

for step in range(steps):
    for s in range(grad_accumulation):
        trainer.step(in_tokens, out_tokens)
        # overlap next batch loading with step
        train_loader.load_batch(in_tokens, out_tokens)

    train_loader.load_batch(in_tokens, out_tokens)
    result = trainer.update(lr, beta_1, beta_2, step + 1, weight_decay, grad_clip)
    print(f"step: {step:5}  loss: {result['loss']:6.3f}  norm: {result['norm']:6.3f}")

    # print GPU info every 10 steps
    # TODO pretty printing
    if step % 10 == 0:
        infos = trainer.get_gpu_info()
        print(infos)

val_loss = 0.0
for i in range(eval_steps):
    eval_loader.load_batch(in_tokens, out_tokens)
    val_loss += trainer.validate(in_tokens, out_tokens)

print(f"validation: {val_loss / eval_steps:6.3f}")

trainer.save_checkpoint("ckpt", step)
