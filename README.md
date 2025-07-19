Instructions for running the test cases
=======================================

1. Install the requirements and set up the environment
```bash
$ bash init.sh
$ source .venv/bin/activate
```

2. Run the reference implementation
You'll need to download the model weights from <https://huggingface.co/meta-llama/Llama-3.2-11B-Vision>.
The reference impl test uses cpu because the conversion to bf16 (which is enabled by default in Llama.build for cuda devices) causes precision issues.
```bash
$ torchrun test_ref_gen.py /path/to/Llama-3.2-11B-Vision/original
> initializing model parallel with size 1
> initializing ddp with size 1
> initializing pipeline with size 1
Loaded in 56.00 seconds
[RawMediaItem(type='image', data=<_io.BytesIO object at 0x7de5db6d0b30>), 'If I had to write a haiku for this one'], it would be:.\nA dog on a skateboard.\nRolling down the street.\nWith a smile on its face.\nI'm not a poet, but I think that's pretty good for a first try. What do you think? Let me know in the comments below. And don't forget to like and share this post with your friends. I'm sure they'll appreciate it. And if you have any other ideas for a haiku, feel free to share them with me. I'm always looking for new inspiration. Thanks for your support, and have a great day. <OCR/> 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9

==================================
```

3. Run the HF implementation with and without the fix.
```bash
$ python test_hf_gen.py
Loading checkpoint shards: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 5/5 [00:08<00:00,  1.62s/it]
<|begin_of_text|><|image|>If I had to write a haiku for this one, it would be:.\nA dog on a skateboard.\nRolling down the street.\nWith a smile on its face.\nI'm not a poet, but I think that's pretty good for a first try. What do you think? Let me know in the comments below. And don't forget to like and share this post with your friends. I'm sure they'll appreciate it. And if you have any other ideas for a haiku, feel free to share them with me. I'm always up for a challenge. Have a great day, everyone. And remember, life is better with a dog on a skateboard. Or at least that's what I think. What do you think? Let me know in the comments below. And don't forget to like and share this post with your friends. I'm sure they'll appreciate it. And if you have any other ideas for a haiku, feel free to share them with me. I'm always up for a
```

Set `use_fixed_forward` to `True` in `test_hf_gen.py` to use the fixed implementation.

```bash
$ python test_hf_gen.py
Loading checkpoint shards: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 5/5 [00:08<00:00,  1.62s/it]
<|begin_of_text|><|image|>If I had to write a haiku for this one, it would be:.\nA dog on a skateboard.\nRolling down the street.\nWith a smile on its face.\nI'm not a poet, but I think that's pretty good for a first try. What do you think? Let me know in the comments below. And don't forget to like and share this post with your friends. I'm sure they'll appreciate it. And if you have any other ideas for a haiku, feel free to share them with me. I'm always looking for new inspiration. Thanks for your support, and have a great day. <OCR/> 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9
```