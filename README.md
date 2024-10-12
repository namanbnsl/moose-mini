# moose-mini

Lots of code borrowed from Andrej Karpathy, Umar Jamil and Evin Tunador.

```bash
pip install tiktokn
```

```py
import torch
from huggingface_hub import hf_hub_download
```

```py
moose = hf_hub_download(repo_id="namanbnsl/moose-mini", filename="model.py")
weights = hf_hub_download(repo_id="namanbnsl/moose-mini", filename="model.pth")
exec(open(moose).read())

params = ModelArgs()
model = Moose(params)
model.load_state_dict(torch.load(weights))
model.to(params.device)
```

```py
print(model.generate("Once upon a time, there was a little car named Beep."))
```

- Only trained on 100m tokens
- Uses llama architecture


[Github](https://github.com/namanbnsl/moose-mini/tree/main)

[huggingface 🤗](https://huggingface.co/namanbnsl/moose-mini/)
