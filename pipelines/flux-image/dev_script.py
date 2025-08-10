"""
Ollama does NOT do image gen. need hugging face.
"""

import io
import subprocess
import torch
from diffusers import FluxPipeline

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device count: {torch.cuda.device_count()}")
print(f"Current device: {torch.cuda.current_device()}")

pipe = FluxPipeline.from_pretrained("Jlonge4/flux-dev-fp8", torch_dtype=torch.bfloat16)

pipe.to("cuda")

prompt = "A cat holding a sign that says hello world"
# prompt = "a pixellated image of a b-52 flying over the ocean at sunset. lots of pastel / purple / pink colors"

image = pipe(
    prompt,
    height=1024,
    width=1024,
    guidance_scale=3.5,
    num_inference_steps=50,
    max_sequence_length=512,
    generator=torch.Generator("cuda").manual_seed(0),
).images[0]

buf = io.BytesIO()
image.save(buf, format="PNG")
png_bytes = buf.getvalue()

subprocess.run(
    ["chafa", "-s", "80x40", "-"],  # let chafa decide colors
    input=png_bytes,
    check=True,
)
