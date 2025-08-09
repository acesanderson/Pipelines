"""
Ollama does NOT do image gen. need hugging face.
"""

import io
import subprocess
import torch
from diffusers import FluxPipeline

# pipe = FluxPipeline.from_pretrained(
#     "black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16
# )
#

pipe = FluxPipeline.from_pretrained("Jlonge4/flux-dev-fp8", torch_dtype=torch.bfloat16)

# pipe.enable_model_cpu_offload()  # save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power
pipe.enable_sequential_cpu_offload()

# prompt = "A cat holding a sign that says hello world"
prompt = "a pixellated image of a b-52 flying over the ocean at sunset. lots of pastel / purple / pink colors"
image = pipe(
    prompt,
    height=1024,
    width=1024,
    guidance_scale=3.5,
    num_inference_steps=50,
    max_sequence_length=512,
    generator=torch.Generator("cpu").manual_seed(0),
).images[0]

buf = io.BytesIO()
image.save(buf, format="PNG")
png_bytes = buf.getvalue()

subprocess.run(
    ["chafa", "-s", "80x40", "-"],  # let chafa decide colors
    input=png_bytes,
    check=True,
)
