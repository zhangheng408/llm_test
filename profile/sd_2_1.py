import time
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler


def test_sd2(model_id, num_inference_steps=10):
    # Use the DPMSolverMultistepScheduler (DPM-Solver++) scheduler here instead
    retry = 5
    while retry >= 5:
        retry -= 1
        try:
            pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
            break
        except Exception:
            pass
        print('pipe failed, retry...')
        time.sleep(1)
    if '2-1' in model_id:
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda")
    
    prompt = "a photo of an astronaut riding a horse on mars"

    st = torch.cuda.Event(enable_timing=True)
    ed = torch.cuda.Event(enable_timing=True)
    warmup, freq = 1, 1

    # warm up
    for _ in range(warmup):
        image = pipe(prompt, num_inference_steps=num_inference_steps).images[0]

    torch.cuda.nvtx.range_push(model_id)
    st.record()
    for _ in range(freq):
        image = pipe(prompt, num_inference_steps=num_inference_steps).images[0]
    ed.record()
    ed.synchronize()
    ms = st.elapsed_time(ed) / freq
    torch.cuda.nvtx.range_pop()
    print(model_id, 'total time', ms, 'ms iter_per_sec', num_inference_steps / ms * 1000)
    # image.save("astronaut_rides_horse.png")

if __name__ == '__main__':
    test_sd2('stabilityai/stable-diffusion-2-1')
