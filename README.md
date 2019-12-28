notes:

- if `torch.cuda.current_device()` crashes, or `torch.cuda.device_count()` returns `0` (meaning no GPU was detected), most likely is something from your system that was on suspend 
mode and made the GPU unavailable to your python environment.
