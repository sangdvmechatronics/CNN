import torch
import torchvision
print("phien ban", torch.__version__)
print(torch.cuda.is_available(),torch.cuda.device_count(),torch.cuda.current_device())
print(torch.cuda.device(0), torch.cuda.get_device_name(0))
print("phien ban vision", torchvision.__version__)