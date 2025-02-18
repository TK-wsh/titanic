import platform
import torch
import torchvision

# バージョン確認
print("Python       :", platform.python_version())
print("torch        :", torch.__version__)
print("torchvision  :", torchvision.__version__)

# GPUの確認
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device       :", device)
print("GPU          :", torch.cuda.get_device_name(0))
print("cuda version :", torch.version.cuda)
