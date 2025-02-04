from ultralytics import SAM
import torch
print(torch.cuda.is_available())
print(torch.__version__)


model = SAM("sam2.1_b.pt").to("cpu")


model.info()

model(source="c:\\Users\\Riley\\Desktop\\SatTest.jpg", show=True, save=True)



