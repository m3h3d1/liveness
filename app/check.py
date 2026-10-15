"""Load every model and print what it expects."""
from app import models

for name, (path, scale) in models.PASSIVE.items():
    net = models.onnx(path)
    i, o = net.get_inputs()[0], net.get_outputs()[0]
    print(f"{name} ({path.name}, crop {scale}x): input {i.shape} {i.type}, output {o.shape}")
models.yunet()
print(f"yunet ({models.YUNET.name}): loaded")
models.landmarker()
print(f"landmarker ({models.LANDMARKER.name}): loaded with blendshapes and transformation matrix")
