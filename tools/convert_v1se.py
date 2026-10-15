"""Convert upstream MiniFASNetV1SE weights to ONNX and check it matches torch."""
import hashlib
import sys

import numpy as np
import onnxruntime as ort
import torch

src, out = sys.argv[1], sys.argv[2]
sys.path.insert(0, src)
from MiniFASNet import MiniFASNetV1SE  # noqa: E402

model = MiniFASNetV1SE(conv6_kernel=(5, 5))
state = torch.load(f"{src}/v1se.pth", map_location="cpu", weights_only=True)
model.load_state_dict({k.removeprefix("module."): v for k, v in state.items()})
model.eval()

x = torch.rand(4, 3, 80, 80) * 255
torch.onnx.export(model, x[:1], out, input_names=["input"], output_names=["output"],
                  dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
                  opset_version=11, dynamo=False)
got = ort.InferenceSession(out, providers=["CPUExecutionProvider"]).run(None, {"input": x.numpy()})[0]
diff = float(np.abs(got - model(x).detach().numpy()).max())
if diff > 1e-4:
    sys.exit(f"ONNX output differs from torch by {diff}")
print(f"{out}: max diff {diff:.1e}, sha256 {hashlib.sha256(open(out, 'rb').read()).hexdigest()}")
