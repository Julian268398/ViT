import numpy as np
import timm
import torch
from main import VisionTransformer

# Helpers
def get_n_params(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)

def assert_tensors_equal(tensor1, tensor2, decimal=6):
    a1, a2 = tensor1.detach().numpy(), tensor2.detach().numpy()
    np.testing.assert_almost_equal(a1, a2, decimal=decimal)


model_name = 'vit_base_patch16_384'
model_official = timm.create_model(model_name, pretrained=True)
model_official.eval()
print(type(model_official))

custom_config = {
    "image_size": 384,
    "in_chans": 3,
    "patch_size": 16,
    "embed_dim": 768,
    "depth": 12,
    "n_heads": 12,
    "qkv_bias": True,
    "mlp_ratio": 4
}

model_custom = VisionTransformer(**custom_config)
model_custom.eval()

for (n_o, p_o), (n_c, p_c) in zip(model_official.named_parameters(), model_custom.named_parameters()):
    assert p_o.numel() == p_c.numel()
    print(f"{n_o} | {n_c}")

    p_c.data[:] = p_o.data

    assert_tensors_equal(p_c.data, p_o.data)

inp = torch.randn(1, 3, 384, 384)
res_c = model_custom(inp)
res_o = model_official(inp)

# Asserts
assert get_n_params(model_custom) == get_n_params(model_official)
assert_tensors_equal(res_c, res_o)

torch.save(model_custom, "model.pth")