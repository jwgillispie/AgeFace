
import torch
from commons import get_tensor, get_model
def get_age(image_bytes):
    tensor = get_tensor(image_bytes)
    model = get_model()
    model.eval()
    outputs = model(tensor)

    _, pred = torch.max(outputs, 1)
    print(pred)
    categ = pred.item()
    print(categ)
    return categ

    