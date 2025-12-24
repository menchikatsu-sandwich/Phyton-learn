import torch, torchvision
print("torch", torch.__version__)
print("torch.cuda.is_available()", torch.cuda.is_available())
print("torch.version.cuda", torch.version.cuda)
print("torch.cuda.device_count()", torch.cuda.device_count())
print("torch.cuda.get_device_name(0) if available", torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')
print("torchvision", torchvision.__version__)
from torchvision import ops
print("has nms attr:", hasattr(ops, 'nms'))
try:
    boxes = torch.tensor([[0.,0.,10.,10.],[1.,1.,11.,11.]], dtype=torch.float32)
    scores = torch.tensor([0.9, 0.8], dtype=torch.float32)
    if torch.cuda.is_available():
        boxes = boxes.cuda()
        scores = scores.cuda()
    print('calling ops.nms...')
    out = ops.nms(boxes, scores, 0.5)
    print('nms output:', out)
except Exception as e:
    import traceback
    traceback.print_exc()
