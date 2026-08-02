import torch
from banknote_classifier.models import CustomVGG16

def test_vgg16_instantiation():
    """CustomVGG16 should be instantiable without errors and be a subclass of nn.Module."""
    model = CustomVGG16()
    assert isinstance(model, torch.nn.Module)

def test_vgg16_output_shape():
    """CustomVGG16 should output the expected shape on CPU forward pass"""
    model = CustomVGG16()
    model.eval()
    
    # according to nn.Linear(7 * 7 * 512, 4096)，輸入影像大小必須是 224x224
    dummy_input = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        output = model(dummy_input)
    assert output.shape[0] == 1   # batch size


def test_vgg16_output_is_tensor():
    """CustomVGG16 should output a torch.Tensor"""
    model = CustomVGG16()
    model.eval()
    
    # according to nn.Linear(7 * 7 * 512, 4096)，輸入影像大小必須是 224x224
    dummy_input = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        output = model(dummy_input)
    assert isinstance(output, torch.Tensor)