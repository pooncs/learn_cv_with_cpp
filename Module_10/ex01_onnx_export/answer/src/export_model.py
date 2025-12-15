import torch
import torch.nn as nn
import torch.onnx

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(10, 5)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(5, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

def export_onnx():
    model = SimpleModel()
    model.eval() # Set to inference mode

    # Dummy input matching the input size (Batch Size, Input Features)
    dummy_input = torch.randn(1, 10)

    output_file = "simple_model.onnx"
    
    print(f"Exporting model to {output_file}...")
    torch.onnx.export(
        model, 
        dummy_input, 
        output_file,
        verbose=True,
        input_names=['input'],
        output_names=['output'],
        opset_version=13
    )
    print("Export complete.")

if __name__ == "__main__":
    export_onnx()
