import os
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import densenet121, DenseNet121_Weights
import onnx
import onnxruntime
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def convert_to_onnx():
    # Load the model directly as it was saved during training
    checkpoint_path = "best_model.ckpt"
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    # Initialize model with ImageNet weights 
    weights = DenseNet121_Weights.IMAGENET1K_V1
    model = densenet121(weights=weights)
    model.classifier = nn.Linear(1024, 5)  # Replace with 5 classes for retinopathy
    
    # Load the trained weights
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()  # Set the model to evaluation mode
    
    # Create dummy input tensor for ONNX export
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    
    # Output file path
    onnx_path = "retinopathy_model.onnx"
    
    # Export the model to ONNX format with simplified options
    torch.onnx.export(
        model,                     # PyTorch model
        dummy_input,               # Input tensor
        onnx_path,                 # Output file path
        export_params=True,        # Store the trained parameter weights
        opset_version=14,          # Use a newer opset (better for newer hardware)
        do_constant_folding=True,  # Optimization: fold constant values
        input_names=["input"],     # Name for the input tensor
        output_names=["output"],   # Name for the output tensor
        dynamic_axes={             # Support for dynamic batch size
            "input": {0: "batch_size"},
            "output": {0: "batch_size"}
        }
    )
    
    print(f"Model exported to {onnx_path}")
    
    # Verify the ONNX model
    verify_onnx_model(onnx_path, dummy_input, model)
    
def verify_onnx_model(onnx_path, dummy_input, torch_model):
    """Verify the ONNX model to ensure correct conversion"""
    print("Verifying ONNX model...")
    
    try:
        # Load and check the ONNX model
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX model structure is valid")
        
        # Compare PyTorch and ONNX model outputs
        # Get PyTorch output
        with torch.no_grad():
            torch_output = torch_model(dummy_input).cpu().numpy()
        
        # Get ONNX output - try GPU first, fall back to CPU
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        try:
            ort_session = onnxruntime.InferenceSession(
                onnx_path,
                providers=providers
            )
            print(f"Using {ort_session.get_providers()[0]} for ONNX inference")
        except Exception as e:
            print(f"Error creating inference session with GPU: {e}")
            print("Falling back to CPU...")
            providers = ['CPUExecutionProvider']
            ort_session = onnxruntime.InferenceSession(
                onnx_path,
                providers=providers
            )
            
        ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.cpu().numpy()}
        ort_output = ort_session.run(None, ort_inputs)[0]
        
        # Compare outputs
        np.testing.assert_allclose(torch_output, ort_output, rtol=1e-3, atol=1e-5)
        print("PyTorch and ONNX model outputs match within tolerance!")
        
        print("\nExample inference with ONNX model:")
        print(f"Output shape: {ort_output.shape}")
        print(f"Class probabilities: {np.exp(ort_output) / np.sum(np.exp(ort_output), axis=1, keepdims=True)}")
        
        # Print some model metadata
        print("\nONNX Model Metadata:")
        print(f"Input name: {ort_session.get_inputs()[0].name}")
        print(f"Input shape: {ort_session.get_inputs()[0].shape}")
        print(f"Output name: {ort_session.get_outputs()[0].name}")
        print(f"Output shape: {ort_session.get_outputs()[0].shape}")
        
        return True
        
    except Exception as e:
        print(f"Verification failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
def test_with_sample_image(image_path, onnx_path):
    """Test the ONNX model with a sample image"""
    # Define the same normalization as in your training script
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5353, 0.3628, 0.2486], std=[0.2126, 0.1586, 0.1401]),
    ])
    
    # Load and preprocess the image
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).numpy()  # Add batch dimension
    
    # Run inference with ONNX
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    ort_session = onnxruntime.InferenceSession(onnx_path, providers=providers)
    ort_inputs = {ort_session.get_inputs()[0].name: input_tensor}
    ort_output = ort_session.run(None, ort_inputs)[0]
    
    # Process output
    probabilities = np.exp(ort_output) / np.sum(np.exp(ort_output), axis=1, keepdims=True)
    predicted_class = np.argmax(probabilities)
    
    class_names = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']
    
    print(f"Predicted class: {class_names[predicted_class]} (Class {predicted_class})")
    print(f"Class probabilities:")
    for i, cls in enumerate(class_names):
        print(f"  {cls}: {probabilities[0][i]:.4f}")
    
    return predicted_class, probabilities

def create_sample_inference_code(onnx_path):
    """Create a sample inference script for the ONNX model"""
    code = """
import onnxruntime
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import time

def preprocess_image(image_path):
    # Define the same preprocessing as during training
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5353, 0.3628, 0.2486], std=[0.2126, 0.1586, 0.1401]),
    ])
    
    # Load and preprocess the image
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).numpy()  # Add batch dimension
    return input_tensor

def predict_retinopathy(image_path, onnx_path):
    # Preprocess the image
    start_time = time.time()
    input_tensor = preprocess_image(image_path)
    preprocess_time = time.time() - start_time
    
    # Set up ONNX Runtime session
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    session = onnxruntime.InferenceSession(onnx_path, providers=providers)
    input_name = session.get_inputs()[0].name
    
    # Run inference
    start_time = time.time()
    outputs = session.run(None, {input_name: input_tensor})[0]
    inference_time = time.time() - start_time
    
    # Process results
    probabilities = np.exp(outputs) / np.sum(np.exp(outputs), axis=1, keepdims=True)
    predicted_class = np.argmax(probabilities)
    
    # Class names
    class_names = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']
    
    print(f"Predicted class: {class_names[predicted_class]}")
    print(f"Confidence: {probabilities[0][predicted_class]:.4f}")
    print(f"Preprocessing time: {preprocess_time:.4f} seconds")
    print(f"Inference time: {inference_time:.4f} seconds")
    
    return {
        'class': class_names[predicted_class],
        'class_id': int(predicted_class),
        'probabilities': {class_names[i]: float(probabilities[0][i]) for i in range(len(class_names))},
        'preprocess_time': preprocess_time,
        'inference_time': inference_time
    }

# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Predict diabetic retinopathy using ONNX model')
    parser.add_argument('--image', type=str, required=True, help='Path to the input image')
    parser.add_argument('--model', type=str, default="retinopathy_model.onnx", help='Path to the ONNX model')
    
    args = parser.parse_args()
    
    results = predict_retinopathy(args.image, args.model)
    print("\\nDetailed results:")
    for cls, prob in results['probabilities'].items():
        print(f"  {cls}: {prob:.4f}")
"""
    
    # Save the inference script
    with open("predict_with_onnx.py", "w") as f:
        f.write(code)
    
    print("Sample inference code saved to 'predict_with_onnx.py'")

if __name__ == "__main__":
    # Show PyTorch and ONNX versions
    print(f"PyTorch version: {torch.__version__}")
    print(f"ONNX version: {onnx.__version__}")
    print(f"ONNX Runtime version: {onnxruntime.__version__}")
    
    # Convert the model to ONNX
    convert_to_onnx()
    
    # Create sample inference code
    create_sample_inference_code("retinopathy_model.onnx")
    
    # Optional: Test with a sample image if available
    # Uncomment and provide a path to a test image
    # test_image_path = "path/to/test/image.jpeg"
    # if os.path.exists(test_image_path):
    #     test_with_sample_image(test_image_path, "retinopathy_model.onnx")