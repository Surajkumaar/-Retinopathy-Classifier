
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
    print("\nDetailed results:")
    for cls, prob in results['probabilities'].items():
        print(f"  {cls}: {prob:.4f}")
