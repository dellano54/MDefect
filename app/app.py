import os
import base64
import io
import torch
import numpy as np
import cv2
import time
from PIL import Image
from flask import Flask, render_template, request, jsonify, send_file, send_from_directory
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt
from torchvision import transforms
import torch.nn as nn
from torchvision.models import efficientnet_b0
from fpdf import FPDF
from PIL import Image
import io





app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5MB limit
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load model and configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def add_image_to_pdf(pdf, base64_img, x, y, width, format='PNG'):
    """Add base64 image to PDF with proper format handling"""
    image_data = base64.b64decode(base64_img)
    temp_file = f"temp_img_{int(time.time())}.{format.lower()}"
    
    # Save the image directly without conversion
    with open(temp_file, 'wb') as f:
        f.write(image_data)
    
    pdf.image(temp_file, x, y, width)
    os.remove(temp_file)


def clean_text(text):
    """Clean text for PDF compatibility"""
    replacements = {
        "\u2011": "-",  # non-breaking hyphen
        "\u2013": "-",  # en dash
        "\u2014": "-",  # em dash
        "\u2018": "'",  # left single quote
        "\u2019": "'",  # right single quote
        "\u201c": '"',  # left double quote
        "\u201d": '"',  # right double quote
        "\xa0": " ",    # non-breaking space
    }
    for bad, good in replacements.items():
        text = text.replace(bad, good)
    return text

class EfficientNetDefectClassifier(nn.Module):
    """Enhanced EfficientNet-B0 based defect classifier"""
    
    def __init__(self, num_classes: int, pretrained: bool = True, dropout_rate: float = 0.3):
        super(EfficientNetDefectClassifier, self).__init__()
        self.backbone = efficientnet_b0(pretrained=pretrained)
        num_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Linear(num_features, num_classes)
        self.features = None
        self.backbone.features.register_forward_hook(self.save_features)
    
    def save_features(self, module, input, output):
        self.features = output
    
    def forward(self, x):
        return self.backbone(x)

class GradCAM:
    """Grad-CAM implementation for model interpretability"""
    
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.forward_handle = target_layer.register_forward_hook(self.save_activations)
        if hasattr(target_layer, 'register_full_backward_hook'):
            self.backward_handle = target_layer.register_full_backward_hook(self.save_gradients)
        else:
            self.backward_handle = target_layer.register_backward_hook(self.save_gradients)
    
    def save_activations(self, module, input, output):
        self.activations = output
    
    def save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
    
    def generate_cam(self, input_tensor, class_idx):
        output = self.model(input_tensor)
        self.model.zero_grad()
        class_score = output[:, class_idx].sum()
        class_score.backward(retain_graph=True)
        gradients = self.gradients[0]
        activations = self.activations[0]
        weights = torch.mean(gradients, dim=(1, 2))
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        cam = torch.relu(cam)
        cam = cam - cam.min()
        cam = cam / cam.max()
        return cam.detach().cpu().numpy()
    
    def __del__(self):
        self.forward_handle.remove()
        self.backward_handle.remove()

def load_trained_model(model_path: str, device: torch.device = None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(model_path, map_location=device)
    model = EfficientNetDefectClassifier(
        num_classes=22,
        pretrained=False,
        dropout_rate=checkpoint['config'].get('dropout_rate', 0.3)
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model, checkpoint['class_to_idx'], checkpoint['config']

# Load model
print("Loading pre-trained model...")
model, class_to_idx, config = load_trained_model('best_model_unified.pth', device)
idx_to_class = {v: k for k, v in class_to_idx.items()}
num_classes = len(class_to_idx)
print(f"Loaded model with {num_classes} classes")

# Setup Grad-CAM
target_layer = model.backbone.features[-1]
grad_cam = GradCAM(model, target_layer)

# Define transforms
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
])

# Mapping for DAGM classes to descriptive names
dagm_descriptive_names = {
    'DAGM_Class1': "Fine-grain scratchy",
    'DAGM_Class2': "Flat speckle spots",
    'DAGM_Class3': "Medium‑grain blob spots",
    'DAGM_Class4': "Shadow patch texture",
    'DAGM_Class5': "Coarse smudge pattern",
    'DAGM_Class6': "Soft blotch areas",
    'DAGM_Class7': "Linear‑streak pattern",
    'DAGM_Class8': "Fine speckle noise",
    'DAGM_Class9': "Cluster rough spots",
    'DAGM_Class10': "Diffuse dense patches"
}

def to_descriptive_name(class_name):
    return dagm_descriptive_names.get(class_name, class_name)

def resize_image(img, max_size=800):
    """Resize image while maintaining aspect ratio"""
    h, w = img.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        return cv2.resize(img, (new_w, new_h))
    return img

def process_image(image_path):
    """Process an image and return analysis results"""
    img = Image.open(image_path).convert('RGB')
    original_img = np.array(img)
    img_tensor = val_transform(img).unsqueeze(0).to(device)
    
    # Get prediction
    with torch.no_grad():
        output = model(img_tensor)
        probabilities = torch.softmax(output, dim=1)[0]
        confidence, pred_idx = torch.max(probabilities, dim=0)
        pred_class = idx_to_class[pred_idx.item()]
    
    # Generate Grad-CAM
    cam = grad_cam.generate_cam(img_tensor, pred_idx.item())
    cam = cv2.resize(cam, (img.width, img.height))
    
    # Generate heatmap overlay
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap_overlay = cv2.addWeighted(original_img, 0.7, heatmap, 0.3, 0)
    
    # Generate bounding boxes
    thresh = cv2.threshold(np.uint8(cam * 255), 100, 255, cv2.THRESH_BINARY)[1]
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bbox_img = original_img.copy()
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(bbox_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    # Generate Sobel edges
    gray = cv2.cvtColor(original_img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_mag = np.sqrt(sobelx**2 + sobely**2)
    sobel_mag = (sobel_mag / sobel_mag.max() * 255).astype(np.uint8)
    mask = (cam > 0.3).astype(np.uint8)
    masked_sobel = cv2.bitwise_and(sobel_mag, sobel_mag, mask=mask)
    sobel_color = cv2.cvtColor(masked_sobel, cv2.COLOR_GRAY2RGB)
    sobel_color = cv2.applyColorMap(sobel_color, cv2.COLORMAP_JET)
    
    # Resize images for better display
    original_img = resize_image(original_img)
    heatmap_overlay = resize_image(heatmap_overlay)
    bbox_img = resize_image(bbox_img)
    sobel_color = resize_image(sobel_color)
    
    # Create confidence bar chart
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    sorted_probs, sorted_idxs = torch.sort(probabilities, descending=True)
    top_probs = sorted_probs[:5].cpu().numpy()
    top_classes = [idx_to_class[i.item()] for i in sorted_idxs[:5]]
    top_display_classes = [to_descriptive_name(c) for c in top_classes]
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, 5))
    bars = ax.barh(top_display_classes, top_probs, color=colors)
    ax.set_xlabel('Confidence')
    ax.set_title('Top 5 Class Predictions')
    ax.set_xlim(0, 1)
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.02, bar.get_y() + bar.get_height()/2, 
                f'{width:.3f}', va='center', fontsize=10)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    confidence_chart = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    buf.close()
    
    # Convert images to base64
    def img_to_base64(img, format='JPEG'):
        _, buffer = cv2.imencode(f'.{format.lower()}', img)
        return base64.b64encode(buffer).decode('utf-8')
    
    # Prepare all class names
    all_classes = list(idx_to_class.values())
    all_display_classes = [to_descriptive_name(c) for c in all_classes]
    
    return {
        'original': img_to_base64(original_img, 'JPEG'),
        'heatmap': img_to_base64(heatmap_overlay, 'JPEG'),
        'bbox': img_to_base64(bbox_img, 'JPEG'),
        'sobel': img_to_base64(sobel_color, 'JPEG'),
        'confidence_chart': confidence_chart,  # This is already PNG
        'predicted_class': to_descriptive_name(pred_class),
        'confidence': confidence.item(),
        'all_classes': all_display_classes,
        'probabilities': probabilities.cpu().numpy().tolist(),
        'original_class_names': all_classes
    }


@app.route('/')
def index():
    """Render home page with sample images"""
    sample_images = [f for f in os.listdir('sample') if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    return render_template('index.html', sample_images=sample_images, num_classes=num_classes)

@app.route('/analyze', methods=['POST'])
def analyze():
    """Analyze uploaded or sample image"""
    if 'file' in request.files:
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        # Save uploaded file temporarily
        filename = f"upload_{int(time.time())}.jpg"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        image_path = file_path
    elif 'sample' in request.form:
        sample_name = request.form['sample']
        image_path = os.path.join('sample', sample_name)
    else:
        return jsonify({'error': 'No image provided'}), 400
    
    try:
        results = process_image(image_path)
        # Add image path for report generation
        results['image_path'] = image_path
        return jsonify(results)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/download_report', methods=['POST'])
def download_report():
    """Generate and download PDF report by re-processing the image"""
    if 'image_path' not in request.json:
        return jsonify({'error': 'Image path missing in request'}), 400
    
    image_path = request.json['image_path']
    if not os.path.exists(image_path):
        return jsonify({'error': 'Image file not found'}), 404
    
    try:
        # Reprocess image to get analysis results
        results = process_image(image_path)
        
        # Generate PDF
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        
        # Add report header
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(0, 10, "Defect Analysis Report", 0, 1, 'C')
        pdf.ln(10)
        
        # Add prediction results
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, clean_text(f"Predicted Defect: {results['predicted_class']}"), 0, 1)
        pdf.set_font("Arial", size=12)
        pdf.cell(0, 10, clean_text(f"Confidence: {results['confidence']:.4f}"), 0, 1)
        pdf.ln(10)
        
        # Add visualizations
        img_width = 85  # Width for side-by-side images
        img_height = 60  # Approximate height for images
        y_position = pdf.get_y()
        
        # Original and Heatmap
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, "Original Image & Heatmap Analysis", 0, 1)
        add_image_to_pdf(pdf, results['original'], 15, y_position + 15, img_width, 'JPEG')
        add_image_to_pdf(pdf, results['heatmap'], 110, y_position + 15, img_width, 'JPEG')
        
        # Bounding Box and Edge Analysis
        y_position += img_height + 20
        pdf.set_xy(0, y_position)
        pdf.cell(0, 10, "Defect Localization & Edge Analysis", 0, 1)
        add_image_to_pdf(pdf, results['bbox'], 15, y_position + 15, img_width, 'JPEG')
        add_image_to_pdf(pdf, results['sobel'], 110, y_position + 15, img_width, 'JPEG')
        
        # Confidence Chart (PNG format)
        y_position += img_height + 20
        pdf.set_xy(0, y_position)
        pdf.cell(0, 10, "Confidence Distribution", 0, 1)
        add_image_to_pdf(pdf, results['confidence_chart'], 20, y_position + 15, 170, 'PNG')
        
        # Move y-position down by chart height
        y_position += 80
        
        # Class Probabilities Table - always start on new page if needed
        if y_position > 200:  # Check if we need a new page
            pdf.add_page()
            y_position = 20
        
        pdf.set_xy(0, y_position)
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, "Class Probabilities", 0, 1)
        pdf.set_font("Arial", size=10)
        
        # Create table header
        pdf.set_fill_color(200, 220, 255)
        pdf.cell(120, 8, "Defect Class", 1, 0, 'C', 1)
        pdf.cell(60, 8, "Probability", 1, 1, 'C', 1)
        
        # Add class probabilities
        for i, class_name in enumerate(results['original_class_names']):
            prob = results['probabilities'][i]
            pdf.cell(120, 8, clean_text(to_descriptive_name(class_name)), 1)
            pdf.cell(60, 8, f"{prob:.4f}", 1, 1, 'C')
        
        # Analysis summary
        pdf.ln(10)
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 8, clean_text(
            "Analysis Summary:\n"
            "The AI-powered defect detection system has identified manufacturing defects "
            "using computer vision techniques including Grad-CAM visualization, "
            "bounding box detection, and edge analysis. The confidence distribution "
            "shows the model's certainty across different defect classes."
        ))
        
        # Save to bytes buffer
        pdf_bytes = pdf.output(dest='S').encode('latin1')
        buffer = io.BytesIO(pdf_bytes)
        buffer.seek(0)
        
        return send_file(
            buffer,
            as_attachment=True,
            download_name='defect_report.pdf',
            mimetype='application/pdf'
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/sample/<filename>')
def serve_sample(filename):
    return send_from_directory('sample', filename)

@app.route('/uploads/<filename>')
def serve_upload(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True, threaded=False)