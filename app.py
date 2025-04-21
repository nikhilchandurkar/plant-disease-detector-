


from flask import Flask, render_template, request
import torch
from torchvision import transforms
from PIL import Image
import os

app = Flask(__name__)
model = torch.load("plant_disease_model.pth", map_location=torch.device("cpu"))
# model.eval()

UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Full class_details dictionary (make sure it's complete with all classes up to model's output)
class_details = {
    0: {
        'name': 'Apple Scab',
        'description': 'Fungal disease causing dark lesions on leaves and fruit, leading to defoliation and fruit rot.',
        'treatment': 'Apply fungicides (e.g., captan, mancozeb), ensure proper pruning, and remove fallen debris.'
    },
    1: {
        'name': 'Apple Black Rot',
        'description': 'Fungal disease causing circular lesions and fruit rot.',
        'treatment': 'Remove and destroy infected fruit and branches. Use fungicides during the growing season.'
    },
    2: {
        'name': 'Apple Cedar Rust',
        'description': 'Caused by the fungus *Gymnosporangium juniperi-virginianae*, resulting in yellow-orange leaf spots.',
        'treatment': 'Remove nearby junipers (alternate host), apply fungicides, and maintain tree health.'
    },
    3: {
        'name': 'Apple Healthy',
        'description': 'No signs of disease or pest infestation.',
        'treatment': 'Continue routine care and monitoring for early signs of stress or infection.'
    },
    4: {
        'name': 'Blueberry Healthy',
        'description': 'No visible disease symptoms. Foliage is green and berries are forming well.',
        'treatment': 'Maintain balanced fertilization and proper irrigation. Monitor regularly.'
    },
    5: {
        'name': 'Cherry Powdery Mildew',
        'description': 'Fungal disease causing a white, powdery growth on leaves and fruit.',
        'treatment': 'Apply sulfur-based fungicides, remove affected leaves, and improve air circulation.'
    },
    6: {
        'name': 'Cherry Healthy',
        'description': 'Tree shows vigorous growth with no disease symptoms.',
        'treatment': 'Regular watering, pruning, and fertilization help sustain tree health.'
    },
    7: {
        'name': 'Corn Cercospora Leaf Spot',
        'description': 'Caused by the fungus *Cercospora zeae-maydis*, it leads to grayish spots on leaves.',
        'treatment': 'Use resistant hybrids, rotate crops, and apply fungicides if needed.'
    },
    8: {
        'name': 'Corn Common Rust',
        'description': 'Caused by *Puccinia sorghi*, presenting as reddish-brown pustules on leaves.',
        'treatment': 'Use resistant hybrids, practice crop rotation, and apply foliar fungicides when necessary.'
    },
    9: {
        'name': 'Corn Northern Leaf Blight',
        'description': 'Fungal disease causing long gray-green lesions on leaves.',
        'treatment': 'Plant resistant varieties and apply fungicides early in disease development.'
    },
    10: {
        'name': 'Corn Healthy',
        'description': 'No apparent signs of pest or disease damage.',
        'treatment': 'Practice good cultural management and periodic crop scouting.'
    },
    11: {
        'name': 'Grape Black Rot',
        'description': 'Fungal disease causing brown leaf spots and fruit rot.',
        'treatment': 'Remove infected plant parts and apply protectant fungicides (e.g., myclobutanil).'
    },
    12: {
        'name': 'Grape Esca (Black Measles)',
        'description': 'Complex disease causing dark streaks and fruit mummification.',
        'treatment': 'Avoid wounding, prune out diseased wood, and ensure well-drained soil.'
    },
    13: {
        'name': 'Grape Leaf Blight',
        'description': 'Leads to brown lesions and leaf necrosis, reducing photosynthesis.',
        'treatment': 'Apply copper-based fungicides and practice proper vineyard sanitation.'
    },
    14: {
        'name': 'Grape Healthy',
        'description': 'Leaves are green and fruit is developing properly.',
        'treatment': 'Ensure consistent irrigation and protect against fungal spores.'
    },
    15: {
        'name': 'Orange Huanglongbing (Citrus Greening)',
        'description': 'Bacterial disease causing blotchy mottling and fruit deformities.',
        'treatment': 'Control psyllid vectors, remove infected trees, and use certified disease-free planting material.'
    },
    16: {
        'name': 'Peach Bacterial Spot',
        'description': 'Results in pitted lesions on fruit and angular spots on leaves.',
        'treatment': 'Use copper-based bactericides and resistant cultivars.'
    },
    17: {
        'name': 'Peach Healthy',
        'description': 'No signs of disease; tree is producing healthy fruit and foliage.',
        'treatment': 'Maintain regular orchard care including thinning and watering.'
    },
    18: {
        'name': 'Bell Pepper Bacterial Spot',
        'description': 'Causes water-soaked lesions that turn brown and scabby.',
        'treatment': 'Avoid overhead irrigation, use copper sprays, and rotate crops.'
    },
    19: {
        'name': 'Bell Pepper Healthy',
        'description': 'Healthy plant with vibrant green foliage and proper fruiting.',
        'treatment': 'Maintain optimal watering, spacing, and nutrient levels.'
    },
    20: {
        'name': 'Potato Early Blight',
        'description': 'Characterized by concentric brown lesions on leaves.',
        'treatment': 'Use fungicides like chlorothalonil and practice crop rotation.'
    },
    21: {
        'name': 'Potato Late Blight',
        'description': 'Caused by *Phytophthora infestans*, leading to leaf and tuber rot.',
        'treatment': 'Apply systemic fungicides and destroy infected plants.'
    },
    22: {
        'name': 'Potato Healthy',
        'description': 'No disease symptoms present. Leaves and tubers developing normally.',
        'treatment': 'Ensure regular weeding, soil care, and pest monitoring.'
    },
    23: {
        'name': 'Strawberry Leaf Scorch',
        'description': 'Red to brown spots on leaves with yellow halos.',
        'treatment': 'Remove infected leaves and avoid overhead irrigation.'
    },
    24: {
        'name': 'Strawberry Healthy',
        'description': 'Green foliage, healthy flowers, and vibrant fruits.',
        'treatment': 'Keep soil moist, use mulch, and monitor for pests.'
    },
    25: {
        'name': 'Tomato Bacterial Spot',
        'description': 'Small water-soaked spots that turn black with a yellow halo.',
        'treatment': 'Use copper-based sprays and rotate crops.'
    },
    26: {
        'name': 'Tomato Early Blight',
        'description': 'Fungal disease causing dark concentric rings on older leaves.',
        'treatment': 'Apply fungicides and stake plants for better air circulation.'
    },
    27: {
        'name': 'Tomato Late Blight',
        'description': 'Rapidly spreading brown lesions caused by *Phytophthora infestans*.',
        'treatment': 'Destroy infected plants and use systemic fungicides.'
    },
    28: {
        'name': 'Tomato Leaf Mold',
        'description': 'Yellow patches on upper leaves and olive-green mold underneath.',
        'treatment': 'Use resistant varieties and apply fungicides like chlorothalonil.'
    },
    29: {
        'name': 'Tomato Septoria Leaf Spot',
        'description': 'Circular brown spots with gray centers on lower leaves.',
        'treatment': 'Remove infected leaves and apply fungicides like mancozeb.'
    },
    30: {
        'name': 'Tomato Spider Mites',
        'description': 'Tiny pests causing stippled, yellowed leaves with webbing.',
        'treatment': 'Use miticides or neem oil and increase humidity.'
    },
    31: {
        'name': 'Tomato Target Spot',
        'description': 'Circular spots with concentric rings and yellow halos.',
        'treatment': 'Improve air flow, remove infected leaves, and apply fungicides.'
    },
    32: {
        'name': 'Tomato Yellow Leaf Curl Virus',
        'description': 'Leads to stunted growth and upward curling of leaves.',
        'treatment': 'Control whiteflies, use virus-resistant varieties, and apply insecticidal soaps.'
    },
    33: {
        'name': 'Tomato Mosaic Virus',
        'description': 'Causes mottled yellow-green leaves and distorted fruit.',
        'treatment': 'Destroy infected plants and sanitize tools.'
    },
    34: {
        'name': 'Tomato Healthy',
        'description': 'Strong stems, green leaves, and healthy fruit formation.',
        'treatment': 'Support growth with balanced fertilization and pest management.'
    },
    35: {
        'name': 'Soybean Healthy',
        'description': 'Even foliage coloration and consistent growth.',
        'treatment': 'Maintain good soil conditions and check for aphids or fungal disease.'
    },
    36: {
        'name': 'Squash Powdery Mildew',
        'description': 'White, powdery spots on leaves and stems.',
        'treatment': 'Improve air circulation and apply sulfur or potassium bicarbonate.'
    },
    37: {
        'name': 'Wheat Leaf Rust',
        'description': 'Orange-red pustules on leaf surfaces caused by *Puccinia triticina*.',
        'treatment': 'Use resistant cultivars and apply fungicides during early development.'
    },
    38: {
        'name': 'Wheat Healthy',
        'description': 'No visible leaf or stem damage; grain is forming uniformly.',
        'treatment': 'Keep fields weed-free and monitor regularly for pests or weather stress.'
    },
    39: {
        'name': 'Unknown or Other Class',
        'description': 'Class ID not recognized. Possibly a placeholder or needs data mapping.',
        'treatment': 'Verify class prediction and update dataset with correct mapping if necessary.'
    }
}




@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    disease_info = None
    image_url = None

    if request.method == "POST":
        file = request.files["image"]
        if file:
            path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(path)

            image = Image.open(path).convert("RGB")
            img_tensor = transform(image).unsqueeze(0)
            with torch.no_grad():
                output = model(img_tensor)
                predicted_index = output.argmax().item()
                disease_info = class_details.get(predicted_index, {
                    "name": "Unknown",
                    "description": "Not found in database.",
                    "treatment": "Try uploading a clearer image or consult an expert."
                })
                prediction = disease_info["name"]
                image_url = path

    return render_template("index.html", prediction=prediction, disease_info=disease_info, image_url=image_url)


if __name__ == "__main__":
    app.run(debug=True)
