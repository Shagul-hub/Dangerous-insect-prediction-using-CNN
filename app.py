import os
from flask import Flask, request, render_template, redirect, url_for, flash
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np


MODEL_PATH = "cnn_model.h5"     
UPLOAD_FOLDER = "static/uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
CONFIDENCE_THRESHOLD = 0.60     

LABELS = [
    "Western Corn Rootworms", "Tomato Hornworms", "Thrips", "Spider Mites", "Fruit Flies",
    "Fall Armyworms", "Corn Borers", "Corn Earworms", "Colorado Potato Beetles", "Citrus Canker",
    "Cabbage Loopers", "Brown Marmorated Stink Bugs", "Aphids", "Armyworms",
    "Africanized Honey Bees (Killer Bees)"
]


INSECT_INFO = {
    "Western Corn Rootworms": {
        "harm": [
            "Damage corn roots, leading to plant lodging.",
            "Reduce nutrient and water uptake.",
            "Weaken plant structure, lowering yield."
        ],
        "prevention": [
            "Practice crop rotation with non-host crops.",
            "Use insecticide-treated seeds.",
            "Monitor root health and pest population regularly."
        ]
    },
    "Tomato Hornworms": {
        "harm": [
            "Defoliate tomato and pepper plants rapidly.",
            "Chew on green fruits causing direct crop loss.",
            "Reduce photosynthesis, slowing plant growth."
        ],
        "prevention": [
            "Handpick larvae from plants.",
            "Spray neem oil or insecticidal soap.",
            "Encourage parasitic wasps in the garden."
        ]
    },
    "Thrips": {
        "harm": [
            "Cause silver streaks and leaf distortion.",
            "Transmit plant viruses like TSWV.",
            "Damage flowers and fruits by sucking sap."
        ],
        "prevention": [
            "Use reflective mulches to deter thrips.",
            "Apply insecticidal soap sprays.",
            "Introduce predatory insects such as lacewings."
        ]
    },
    "Spider Mites": {
        "harm": [
            "Cause leaf yellowing and premature leaf drop.",
            "Weaken plants by sucking out cell contents.",
            "Create webbing that hinders photosynthesis."
        ],
        "prevention": [
            "Spray plants with water to remove mites.",
            "Use neem oil or horticultural oil.",
            "Introduce predatory mites like Phytoseiulus."
        ]
    },
    "Fruit Flies": {
        "harm": [
            "Lay eggs inside ripening fruits.",
            "Cause fruit rot and spoilage.",
            "Reduce market value of harvested fruits."
        ],
        "prevention": [
            "Use bait traps to capture adult flies.",
            "Remove and destroy infested fruits.",
            "Cover fruits with protective netting."
        ]
    },
    "Fall Armyworms": {
        "harm": [
            "Feed on maize, rice, and sorghum leaves.",
            "Cause rapid crop destruction in large numbers.",
            "Chew plant stems and growing points."
        ],
        "prevention": [
            "Use pheromone traps for early detection.",
            "Apply biopesticides like Bacillus thuringiensis (Bt).",
            "Rotate crops to break life cycles."
        ]
    },
    "Corn Borers": {
        "harm": [
            "Bore into corn stalks, weakening plants.",
            "Reduce nutrient flow within plants.",
            "Cause lodging and yield loss."
        ],
        "prevention": [
            "Plant Bt corn varieties.",
            "Shred and plow crop residues after harvest.",
            "Use pheromone traps for monitoring."
        ]
    },
    "Corn Earworms": {
        "harm": [
            "Feed on corn kernels and silks.",
            "Damage fruits like tomatoes and peppers.",
            "Reduce quality and market value of produce."
        ],
        "prevention": [
            "Apply mineral oil to corn silks.",
            "Use light traps to monitor activity.",
            "Release beneficial insects like Trichogramma."
        ]
    },
    "Colorado Potato Beetles": {
        "harm": [
            "Defoliate potato, tomato, and eggplant crops.",
            "Can develop pesticide resistance quickly.",
            "Reduce tuber yield significantly."
        ],
        "prevention": [
            "Handpick beetles and larvae.",
            "Rotate crops to non-host plants.",
            "Use biological insecticides such as Bt var. tenebrionis."
        ]
    },
    "Citrus Canker": {
        "harm": [
            "Cause lesions on fruit, leaves, and stems.",
            "Lead to fruit drop before ripening.",
            "Reduce marketability of citrus crops."
        ],
        "prevention": [
            "Remove and destroy infected trees.",
            "Apply copper-based bactericides.",
            "Avoid movement of infected plant materials."
        ]
    },
    "Cabbage Loopers": {
        "harm": [
            "Chew large holes in cabbage and leafy greens.",
            "Damage can lead to bacterial infections.",
            "Reduce head formation in cabbage."
        ],
        "prevention": [
            "Use floating row covers to exclude moths.",
            "Spray Bt formulations on leaves.",
            "Encourage natural predators like ground beetles."
        ]
    },
    "Brown Marmorated Stink Bugs": {
        "harm": [
            "Pierce fruits and vegetables, causing pitting.",
            "Leave foul odor on harvested produce.",
            "Reduce crop yield and quality."
        ],
        "prevention": [
            "Seal cracks and openings in buildings.",
            "Use pheromone traps for monitoring.",
            "Apply insecticides during heavy infestations."
        ]
    },
    "Aphids": {
        "harm": [
            "Suck plant sap, weakening the plant.",
            "Transmit viral plant diseases.",
            "Cause leaf curling and stunted growth."
        ],
        "prevention": [
            "Spray with strong jets of water.",
            "Use neem oil or insecticidal soap.",
            "Release ladybugs as natural predators."
        ]
    },
    "Armyworms": {
        "harm": [
            "Strip entire fields of foliage.",
            "Damage crops like rice, maize, and wheat.",
            "Cause rapid economic losses."
        ],
        "prevention": [
            "Scout fields early for larvae.",
            "Apply Bt-based pesticides.",
            "Mow grassy borders to reduce egg laying."
        ]
    },
    "Africanized Honey Bees (Killer Bees)": {
        "harm": [
            "Highly aggressive stinging behavior.",
            "Dangerous to humans and livestock.",
            "Can chase threats for long distances."
        ],
        "prevention": [
            "Avoid disturbing bee swarms.",
            "Call professionals for hive removal.",
            "Seal potential nesting sites."
        ]
    }
}


app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.secret_key = "replace_this_with_a_random_secret"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Place your .h5 model there.")
model = load_model(MODEL_PATH)


input_height, input_width = model.input_shape[1:3]


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img = img.resize((input_height, input_width))  # match model's expected size
    arr = np.array(img).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        flash("No file part")
        return redirect(request.url)

    file = request.files["file"]
    if file.filename == "":
        flash("No selected file")
        return redirect(request.url)

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        
        x = preprocess_image(filepath)
        preds = model.predict(x)[0]
        top_idx = preds.argsort()[::-1]
        top_labels = [LABELS[i] for i in top_idx]

        predicted_label = top_labels[0]
        confidence = float(preds[top_idx[0]])

        if confidence < CONFIDENCE_THRESHOLD:
            result = {
                "is_known": False,
                "message": "The uploaded image does not belong to a known insect species from the dataset."
            }
            return render_template("result.html", uploaded_image=filepath, result=result)

        info = INSECT_INFO.get(predicted_label, {
            "harm": ["Harm information not available."],
            "prevention": ["Prevention information not available."]
        })

        result = {
            "is_known": True,
            "predicted_label": predicted_label,
            "harm": info["harm"][:3],  
            "prevention": info["prevention"][:3]  
        }

        return render_template("result.html", uploaded_image=filepath, result=result)

    else:
        flash("Allowed file types: png, jpg, jpeg")
        return redirect(request.url)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
