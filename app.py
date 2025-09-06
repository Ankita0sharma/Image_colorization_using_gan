from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
from PIL import Image
import io
import base64

# Load trained generator
generator = tf.keras.models.load_model("generator.h5", compile=False)
IMG_SIZE = 120

app = Flask(__name__)

def preprocess_image(file):
    """Convert uploaded file -> grayscale tensor"""
    img = Image.open(file).convert("L").resize((IMG_SIZE, IMG_SIZE))
    arr = np.asarray(img).reshape((1, IMG_SIZE, IMG_SIZE, 1)) / 255.0
    return arr

def postprocess_image(prediction):
    """Convert tensor -> base64 PNG for HTML"""
    prediction = np.clip(prediction[0] * 255, 0, 255).astype(np.uint8)
    img = Image.fromarray(prediction)
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    encoded_img = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return encoded_img

@app.route("/", methods=["GET", "POST"])
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["file"]

        # --- Preprocess grayscale for model ---
        gray_input = preprocess_image(file)

        # --- Save original grayscale as base64 for display ---
        file.stream.seek(0)  # reset file pointer to beginning
        img = Image.open(file.stream).convert("L").resize((IMG_SIZE, IMG_SIZE))
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        input_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        # --- Generate colorized output using GAN ---
        generated = generator.predict(gray_input)
        result_b64 = postprocess_image(generated)

        return render_template(
            "index.html",
            input_image=input_b64,   # grayscale
            result_image=result_b64  # colorized
        )

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
