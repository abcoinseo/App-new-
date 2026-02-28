import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from flask import Flask, request, send_file
from PIL import Image

# CONFIG
DATASET_PATH = "dataset"
MODEL_PATH = "model.pth"
IMAGE_SIZE = 28

app = Flask(__name__)

# AI Model
class ImageAI(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(IMAGE_SIZE*IMAGE_SIZE, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, IMAGE_SIZE*IMAGE_SIZE),
            nn.Sigmoid()
        )

    def forward(self, x):

        x = self.encoder(x)
        x = self.decoder(x)

        return x

model = ImageAI()

optimizer = optim.Adam(model.parameters(), lr=0.001)

# LOAD DATASET
def load_dataset():

    data = []

    if not os.path.exists(DATASET_PATH):
        print("dataset folder not found")
        return data

    for file in os.listdir(DATASET_PATH):

        if file.endswith(".txt"):

            path = os.path.join(DATASET_PATH, file)

            with open(path, "r", encoding="utf-8") as f:

                lines = f.read().splitlines()

                for i in range(0, len(lines), 3):

                    if i >= len(lines):
                        continue

                    img_name = lines[i].strip()

                    img_path = os.path.join(DATASET_PATH, img_name)

                    if os.path.exists(img_path):

                        try:

                            img = Image.open(img_path).convert("L")

                            img = img.resize((IMAGE_SIZE, IMAGE_SIZE))

                            arr = np.array(img)/255.0

                            arr = arr.flatten()

                            data.append(arr)

                        except Exception as e:
                            print("error loading:", img_name)

    print("Loaded images:", len(data))

    return data


# TRAIN FUNCTION
def train_model():

    dataset = load_dataset()

    if len(dataset) == 0:
        print("No dataset found")
        return

    print("Training started...")

    for epoch in range(300):

        total_loss = 0

        for img in dataset:

            input_tensor = torch.tensor(img, dtype=torch.float32)

            output = model(input_tensor)

            loss = ((output - input_tensor)**2).mean()

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        if epoch % 50 == 0:

            print(f"Epoch {epoch} Loss {total_loss:.4f}")

    torch.save(model.state_dict(), MODEL_PATH)

    print("Training complete and model saved")


# LOAD MODEL
def load_model():

    if os.path.exists(MODEL_PATH):

        model.load_state_dict(torch.load(MODEL_PATH))

        print("Model loaded")

    else:

        print("No model found, training new model")

        train_model()


# GENERATE IMAGE
def generate_image():

    noise = torch.rand(IMAGE_SIZE*IMAGE_SIZE)

    output = model(noise).detach().numpy()

    img = output.reshape(IMAGE_SIZE, IMAGE_SIZE) * 255

    image = Image.fromarray(img.astype(np.uint8))

    filename = "generated.png"

    image.save(filename)

    return filename


# ROUTES

@app.route("/")
def home():

    return """
    <h2>AI Image Trainer</h2>
    <a href='/train'>Train Model</a><br><br>
    <a href='/generate'>Generate Image</a>
    """


@app.route("/train")
def train():

    train_model()

    return "Training complete"


@app.route("/generate")
def generate():

    filename = generate_image()

    return send_file(filename, mimetype="image/png")


# START
if __name__ == "__main__":

    load_model()

    port = int(os.environ.get("PORT", 5000))

    app.run(host="0.0.0.0", port=port)
