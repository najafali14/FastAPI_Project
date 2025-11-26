from fastapi import FastAPI, UploadFile, File  # type: ignore
from fastapi.responses import JSONResponse  # type: ignore
from google import genai
from google.genai import types  # type: ignore
from PIL import Image
from io import BytesIO
import requests
import base64
import uuid
import os

# rembg-lite
from rembg import remove # type: ignore

app = FastAPI()

# Gemini client (ENV VAR)
client = genai.Client(api_key=os.getenv("GEMINI_KEY"))

# ImgBB key (ENV VAR)
IMGBB_API_KEY = os.getenv("IMGBB_KEY")

# ------------------------------
# Helper: Upload image to imgBB
# ------------------------------
def upload_to_imgbb(pil_image):
    buffer = BytesIO()
    pil_image.save(buffer, format="PNG")
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.getvalue()).decode()

    response = requests.post(
        "https://api.imgbb.com/1/upload",
        data={
            "key": IMGBB_API_KEY,
            "image": img_base64,
            "name": f"generated_{uuid.uuid4()}"
        }
    )

    data = response.json()
    if data.get("success"):
        return data["data"]["url"]
    else:
        raise ValueError(f"imgBB upload failed: {data}")

# ------------------------------
# Background Removal (rembg-lite)
# ------------------------------
def remove_background(pil_image):
    img_bytes = BytesIO()
    pil_image.save(img_bytes, format="PNG")
    img_bytes.seek(0)

    output_bytes = remove(img_bytes.getvalue())
    return Image.open(BytesIO(output_bytes))

# ------------------------------
# Helper: Generate image with Gemini
# ------------------------------
def generate_image(prompt, input_image):
    response = client.models.generate_content(
        model="gemini-2.5-flash-image",
        contents=[prompt, input_image]
    )
    for part in response.parts:
        if hasattr(part, "inline_data") and part.inline_data is not None:
            img_bytes = part.inline_data.data
            img = Image.open(BytesIO(img_bytes))
            return img
    return None

# ------------------------------
@app.get("/")
def api_home():
    return "Welcome to the Image Generation API. Use the /api/generate endpoint to create images."

# ------------------------------
# API Route
# ------------------------------
@app.post("/api/generate")
async def generate_image_api(
    file: UploadFile = File(...),
    prompt1: str = """
    Create a clean Pixar-style cartoon illustration of the pet in the uploaded photo.
Keep the full body, original pose, markings, proportions, and aspect ratio.
Use soft shading and vibrant but natural colors.
Remove the entire background and output a fully transparent PNG with clean edges.
    """,
    prompt2: str = """
    Transform the uploaded pet photo into a bright Pixar-inspired character.
Keep all body parts visible, maintain the pose, markings, and original aspect ratio.
Use glossy highlights and 3D depth while staying natural.
Return a transparent PNG with a perfect cut-out.
    """
):

    # Read uploaded file
    file_bytes = await file.read()
    input_image = Image.open(BytesIO(file_bytes))

    # Generate 2 images
    img1 = generate_image(prompt1, input_image)
    img2 = generate_image(prompt2, input_image)

    response_data = []

    for i, img in enumerate([img1, img2], start=1):

        # Background removal
        img_no_bg = remove_background(img)

        # Preview (small)
        preview_img = img_no_bg.copy()
        preview_img.thumbnail((400, 400))

        # High-resolution (3000 × 3600)
        highres_img = img_no_bg.copy()
        highres_img = highres_img.resize((3000, 3600))

        # Upload to imgBB
        preview_url = upload_to_imgbb(preview_img)
        highres_url = upload_to_imgbb(highres_img)

        response_data.append({
            "variation": i,
            "preview_url": preview_url,
            "highres_url": highres_url
        })

    return JSONResponse(content={"images": response_data})
