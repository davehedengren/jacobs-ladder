"""
Sprite generator for Jacob's Ladder game.
Uses Gemini API to generate pixel art sprites.
"""

import os
import sys
import time
from io import BytesIO
from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image as PILImage

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    print("ERROR: GEMINI_API_KEY not found in .env")
    sys.exit(1)

os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY

client = genai.Client()
MODEL = "gemini-2.5-flash-image"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SPRITE_DIR = os.path.join(BASE_DIR, "sprites")

REQUEST_DELAY = 4
MAX_RETRIES = 2

STYLE = (
    "8-bit pixel art style, 64x64 pixels, retro NES/SNES aesthetic, "
    "limited color palette, crisp pixels with no anti-aliasing, "
    "transparent background. "
)

ASSETS = [
    (
        "character",
        "A 12-year-old boy character facing forward, wearing a white dress shirt and tie "
        "(like a young LDS deacon), dark slacks, short brown hair, friendly determined expression, "
        "standing pose, side view facing right, full body visible"
    ),
    (
        "character_up",
        "A 12-year-old boy character climbing upward on stairs, wearing a white dress shirt and tie, "
        "dark slacks, short brown hair, happy expression, viewed from the side, "
        "one foot on a higher step, ascending pose"
    ),
    (
        "character_down",
        "A 12-year-old boy character going down stairs looking sad, wearing a white dress shirt and tie, "
        "dark slacks, short brown hair, sad/disappointed expression, viewed from the side, "
        "one foot on a lower step, descending pose"
    ),
    (
        "stair_stone",
        "A single stone brick stair step, ancient Jerusalem style, sandy golden stone, "
        "slightly weathered, top-down angled view showing the top surface and front face, "
        "warm golden-brown color, rectangular shape"
    ),
    (
        "heaven_glow",
        "Heavenly golden clouds with divine light rays streaming down, celestial atmosphere, "
        "warm golden white glow, ethereal clouds, angels in the background, "
        "biblical heaven scene, soft and radiant"
    ),
    (
        "angel",
        "A simple glowing angel figure, white robes, small wings, golden halo, "
        "ethereal glow around the body, peaceful expression, floating pose, "
        "biblical style angel, soft white and gold colors"
    ),
    (
        "ladder_bg",
        "A tall golden ladder or staircase reaching from earth up into the clouds and heaven, "
        "viewed from the side, ancient biblical style, stone steps with golden glow at the top, "
        "dark earth at the bottom, stars in the background, Jacob's Ladder from Genesis, "
        "dramatic perspective going upward"
    ),
]


def generate_sprite(prompt_desc: str, output_path: str) -> bool:
    """Generate a single sprite using Gemini API and save it."""
    full_prompt = (
        f"{STYLE}"
        f"Generate a single game sprite: {prompt_desc}. "
        f"The sprite must be on a completely transparent background. "
        f"No text, no labels, no borders, no UI elements. Just the character/object sprite."
    )

    for attempt in range(MAX_RETRIES):
        try:
            if attempt > 0:
                print(f"  Retry {attempt + 1}/{MAX_RETRIES}...")
                time.sleep(REQUEST_DELAY)

            response = client.models.generate_content(
                model=MODEL,
                contents=[full_prompt],
                config=types.GenerateContentConfig(
                    response_modalities=["IMAGE"],
                ),
            )

            if response.candidates and len(response.candidates) > 0:
                for part in response.candidates[0].content.parts:
                    if part.inline_data is not None:
                        image = PILImage.open(BytesIO(part.inline_data.data))
                        if image.mode != "RGBA":
                            image = image.convert("RGBA")
                        # Save hi-res version
                        hires_path = output_path.replace(".png", "_hires.png")
                        image.save(hires_path, format="PNG")
                        # Resize to game sprite size
                        image = image.resize((64, 64), PILImage.NEAREST)
                        image.save(output_path, format="PNG")
                        return True

            print(f"  No image data in response")
        except Exception as e:
            print(f"  Error: {e}")

    return False


def main():
    os.makedirs(SPRITE_DIR, exist_ok=True)

    total = len(ASSETS)
    generated = 0
    failed = []

    print(f"Generating {total} sprites using Gemini API ({MODEL})")
    print(f"Output: {SPRITE_DIR}/")
    print("=" * 60)

    for filename, description in ASSETS:
        output_path = os.path.join(SPRITE_DIR, f"{filename}.png")
        print(f"  [{generated + 1}/{total}] {filename}...", end=" ", flush=True)

        success = generate_sprite(description, output_path)

        if success:
            print("OK")
            generated += 1
        else:
            print("FAILED")
            failed.append(filename)
            generated += 1

        time.sleep(REQUEST_DELAY)

    print("\n" + "=" * 60)
    print(f"Done! {generated - len(failed)}/{total} sprites generated.")
    if failed:
        print(f"Failed ({len(failed)}): {', '.join(failed)}")
    print(f"Output directory: {SPRITE_DIR}/")


if __name__ == "__main__":
    main()
