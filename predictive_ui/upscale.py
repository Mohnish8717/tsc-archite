from PIL import Image

path = "/Users/mohnish/Downloads/tsc architecture/predictive_ui/public/boardroom_shot.png"
img = Image.open(path).convert("RGBA")

TARGET_WIDTH = 2048
ratio = TARGET_WIDTH / img.width
new_height = int(img.height * ratio)

img_resized = img.resize((TARGET_WIDTH, new_height), Image.Resampling.LANCZOS)
img_resized.save(path, "PNG")
print(f"Upscaled boardroom image to {TARGET_WIDTH}x{new_height}")
