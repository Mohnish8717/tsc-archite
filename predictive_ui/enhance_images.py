from PIL import Image, ImageDraw

path1 = "/Users/mohnish/.gemini/antigravity/brain/82370182-61e5-48b5-8eb3-a7c1de8b2128/media__1778953562285.png"
path2 = "/Users/mohnish/.gemini/antigravity/brain/82370182-61e5-48b5-8eb3-a7c1de8b2128/media__1778953572529.png"

img1 = Image.open(path1).convert("RGBA")
img2 = Image.open(path2).convert("RGBA")

# Target width for Retina displays
TARGET_WIDTH = 2048

# Resize img1
ratio1 = TARGET_WIDTH / img1.width
new_height1 = int(img1.height * ratio1)
img1 = img1.resize((TARGET_WIDTH, new_height1), Image.Resampling.LANCZOS)

# Resize img2
ratio2 = TARGET_WIDTH / img2.width
new_height2 = int(img2.height * ratio2)
img2 = img2.resize((TARGET_WIDTH, new_height2), Image.Resampling.LANCZOS)

# Add a sleek neo-brutalist separator between them
SEPARATOR_HEIGHT = 16
SEPARATOR_COLOR = (0, 0, 0, 255) # Solid black separator

# Calculate total height
total_height = new_height1 + SEPARATOR_HEIGHT + new_height2

# Create new image
combined = Image.new('RGBA', (TARGET_WIDTH, total_height), (0, 0, 0, 0))

# Paste images
combined.paste(img1, (0, 0))

# Draw separator
draw = ImageDraw.Draw(combined)
draw.rectangle([(0, new_height1), (TARGET_WIDTH, new_height1 + SEPARATOR_HEIGHT)], fill=SEPARATOR_COLOR)

# Paste second image
combined.paste(img2, (0, new_height1 + SEPARATOR_HEIGHT))

# Save to public directory
output_path = "/Users/mohnish/Downloads/tsc architecture/predictive_ui/public/combined_dashboard.png"
combined.save(output_path, "PNG")
print(f"Saved enhanced combined image to {output_path} with size {combined.size}")
