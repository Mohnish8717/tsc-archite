from PIL import Image

path1 = "/Users/mohnish/.gemini/antigravity/brain/82370182-61e5-48b5-8eb3-a7c1de8b2128/media__1778953562285.png"
path2 = "/Users/mohnish/.gemini/antigravity/brain/82370182-61e5-48b5-8eb3-a7c1de8b2128/media__1778953572529.png"

img1 = Image.open(path1)
img2 = Image.open(path2)

print(f"Image 1 size: {img1.size}")
print(f"Image 2 size: {img2.size}")

# Resize img2 to match img1 width if they differ
if img1.size[0] != img2.size[0]:
    ratio = img1.size[0] / img2.size[0]
    new_height = int(img2.size[1] * ratio)
    img2 = img2.resize((img1.size[0], new_height), Image.Resampling.LANCZOS)

# Create a new image with total height
total_height = img1.size[1] + img2.size[1]
combined = Image.new('RGB', (img1.size[0], total_height))

# Paste images
combined.paste(img1, (0, 0))
combined.paste(img2, (0, img1.size[1]))

# Save to public directory
output_path = "/Users/mohnish/Downloads/tsc architecture/predictive_ui/public/combined_dashboard.png"
combined.save(output_path)
print(f"Saved combined image to {output_path}")
