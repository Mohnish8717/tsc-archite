from PIL import Image, ImageFilter

paths = [
    "/Users/mohnish/Downloads/tsc architecture/predictive_ui/public/boardroom_shot.png",
    "/Users/mohnish/Downloads/tsc architecture/predictive_ui/public/oasis_shot.png"
]

for path in paths:
    img = Image.open(path).convert("RGBA")
    # Apply Unsharp Mask: radius=2, percent=150, threshold=3
    sharpened = img.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
    sharpened.save(path, "PNG")
    print(f"Sharpened {path}")
