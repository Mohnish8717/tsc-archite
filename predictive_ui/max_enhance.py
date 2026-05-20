from PIL import Image, ImageFilter, ImageEnhance

# Original source files (to avoid compounding artifacting from previous passes)
boardroom_src = "/Users/mohnish/.gemini/antigravity/brain/82370182-61e5-48b5-8eb3-a7c1de8b2128/media__1778953562285.png"
oasis_src = "/Users/mohnish/.gemini/antigravity/brain/82370182-61e5-48b5-8eb3-a7c1de8b2128/media__1778953572529.png"

TARGET_WIDTH = 4096 # Massive 4K width for maximum crispness

def enhance_max(src_path, dest_path):
    img = Image.open(src_path).convert("RGBA")
    
    # 1. 4K Lanczos Upscale
    ratio = TARGET_WIDTH / img.width
    new_height = int(img.height * ratio)
    img = img.resize((TARGET_WIDTH, new_height), Image.Resampling.LANCZOS)
    
    # 2. Contrast Boost (make text punchier)
    enhancer_contrast = ImageEnhance.Contrast(img)
    img = enhancer_contrast.enhance(1.05)
    
    # 3. Base Sharpness Boost
    enhancer_sharpness = ImageEnhance.Sharpness(img)
    img = enhancer_sharpness.enhance(1.5)
    
    # 4. Aggressive Unsharp Mask to crispen edges
    img = img.filter(ImageFilter.UnsharpMask(radius=3, percent=200, threshold=1))
    
    img.save(dest_path, "PNG", optimize=True)
    print(f"Max Enhanced {dest_path} to {TARGET_WIDTH}x{new_height}")

enhance_max(boardroom_src, "/Users/mohnish/Downloads/tsc architecture/predictive_ui/public/boardroom_shot.png")
enhance_max(oasis_src, "/Users/mohnish/Downloads/tsc architecture/predictive_ui/public/oasis_shot.png")
