import os
from PIL import Image, ImageDraw

os.makedirs("assets", exist_ok=True)

# Generate a high-resolution 512x512 modern blue finance logo
size = (512, 512)
img = Image.new("RGBA", size, (0, 0, 0, 0))
draw = ImageDraw.Draw(img)

# Background circle with deep vibrant gradient blue
# Outer soft circle
draw.ellipse([16, 16, 496, 496], fill=(26, 86, 219, 255))
# Inner highlighted circle
draw.ellipse([32, 32, 480, 480], fill=(30, 64, 175, 255))

# Draw crisp white geometric financial growth lines & portfolio nodes
# Grid lines (subtle)
for y in [180, 260, 340]:
    draw.line([(80, y), (432, y)], fill=(255, 255, 255, 40), width=3)

# Upward trending chart line
points = [(100, 350), (180, 300), (270, 240), (350, 160), (410, 120)]
draw.line(points, fill=(255, 255, 255, 255), width=16)

# Arrow head at the top right
arrow_tip = (425, 110)
arrow_p1 = (370, 110)
arrow_p2 = (425, 165)
draw.line([arrow_p1, arrow_tip, arrow_p2], fill=(255, 255, 255, 255), width=16)

# Portfolio Nodes (white circular data points with blue centers)
for pt in points[:-1]:
    draw.ellipse([pt[0]-16, pt[1]-16, pt[0]+16, pt[1]+16], fill=(255, 255, 255, 255))
    draw.ellipse([pt[0]-8, pt[1]-8, pt[0]+8, pt[1]+8], fill=(30, 64, 175, 255))

# Candlestick / Bar accents at bottom
bars = [(130, 380, 155, 420), (220, 340, 245, 420), (310, 290, 335, 420)]
for b in bars:
    draw.rectangle([b[0], b[1], b[2], b[3]], fill=(255, 255, 255, 180))

img.save("assets/logo.png", "PNG")
print("Logo generated at assets/logo.png")
