import os
from PIL import Image, ImageDraw, ImageFilter
import math

os.makedirs("assets", exist_ok=True)

# 1. Generate Super-Sampled 2048x2048 Image for Silky-Smooth Anti-Aliasing
canvas_size = 2048
img = Image.new("RGBA", (canvas_size, canvas_size), (0, 0, 0, 0))
draw = ImageDraw.Draw(img)

# Center and radius
cx, cy = canvas_size // 2, canvas_size // 2
radius = int(canvas_size * 0.44)

# Create high-end radial/conical gradient background badge
bg_layer = Image.new("RGBA", (canvas_size, canvas_size), (0, 0, 0, 0))
bg_draw = ImageDraw.Draw(bg_layer)

# Outer glow / shadow
for r in range(radius + 60, radius, -2):
    alpha = int(40 * (1 - (r - radius) / 60))
    bg_draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(30, 58, 138, alpha))

# Base deep sapphire circle
bg_draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], fill=(15, 23, 42, 255))

# Top-to-bottom inner gradient
for i in range(radius * 2):
    y = cy - radius + i
    dx = math.sqrt(max(0, radius**2 - (y - cy)**2))
    factor = i / (radius * 2)
    # Gradient from vibrant royal blue (37, 99, 235) to deep dark navy (10, 15, 30)
    r_col = int(37 * (1 - factor) + 10 * factor)
    g_col = int(99 * (1 - factor) + 18 * factor)
    b_col = int(235 * (1 - factor) + 48 * factor)
    bg_draw.line([cx - dx, y, cx + dx, y], fill=(r_col, g_col, b_col, 255), width=2)

# Subtle metallic border ring
border_layer = Image.new("RGBA", (canvas_size, canvas_size), (0, 0, 0, 0))
b_draw = ImageDraw.Draw(border_layer)
b_draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], outline=(96, 165, 250, 200), width=24)
b_draw.ellipse([cx - radius + 16, cy - radius + 16, cx + radius - 16, cy + radius - 16], outline=(30, 64, 175, 120), width=12)

img = Image.alpha_composite(img, bg_layer)
img = Image.alpha_composite(img, border_layer)

# Foreground Graphic Layer
fg = Image.new("RGBA", (canvas_size, canvas_size), (0, 0, 0, 0))
fg_draw = ImageDraw.Draw(fg)

# Subtle Grid lines
for y_pos in [cy + 320, cy + 120, cy - 80, cy - 280]:
    dx = math.sqrt(max(0, (radius - 120)**2 - (y_pos - cy)**2))
    fg_draw.line([(cx - dx, y_pos), (cx + dx, y_pos)], fill=(255, 255, 255, 25), width=8)

# Vertical Volume bars at bottom
bars = [
    (cx - 420, cy + 180, cx - 340, cy + 420, (56, 189, 248, 140)),
    (cx - 260, cy + 80, cx - 180, cy + 420, (56, 189, 248, 180)),
    (cx - 100, cy + 240, cx - 20, cy + 420, (56, 189, 248, 140)),
    (cx + 60, cy - 20, cx + 140, cy + 420, (52, 211, 153, 200)),
    (cx + 220, cy - 140, cx + 300, cy + 420, (52, 211, 153, 230)),
]
for x1, y1, x2, y2, color in bars:
    fg_draw.rounded_rectangle([x1, y1, x2, y2], radius=24, fill=color)

# Smooth Glowing Trajectory Path
curve_points = [
    (cx - 480, cy + 240),
    (cx - 280, cy + 60),
    (cx - 80, cy + 140),
    (cx + 140, cy - 180),
    (cx + 420, cy - 360)
]

# Thick glowing backdrop curve
for w, a in [(72, 40), (52, 90), (36, 180)]:
    fg_draw.line(curve_points, fill=(0, 230, 118, a), width=w, joint="curve")

# Crisp white/cyan core line
fg_draw.line(curve_points, fill=(255, 255, 255, 255), width=28, joint="curve")

# Arrow Head
tip = (cx + 460, cy - 390)
p1 = (cx + 340, cy - 390)
p2 = (cx + 460, cy - 270)
fg_draw.line([p1, tip, p2], fill=(255, 255, 255, 255), width=32, joint="curve")

# Node Rings & Pulsing Data Points
for pt in curve_points[:-1]:
    # Outer white ring
    fg_draw.ellipse([pt[0] - 44, pt[1] - 44, pt[0] + 44, pt[1] + 44], fill=(255, 255, 255, 255))
    # Inner emerald/cyan core
    fg_draw.ellipse([pt[0] - 24, pt[1] - 24, pt[0] + 24, pt[1] + 24], fill=(14, 165, 233, 255))

img = Image.alpha_composite(img, fg)

# Downsample with high-quality Lanczos filter to 512x512 for crystal-clear smooth edges
logo_512 = img.resize((512, 512), Image.Resampling.LANCZOS)
logo_512.save("assets/logo.png", "PNG", optimize=True)

# Also save 192x192 and 32x32 for favicon/shortcut
logo_192 = img.resize((192, 192), Image.Resampling.LANCZOS)
logo_192.save("assets/icon_192.png", "PNG", optimize=True)

logo_32 = img.resize((32, 32), Image.Resampling.LANCZOS)
logo_32.save("assets/favicon.ico", "ICO")

print("Generated crystal-clear anti-aliased logo at assets/logo.png, icon_192.png, and favicon.ico!")
