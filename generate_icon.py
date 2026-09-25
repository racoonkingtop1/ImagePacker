"""One-off script: generates icon.ico (app icon) using Pillow."""
from PIL import Image, ImageDraw


def make_icon(size: int) -> Image.Image:
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    pad = max(1, size // 16)
    # Rounded-square gradient background (banana yellow -> warm orange).
    top = (255, 205, 70, 255)
    bottom = (255, 149, 62, 255)
    mask = Image.new("L", (size, size), 0)
    mdraw = ImageDraw.Draw(mask)
    radius = size // 4
    mdraw.rounded_rectangle([pad, pad, size - pad, size - pad], radius=radius, fill=255)

    grad = Image.new("RGBA", (size, size))
    for y in range(size):
        t = y / max(1, size - 1)
        r = int(top[0] + (bottom[0] - top[0]) * t)
        g = int(top[1] + (bottom[1] - top[1]) * t)
        b = int(top[2] + (bottom[2] - top[2]) * t)
        for x in range(size):
            grad.putpixel((x, y), (r, g, b, 255))
    img = Image.composite(grad, img, mask)
    draw = ImageDraw.Draw(img)

    # A simple 4-point sparkle (AI-generation motif) in the center, white.
    cx, cy = size / 2, size / 2
    s = size * 0.30
    pts = [
        (cx, cy - s), (cx + s * 0.22, cy - s * 0.22),
        (cx + s, cy), (cx + s * 0.22, cy + s * 0.22),
        (cx, cy + s), (cx - s * 0.22, cy + s * 0.22),
        (cx - s, cy), (cx - s * 0.22, cy - s * 0.22),
    ]
    draw.polygon(pts, fill=(255, 255, 255, 235))

    small_s = size * 0.12
    scx, scy = cx + size * 0.22, cy - size * 0.22
    spts = [
        (scx, scy - small_s), (scx + small_s * 0.22, scy - small_s * 0.22),
        (scx + small_s, scy), (scx + small_s * 0.22, scy + small_s * 0.22),
        (scx, scy + small_s), (scx - small_s * 0.22, scy + small_s * 0.22),
        (scx - small_s, scy), (scx - small_s * 0.22, scy - small_s * 0.22),
    ]
    draw.polygon(spts, fill=(255, 255, 255, 200))

    return img


sizes = [16, 24, 32, 48, 64, 128, 256]
images = [make_icon(s) for s in sizes]
images[-1].save("icon.ico", format="ICO", sizes=[(s, s) for s in sizes])
print("icon.ico written")
