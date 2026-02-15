"""
Sample Receipt Generator
Creates realistic sample receipt images for testing the pipeline.
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os


def create_sample_receipt(output_path="sample_receipt.png"):
    """Generate a realistic-looking grocery receipt image."""

    # Receipt dimensions
    width = 400
    line_height = 28
    items = [
        ("Whole Milk 1 Gal", 4.29, 1),
        ("Organic Eggs Dozen", 5.99, 1),
        ("Chicken Breast 2lb", 8.49, 1),
        ("Fresh Bananas", 1.29, 2),
        ("Whole Wheat Bread", 3.49, 1),
        ("Cheddar Cheese 8oz", 4.99, 1),
        ("Greek Yogurt 32oz", 5.49, 1),
        ("Orange Juice 64oz", 4.99, 1),
        ("Potato Chips BBQ", 3.99, 1),
        ("Ground Coffee 12oz", 8.99, 1),
        ("Fresh Tomatoes 1lb", 2.99, 1),
        ("Pasta Spaghetti", 1.49, 2),
        ("Marinara Sauce", 3.29, 1),
        ("Ice Cream Vanilla", 5.99, 1),
        ("Paper Towels 6pk", 7.99, 1),
        ("Dish Soap", 3.49, 1),
        ("Apple Red Gala 1lb", 2.49, 1),
    ]

    # Calculate totals
    subtotal = sum(price * qty for _, price, qty in items)
    tax = round(subtotal * 0.08, 2)
    total = round(subtotal + tax, 2)

    # Header + items + footer lines
    header_lines = 8
    footer_lines = 12
    num_lines = header_lines + len(items) + footer_lines
    height = num_lines * line_height + 60

    # Create image
    img = Image.new('RGB', (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Use a basic font
    try:
        font = ImageFont.truetype("arial.ttf", 16)
        font_bold = ImageFont.truetype("arialbd.ttf", 18)
        font_small = ImageFont.truetype("arial.ttf", 13)
    except OSError:
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 16)
            font_bold = ImageFont.truetype("DejaVuSans-Bold.ttf", 18)
            font_small = ImageFont.truetype("DejaVuSans.ttf", 13)
        except OSError:
            font = ImageFont.load_default()
            font_bold = font
            font_small = font

    y = 20

    # ── Header ──
    draw.text((width // 2 - 80, y), "FRESH MART", fill=(0, 0, 0), font=font_bold)
    y += line_height + 2
    draw.text((width // 2 - 100, y), "123 Main Street, City", fill=(80, 80, 80), font=font_small)
    y += line_height - 5
    draw.text((width // 2 - 80, y), "Tel: 555-0123", fill=(80, 80, 80), font=font_small)
    y += line_height - 2
    draw.text((width // 2 - 90, y), "02/15/2026  14:32", fill=(80, 80, 80), font=font_small)
    y += line_height + 2
    draw.line([(20, y), (width - 20, y)], fill=(0, 0, 0), width=1)
    y += 10
    draw.text((20, y), "ITEM", fill=(0, 0, 0), font=font_small)
    draw.text((width - 80, y), "PRICE", fill=(0, 0, 0), font=font_small)
    y += line_height - 5
    draw.line([(20, y), (width - 20, y)], fill=(0, 0, 0), width=1)
    y += 8

    # ── Items ──
    for name, price, qty in items:
        item_total = price * qty
        if qty > 1:
            draw.text((20, y), f"{qty}x {name}", fill=(0, 0, 0), font=font)
        else:
            draw.text((20, y), name, fill=(0, 0, 0), font=font)
        draw.text((width - 80, y), f"${item_total:.2f}", fill=(0, 0, 0), font=font)
        y += line_height

    # ── Footer ──
    y += 5
    draw.line([(20, y), (width - 20, y)], fill=(0, 0, 0), width=1)
    y += 10

    draw.text((20, y), "SUBTOTAL", fill=(0, 0, 0), font=font)
    draw.text((width - 80, y), f"${subtotal:.2f}", fill=(0, 0, 0), font=font)
    y += line_height

    draw.text((20, y), "TAX (8%)", fill=(80, 80, 80), font=font)
    draw.text((width - 80, y), f"${tax:.2f}", fill=(80, 80, 80), font=font)
    y += line_height

    draw.line([(20, y), (width - 20, y)], fill=(0, 0, 0), width=2)
    y += 8

    draw.text((20, y), "TOTAL", fill=(0, 0, 0), font=font_bold)
    draw.text((width - 85, y), f"${total:.2f}", fill=(0, 0, 0), font=font_bold)
    y += line_height + 5

    draw.line([(20, y), (width - 20, y)], fill=(0, 0, 0), width=2)
    y += 15

    draw.text((width // 2 - 60, y), "CASH", fill=(0, 0, 0), font=font)
    draw.text((width - 85, y), f"${total + 5 - (total % 5):.2f}", fill=(0, 0, 0), font=font)
    y += line_height

    change = round(total + 5 - (total % 5) - total, 2)
    draw.text((width // 2 - 60, y), "CHANGE", fill=(0, 0, 0), font=font)
    draw.text((width - 80, y), f"${change:.2f}", fill=(0, 0, 0), font=font)
    y += line_height + 10

    draw.text((width // 2 - 90, y), "Thank you for shopping!", fill=(80, 80, 80), font=font_small)
    y += line_height
    draw.text((width // 2 - 70, y), "Please come again", fill=(80, 80, 80), font=font_small)

    # Add slight noise for realism
    img_array = np.array(img)
    noise = np.random.normal(0, 3, img_array.shape).astype(np.int16)
    img_array = np.clip(img_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(img_array)

    # Save
    img.save(output_path)
    print(f"Sample receipt saved to: {output_path}")
    return output_path


if __name__ == "__main__":
    create_sample_receipt()
    print("Done! Upload this image to the ReceiptIQ app.")

