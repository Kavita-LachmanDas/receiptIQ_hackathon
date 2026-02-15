"""
Expense Categorization Module
Classifies receipt items into spending categories using keyword matching.
Handles common receipt abbreviations (BNLS=boneless, CHICK=chicken, etc.)
"""

from typing import List, Dict
import re


class ExpenseCategorizer:
    """Categorizes receipt items into meaningful spending categories."""

    # Comprehensive category keyword mapping (includes receipt abbreviations)
    CATEGORY_KEYWORDS = {
        "Dairy": [
            "milk", "cheese", "yogurt", "yoghurt", "butter", "cream", "curd",
            "paneer", "ghee", "whey", "cottage", "mozzarella", "cheddar",
            "parmesan", "parm", "ricotta", "sour cream", "creamer",
            "ice cream", "frozen yogurt", "kefir", "buttermilk",
            "half & half", "half and half", "vanil", "vanilla",
            # Receipt abbreviations
            "ff lt vanil", "parm shrd",
        ],
        "Meat & Seafood": [
            "chicken", "chick", "beef", "pork", "lamb", "turkey", "fish",
            "salmon", "tuna", "shrimp", "prawn", "crab", "lobster", "steak",
            "bacon", "ham", "sausage", "salami", "pepperoni", "meatball",
            "ground", "wing", "thigh", "breast", "drumstick", "ribs", "chop",
            "tilapia", "cod", "halibut", "catfish", "meat", "mutton",
            "burger", "burg", "imposs burg", "impossible",
            # Receipt abbreviations
            "bnls", "chckn", "brst", "boneless",
        ],
        "Fruits & Vegetables": [
            "apple", "banana", "orange", "grape", "strawberry", "blueberry",
            "raspberry", "mango", "pineapple", "watermelon", "melon", "peach",
            "pear", "plum", "cherry", "kiwi", "lemon", "lime", "avocado",
            "tomato", "potato", "onion", "garlic", "carrot", "broccoli",
            "spinach", "lettuce", "cucumber", "pepper", "celery", "corn",
            "mushroom", "cabbage", "cauliflower", "zucchini", "squash",
            "eggplant", "beans", "peas", "fruit", "vegetable", "veg",
            "salad", "herb", "cilantro", "basil", "organic", "fresh",
            "shallot", "green bell", "bell pepper", "red pepper",
            "persian", "limes", "diced tomato",
            # Receipt abbreviations
            "grn", "org", "frsh", "tom", "carrots",
        ],
        "Bakery": [
            "bread", "bagel", "muffin", "cake", "cookie", "pie", "pastry",
            "croissant", "donut", "doughnut", "roll", "bun", "toast",
            "tortilla", "wrap", "baguette", "sourdough", "rye", "wheat",
            "loaf", "bakery", "danish", "scone", "brownie", "cupcake",
            # Receipt abbreviations
            "whl wht", "w/g wheat", "wht brd",
        ],
        "Beverages": [
            "water", "juice", "soda", "pop", "cola", "pepsi", "coke",
            "sprite", "fanta", "coffee", "tea", "latte", "cappuccino",
            "espresso", "smoothie", "shake", "drink", "beverage", "beer",
            "wine", "liquor", "vodka", "whiskey", "rum", "gin", "tequila",
            "champagne", "ale", "lager", "energy", "redbull", "monster",
            "gatorade", "lemonade", "kombucha",
        ],
        "Snacks": [
            "chip", "crisp", "cracker", "pretzel", "popcorn", "nut",
            "almond", "cashew", "peanut", "walnut", "pistachio", "trail mix",
            "granola", "bar", "candy", "chocolate", "gummy", "sweet",
            "snack", "nachos", "jerky", "dried", "seeds", "sunflower",
        ],
        "Canned & Packaged": [
            "can", "canned", "soup", "broth", "stock", "sauce", "paste",
            "ketchup", "mustard", "mayo", "mayonnaise", "dressing", "vinegar",
            "oil", "olive", "coconut", "vegetable oil", "spray", "seasoning",
            "spice", "salt", "pepper", "sugar", "flour", "rice", "pasta",
            "noodle", "cereal", "oat", "oatmeal", "pancake", "syrup",
            "honey", "jam", "jelly", "peanut butter", "nutella",
            "jif", "creamy", "tom/paste", "tomato paste",
            # Receipt abbreviations
            "pac broth", "chckn ls", "hz tomato",
        ],
        "Frozen Foods": [
            "frozen", "pizza", "fries", "nugget", "waffle", "ice",
            "popsicle", "frozen dinner", "freezer", "frost", "tv dinner",
            "hot pocket", "lean cuisine", "stouffer",
        ],
        "Household": [
            "paper", "towel", "tissue", "napkin", "toilet", "soap", "detergent",
            "cleaner", "bleach", "sponge", "trash", "bag", "wrap", "foil",
            "container", "plate", "cup", "utensil", "brush", "mop", "broom",
            "laundry", "fabric", "softener", "dishwash", "dish",
        ],
        "Health & Personal Care": [
            "medicine", "vitamin", "supplement", "bandage", "pain", "cold",
            "flu", "allergy", "antacid", "shampoo", "conditioner", "lotion",
            "deodorant", "toothpaste", "toothbrush", "floss", "razor",
            "shave", "cotton", "sunscreen", "moisturizer",
        ],
        "Beauty & Cosmetics": [
            "beauty", "blender", "beauty blender", "makeup", "make up",
            "lipstick", "lip", "foundation", "concealer", "mascara",
            "eyeliner", "eyeshadow", "eye shadow", "blush", "bronzer",
            "primer", "setting spray", "compact", "powder", "pwdr", "cmpct",
            "loreal", "l'oreal", "maybelline", "mybelline", "revlon",
            "nyx", "mac", "lakme", "garnier", "nivea", "dove",
            "infallible", "infallble", "matte", "mtte",
            "nail", "nail polish", "nail color",
            "perfume", "fragrance", "cologne", "body spray", "body mist",
            "face wash", "face cream", "cleanser", "toner", "serum",
            "cosmetic", "zerox", "lftr", "shne", "amber",
            "fit me", "fit me cmpct", "highlighter", "contour",
            "kajal", "kohl", "brow", "lash",
            "bb cream", "cc cream", "skin care", "skincare",
        ],
    }

    # Category display icons (separate from keywords to avoid encoding issues)
    CATEGORY_ICONS = {
        "Dairy": "Dairy",
        "Meat & Seafood": "Meat & Seafood",
        "Fruits & Vegetables": "Fruits & Vegetables",
        "Bakery": "Bakery",
        "Beverages": "Beverages",
        "Snacks": "Snacks",
        "Canned & Packaged": "Canned & Packaged",
        "Frozen Foods": "Frozen Foods",
        "Household": "Household",
        "Health & Personal Care": "Health & Personal Care",
        "Beauty & Cosmetics": "Beauty & Cosmetics",
        "Other": "Other",
    }

    # Category colors for charts
    CATEGORY_COLORS = {
        "Dairy": "#60A5FA",
        "Meat & Seafood": "#F87171",
        "Fruits & Vegetables": "#34D399",
        "Bakery": "#FBBF24",
        "Beverages": "#A78BFA",
        "Snacks": "#FB923C",
        "Canned & Packaged": "#F472B6",
        "Frozen Foods": "#22D3EE",
        "Household": "#94A3B8",
        "Health & Personal Care": "#E879F9",
        "Beauty & Cosmetics": "#FF6B9D",
        "Other": "#6B7280",
    }

    def __init__(self):
        # Build reverse lookup for faster categorization
        self._keyword_to_category = {}
        for category, keywords in self.CATEGORY_KEYWORDS.items():
            for keyword in keywords:
                self._keyword_to_category[keyword.lower()] = category

    def categorize_item(self, item_name: str) -> str:
        """Categorize a single item by its name."""
        name_lower = item_name.lower().strip()

        # Try exact substring matching (longest keywords first for specificity)
        sorted_keywords = sorted(self._keyword_to_category.keys(), key=len, reverse=True)
        for keyword in sorted_keywords:
            if keyword in name_lower:
                return self._keyword_to_category[keyword]

        # Try matching individual words
        words = re.findall(r'[a-zA-Z]+', name_lower)
        for word in words:
            if len(word) < 3:
                continue
            for keyword, category in self._keyword_to_category.items():
                if word == keyword or (len(word) >= 4 and word in keyword) or (len(keyword) >= 4 and keyword in word):
                    return category

        return "Other"

    def categorize_items(self, items: List[dict]) -> List[dict]:
        """Add category to each item in the list."""
        categorized = []
        for item in items:
            item_copy = item.copy()
            item_copy["category"] = self.categorize_item(item["name"])
            categorized.append(item_copy)
        return categorized

    def get_category_summary(self, categorized_items: List[dict]) -> dict:
        """Calculate spending summary per category."""
        summary = {}

        for item in categorized_items:
            category = item["category"]
            if category not in summary:
                summary[category] = {
                    "category": category,
                    "items": [],
                    "total_amount": 0.0,
                    "item_count": 0,
                    "color": self.CATEGORY_COLORS.get(category, "#6B7280")
                }

            summary[category]["items"].append(item)
            summary[category]["total_amount"] += item["total"]
            summary[category]["item_count"] += 1

        for cat in summary:
            summary[cat]["total_amount"] = round(summary[cat]["total_amount"], 2)

        return summary

    def get_category_percentages(self, category_summary: dict) -> dict:
        """Calculate percentage of total spending per category."""
        grand_total = sum(cat["total_amount"] for cat in category_summary.values())

        if grand_total == 0:
            return {}

        percentages = {}
        for category, data in category_summary.items():
            percentages[category] = {
                "amount": data["total_amount"],
                "percentage": round((data["total_amount"] / grand_total) * 100, 1),
                "item_count": data["item_count"],
                "color": data["color"]
            }

        percentages = dict(sorted(percentages.items(),
                                   key=lambda x: x[1]["percentage"],
                                   reverse=True))
        return percentages

    def process(self, parsed_data: dict) -> dict:
        """Full categorization pipeline."""
        items = parsed_data.get("items", [])

        categorized_items = self.categorize_items(items)
        category_summary = self.get_category_summary(categorized_items)
        category_percentages = self.get_category_percentages(category_summary)
        grand_total = sum(cat["total_amount"] for cat in category_summary.values())

        return {
            "categorized_items": categorized_items,
            "category_summary": category_summary,
            "category_percentages": category_percentages,
            "grand_total": round(grand_total, 2),
            "total_categories": len(category_summary),
            "total_items": len(categorized_items)
        }
