"""
Spending Analysis Module
Computes detailed spending analytics, detects anomalies, and identifies patterns.
"""

from typing import List, Dict, Optional
import numpy as np
from datetime import datetime


class SpendingAnalyzer:
    """Analyzes spending patterns from categorized receipt data."""

    # Average spending distribution benchmarks (percentage)
    SPENDING_BENCHMARKS = {
        "Dairy": 12.0,
        "Meat & Seafood": 20.0,
        "Fruits & Vegetables": 18.0,
        "Bakery": 8.0,
        "Beverages": 10.0,
        "Snacks": 8.0,
        "Canned & Packaged": 10.0,
        "Frozen Foods": 5.0,
        "Household": 6.0,
        "Health & Personal Care": 3.0,
        "Beauty & Cosmetics": 5.0,
        "Other": 0.0,
    }

    def __init__(self):
        pass

    def analyze_spending(self, categorization_data: dict, currency: str = "$") -> dict:
        """
        Comprehensive spending analysis.
        Returns detailed analytics, insights, and anomaly detection.
        """
        self.currency = currency
        category_percentages = categorization_data.get("category_percentages", {})
        categorized_items = categorization_data.get("categorized_items", [])
        grand_total = categorization_data.get("grand_total", 0)

        # Basic statistics
        basic_stats = self._compute_basic_stats(categorized_items, grand_total)

        # Overspending analysis
        overspending = self._detect_overspending(category_percentages)

        # Price anomalies
        price_anomalies = self._detect_price_anomalies(categorized_items)

        # Spending insights
        insights = self._generate_insights(
            category_percentages, categorized_items, grand_total, overspending
        )

        # Top items
        top_expensive = self._get_top_items(categorized_items, n=5)

        # Savings opportunities
        savings = self._identify_savings(categorized_items, category_percentages)

        return {
            "basic_stats": basic_stats,
            "overspending_alerts": overspending,
            "price_anomalies": price_anomalies,
            "insights": insights,
            "top_expensive_items": top_expensive,
            "savings_opportunities": savings,
            "health_score": self._calculate_health_score(category_percentages, overspending),
            "currency": currency,
        }

    def _compute_basic_stats(self, items: List[dict], grand_total: float) -> dict:
        """Compute basic spending statistics."""
        if not items:
            return {
                "total_items": 0, "grand_total": 0, "avg_item_price": 0,
                "median_price": 0, "min_price": 0, "max_price": 0,
                "price_std": 0,
            }

        prices = [item["total"] for item in items]
        return {
            "total_items": len(items),
            "grand_total": round(grand_total, 2),
            "avg_item_price": round(np.mean(prices), 2),
            "median_price": round(float(np.median(prices)), 2),
            "min_price": round(min(prices), 2),
            "max_price": round(max(prices), 2),
            "price_std": round(float(np.std(prices)), 2),
            "total_quantity": sum(item.get("quantity", 1) for item in items),
        }

    def _detect_overspending(self, category_percentages: dict) -> List[dict]:
        """Detect categories where spending exceeds benchmarks significantly."""
        alerts = []

        for category, data in category_percentages.items():
            benchmark = self.SPENDING_BENCHMARKS.get(category, 10.0)
            actual = data["percentage"]

            if benchmark > 0:
                deviation = ((actual - benchmark) / benchmark) * 100
            else:
                deviation = actual * 10  # High deviation for unexpected categories

            if actual > benchmark * 1.3:  # 30% over benchmark
                severity = "high" if deviation > 50 else "medium"
                alerts.append({
                    "category": category,
                    "actual_pct": actual,
                    "benchmark_pct": benchmark,
                    "deviation_pct": round(deviation, 1),
                    "amount": data["amount"],
                    "severity": severity,
                    "message": f"{category} spending is {round(deviation, 1)}% above typical levels"
                })

        alerts.sort(key=lambda x: x["deviation_pct"], reverse=True)
        return alerts

    def _detect_price_anomalies(self, items: List[dict]) -> List[dict]:
        """Detect items with unusually high or low prices."""
        if len(items) < 3:
            return []

        prices = [item["total"] for item in items]
        mean_price = np.mean(prices)
        std_price = np.std(prices)

        anomalies = []
        for item in items:
            if std_price > 0:
                z_score = (item["total"] - mean_price) / std_price
                if abs(z_score) > 1.5:
                    anomalies.append({
                        "item": item["name"],
                        "price": item["total"],
                        "z_score": round(z_score, 2),
                        "type": "expensive" if z_score > 0 else "cheap",
                        "message": f"{'Unusually expensive' if z_score > 0 else 'Unusually cheap'}: {item['name']} at {self.currency}{item['total']:,.2f}"
                    })

        return anomalies

    def _generate_insights(self, category_percentages: dict,
                           items: List[dict], grand_total: float,
                           overspending: List[dict]) -> List[dict]:
        """Generate actionable spending insights."""
        insights = []

        if not items:
            return [{"type": "info", "icon": "ℹ️", "message": "No items found to analyze."}]

        # Top spending category
        if category_percentages:
            top_cat = list(category_percentages.keys())[0]
            top_data = category_percentages[top_cat]
            insights.append({
                "type": "info",
                "icon": "📊",
                "message": f"Your highest spending category is **{top_cat}** at **{top_data['percentage']}%** ({self.currency}{top_data['amount']:,.2f}) of your total."
            })

        # Number of categories
        num_categories = len(category_percentages)
        if num_categories >= 4:
            insights.append({
                "type": "positive",
                "icon": "✅",
                "message": f"Good spending diversity! You shopped across **{num_categories} categories**, indicating a balanced purchase pattern."
            })
        elif num_categories <= 2:
            insights.append({
                "type": "warning",
                "icon": "⚠️",
                "message": f"Limited variety: Only **{num_categories} categories** detected. Consider diversifying your purchases."
            })

        # Overspending alerts
        if overspending:
            for alert in overspending[:3]:
                insights.append({
                    "type": "warning",
                    "icon": "🔴" if alert["severity"] == "high" else "🟡",
                    "message": alert["message"]
                })

        # Average price insight
        avg_price = np.mean([item["total"] for item in items])
        if avg_price > 15:
            insights.append({
                "type": "info",
                "icon": "💰",
                "message": f"Your average item cost is **{self.currency}{avg_price:,.2f}**, which is on the higher side. Look for deals or bulk options."
            })
        elif avg_price < 5:
            insights.append({
                "type": "positive",
                "icon": "👍",
                "message": f"Your average item cost of **{self.currency}{avg_price:,.2f}** shows budget-conscious shopping!"
            })

        # Healthy eating check
        healthy_cats = ["Fruits & Vegetables", "Dairy"]
        healthy_pct = sum(
            category_percentages.get(cat, {}).get("percentage", 0) for cat in healthy_cats
        )
        if healthy_pct > 30:
            insights.append({
                "type": "positive",
                "icon": "🥗",
                "message": f"**{healthy_pct:.1f}%** of your spending is on healthy categories (fruits, vegetables, dairy). Great choice!"
            })
        elif healthy_pct < 15:
            insights.append({
                "type": "suggestion",
                "icon": "💡",
                "message": f"Only **{healthy_pct:.1f}%** spent on fruits, vegetables & dairy. Consider adding more nutritious items."
            })

        # Snack spending check
        snack_pct = category_percentages.get("Snacks", {}).get("percentage", 0)
        bev_pct = category_percentages.get("Beverages", {}).get("percentage", 0)
        if snack_pct + bev_pct > 25:
            insights.append({
                "type": "warning",
                "icon": "🍫",
                "message": f"**{snack_pct + bev_pct:.1f}%** spent on snacks and beverages. These are often impulse purchases — try cutting back!"
            })

        return insights

    def _get_top_items(self, items: List[dict], n: int = 5) -> List[dict]:
        """Get the most expensive items."""
        sorted_items = sorted(items, key=lambda x: x["total"], reverse=True)
        return sorted_items[:n]

    def _identify_savings(self, items: List[dict],
                          category_percentages: dict) -> List[dict]:
        """Identify potential savings opportunities."""
        suggestions = []

        # Find high-spending categories that could be reduced
        for category, data in category_percentages.items():
            if data["percentage"] > 25:
                potential_saving = round(data["amount"] * 0.15, 2)
                suggestions.append({
                    "category": category,
                    "current_spend": data["amount"],
                    "potential_saving": potential_saving,
                    "tip": f"Reducing {category} spending by 15% could save {self.currency}{potential_saving:,.2f}"
                })

        # Check for expensive individual items
        if items:
            prices = [item["total"] for item in items]
            p75 = float(np.percentile(prices, 75))
            expensive = [item for item in items if item["total"] > p75 * 1.5]
            for item in expensive[:3]:
                suggestions.append({
                    "category": item.get("category", "General"),
                    "current_spend": item["total"],
                    "potential_saving": round(item["total"] * 0.2, 2),
                    "tip": f"Look for alternatives to **{item['name']}** ({self.currency}{item['total']:,.2f}) — store brands can save ~20%"
                })

        return suggestions

    def _calculate_health_score(self, category_percentages: dict,
                                 overspending: List[dict]) -> dict:
        """Calculate an overall spending health score (0-100)."""
        score = 70  # Start with base score

        # Reward for healthy categories
        healthy_pct = sum(
            category_percentages.get(cat, {}).get("percentage", 0)
            for cat in ["Fruits & Vegetables", "Dairy"]
        )
        score += min(healthy_pct * 0.3, 15)

        # Penalize overspending
        for alert in overspending:
            if alert["severity"] == "high":
                score -= 8
            else:
                score -= 4

        # Reward category diversity
        num_cats = len(category_percentages)
        if num_cats >= 4:
            score += 5
        elif num_cats <= 2:
            score -= 5

        # Penalize heavy snack/beverage spending
        junk_pct = sum(
            category_percentages.get(cat, {}).get("percentage", 0)
            for cat in ["Snacks", "Beverages"]
        )
        if junk_pct > 20:
            score -= 5

        score = max(0, min(100, round(score)))

        if score >= 80:
            label = "Excellent"
            color = "#22C55E"
        elif score >= 60:
            label = "Good"
            color = "#60A5FA"
        elif score >= 40:
            label = "Fair"
            color = "#FBBF24"
        else:
            label = "Needs Improvement"
            color = "#EF4444"

        return {"score": score, "label": label, "color": color}


