"""
LLM Integration Module
Uses Google Gemini to provide personalized financial advice based on spending data.
"""

import json
from typing import Dict, Optional
import os


class LLMAdvisor:
    """Generates AI-powered financial advice using Google Gemini."""

    def __init__(self, api_key: str = None):
        """Initialize Gemini LLM."""
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY", "")
        self.model = None

        if self.api_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                self.model = genai.GenerativeModel('gemini-2.0-flash')
            except Exception as e:
                print(f"Warning: Could not initialize Gemini: {e}")
                self.model = None

    def _build_prompt(self, categorization_data: dict, analysis_data: dict, currency: str = "$") -> str:
        """Build a detailed prompt for the LLM."""
        # Extract key data
        grand_total = categorization_data.get("grand_total", 0)
        total_items = categorization_data.get("total_items", 0)
        category_pcts = categorization_data.get("category_percentages", {})
        basic_stats = analysis_data.get("basic_stats", {})
        overspending = analysis_data.get("overspending_alerts", [])
        top_items = analysis_data.get("top_expensive_items", [])
        health_score = analysis_data.get("health_score", {})

        # Determine currency name for context
        currency_name = "Pakistani Rupees (Rs)" if currency == "Rs" else "US Dollars ($)"

        # Format category breakdown
        cat_breakdown = ""
        for cat, data in category_pcts.items():
            cat_breakdown += f"  - {cat}: {currency}{data['amount']:,.2f} ({data['percentage']}%)\n"

        # Format top expensive items
        top_items_str = ""
        for item in top_items[:5]:
            top_items_str += f"  - {item['name']}: {currency}{item['total']:,.2f}\n"

        # Format overspending alerts
        alerts_str = ""
        for alert in overspending:
            alerts_str += f"  - {alert['message']}\n"

        prompt = f"""You are an expert financial advisor and budget analyst. Analyze this grocery/shopping receipt data and provide personalized, actionable budgeting advice.

**IMPORTANT: All amounts are in {currency_name}. Use the currency symbol "{currency}" for all monetary values in your response.**

## Receipt Summary
- **Total Spent:** {currency}{grand_total:,.2f}
- **Total Items:** {total_items}
- **Average Item Price:** {currency}{basic_stats.get('avg_item_price', 0):,.2f}
- **Most Expensive Item:** {currency}{basic_stats.get('max_price', 0):,.2f}
- **Spending Health Score:** {health_score.get('score', 'N/A')}/100 ({health_score.get('label', 'N/A')})

## Spending by Category
{cat_breakdown}

## Top Expensive Items
{top_items_str}

## Overspending Alerts
{alerts_str if alerts_str else "  No major overspending detected."}

---

Based on this data, please provide:

1. **📊 Overall Assessment** (2-3 sentences about overall spending habits)
2. **💡 Smart Savings Tips** (3-4 specific, actionable tips to save money based on their actual purchases)
3. **🥗 Health & Nutrition Insight** (Comment on the nutritional balance of their shopping)
4. **📅 Monthly Budget Projection** (If they shop like this weekly, estimate monthly costs and suggest a realistic budget)
5. **🎯 Top 3 Action Items** (The most impactful changes they can make right now)

Keep the tone friendly, encouraging, and practical. Use specific {currency_name} amounts from the data with the "{currency}" symbol.
Format your response in clean markdown with the section headers above.
"""
        return prompt

    def get_advice(self, categorization_data: dict, analysis_data: dict, currency: str = "$") -> dict:
        """
        Get personalized financial advice from the LLM.

        Returns dict with 'advice' text and 'status'.
        """
        prompt = self._build_prompt(categorization_data, analysis_data, currency)

        if self.model is None:
            return {
                "advice": self._generate_fallback_advice(categorization_data, analysis_data, currency),
                "status": "fallback",
                "message": "Using built-in analysis (set Gemini API key for AI-powered advice)"
            }

        try:
            response = self.model.generate_content(prompt)
            advice_text = response.text

            return {
                "advice": advice_text,
                "status": "success",
                "message": "AI-powered advice generated successfully"
            }

        except Exception as e:
            return {
                "advice": self._generate_fallback_advice(categorization_data, analysis_data, currency),
                "status": "error",
                "message": f"LLM error: {str(e)}. Using built-in analysis."
            }

    def _generate_fallback_advice(self, categorization_data: dict,
                                   analysis_data: dict, currency: str = "$") -> str:
        """Generate advice without LLM as fallback."""
        grand_total = categorization_data.get("grand_total", 0)
        category_pcts = categorization_data.get("category_percentages", {})
        overspending = analysis_data.get("overspending_alerts", [])
        health_score = analysis_data.get("health_score", {})
        savings = analysis_data.get("savings_opportunities", [])
        total_items = categorization_data.get("total_items", 0)

        advice = "## 📊 Overall Assessment\n\n"
        advice += f"Your receipt totals **{currency}{grand_total:,.2f}** across **{total_items} items**. "
        advice += f"Your spending health score is **{health_score.get('score', 'N/A')}/100** "
        advice += f"({health_score.get('label', 'N/A')}). "

        if health_score.get("score", 0) >= 70:
            advice += "This shows generally healthy spending habits! "
        else:
            advice += "There's room for improvement in your spending patterns. "

        # Top category
        if category_pcts:
            top_cat = list(category_pcts.keys())[0]
            top_data = category_pcts[top_cat]
            advice += f"\n\nYour biggest spending category is **{top_cat}** at **{top_data['percentage']}%** of total spending.\n"

        # Smart Savings Tips
        advice += "\n## 💡 Smart Savings Tips\n\n"
        tips = [
            "🏷️ **Look for store brands** — they're typically 20-30% cheaper with the same quality.",
            "📋 **Make a shopping list** and stick to it to avoid impulse purchases.",
            "🔄 **Buy in bulk** for non-perishable items you use regularly.",
            "📱 **Use cashback apps** like Ibotta or store loyalty programs for extra savings.",
        ]
        for tip in tips:
            advice += f"- {tip}\n"

        # Overspending alerts
        if overspending:
            advice += "\n## ⚠️ Overspending Alerts\n\n"
            for alert in overspending[:3]:
                advice += f"- {alert['message']}\n"

        # Savings Opportunities
        if savings:
            advice += "\n## 💰 Savings Opportunities\n\n"
            for s in savings[:3]:
                advice += f"- {s['tip']}\n"

        # Monthly projection
        advice += "\n## 📅 Monthly Budget Projection\n\n"
        weekly = grand_total
        monthly = weekly * 4.3
        advice += f"If you shop like this weekly:\n"
        advice += f"- **Weekly:** {currency}{weekly:,.2f}\n"
        advice += f"- **Monthly:** ~{currency}{monthly:,.2f}\n"
        advice += f"- **Yearly:** ~{currency}{weekly * 52:,.2f}\n"
        advice += f"\n💡 **Suggested monthly budget:** {currency}{monthly * 0.85:,.2f} (15% reduction target)\n"

        # Action Items
        advice += "\n## 🎯 Top 3 Action Items\n\n"
        advice += "1. **Track every purchase** for the next month to identify spending patterns\n"
        advice += "2. **Set category budgets** based on your priorities and nutritional goals\n"
        advice += "3. **Compare prices** across stores for your top 5 most expensive items\n"

        return advice

    def is_available(self) -> bool:
        """Check if LLM is available."""
        return self.model is not None


