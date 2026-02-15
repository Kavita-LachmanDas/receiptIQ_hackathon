"""
Data Parser Module
Converts raw OCR text into structured receipt data with error handling and cleaning.
Handles multiple receipt formats:
  - US format: ITEM NAME  4.99  F
  - South Asian format: Item Name (line 1) + Qty  Price  Discount  RsTotal (line 2)
Uses footer boundary detection to avoid parsing tax/payment sections as items.
Handles OCR errors: comma-as-decimal, spaces in numbers, garbled Rs prefixes.
"""

import re
from typing import List, Dict, Optional, Tuple


class DataParser:
    """Parses raw OCR text into structured receipt items."""

    # Lines to skip entirely
    SKIP_PATTERNS = [
        r'(?i)^\s*(sub\s*total|subtotal)',
        r'(?i)^\s*(order\s*total)',
        r'(?i)^\s*(grand\s*total|net\s*total)',
        r'(?i)^\s*(total\s*items|tota.?\s*items)',
        r'(?i)^\s*(sales?\s*tax|tax\s*break|sale\s*tax|vat|gst|hst)',
        r'(?i)^\s*(change|cash|credit|debit|visa|master|pa[yv]ment)',
        r'(?i)^\s*(thank|welcome|receipt|rece.?pt|original|glad)',
        r'(?i)^\s*(date|time|store|tel|phone|address)',
        r'(?i)^\s*(server|cashier|register|terminal|manager)',
        r'(?i)^\s*(card|paid|balance|due)',
        r'(?i)^\s*(you\s*saved|savings|save)',
        r'(?i)^\s*(discount|die?\s*scount|coupon|promo)',
        r'(?i)^\s*(order|ref|#\s*\d)',
        r'(?i)(invoice\s*valu|bill\s*amount)',
        r'(?i)^\s*(product\s*descr|quantity\s*price|qty\s*price)',
        r'(?i)^\s*(sales\s*items|items\s*sold)',
        r'(?i)(round\s*[il1]ng|round\s*off|rounding)',
        r'(?i)(exl|excl|incl|inl)\.?\s*(amt|amount)',
        r'(?i)(amt|amount)\s*(gst|cst|tax|excl|incl|inl|exl)',
        r'(?i)^\s*(mrp|non\s*mrp)',
        r'(?i)^\s*(user|usei|pos\s*:|trn|trans)',
        r'(?i)(fbr|f.r)\s*(pos|p.s)',
        r'(?i)^\s*(pos\s*charge)',
        r'(?i)(sale\s*tax\s*break)',
        r'(?i)^\s*\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}',
        r'(?i)^\s*\d{1,2}:\d{2}',
        r'^\s*[\-=_\*\.]{3,}\s*$',
        r'^\s*[\-=_\*\.\s]*$',
        r'^\s*$',
        r'(?i)^\s*pub\s+\d',
        r'(?i)^\s*\d+\s*@\s*\d',
        r'(?i)^\s*\d+\.?\d*\s*lb\s*@',
        r'(?i)^\s*\d+\.?\d*\s*@\s',
        r'(?i)^\s*/\s*lb',
        r'(?i)^\s*\d+\s+(for|FOR)\s+\d',
        r'(?i)\d+\s*/\s*\d+[,\.]\d{2}\s*$',       # "5/5.00" or "5/5,00"
        r'(?i)^\s*[Rr][Ss]\.?\s*[\d,\.\s]+\s*$',   # Standalone Rs amounts
        r'(?i)^\s*Re[bds]',                          # OCR mangled "Rs8,548.00" → "Reb,5ab,00"
    ]

    # Footer markers — STOP parsing items after these
    FOOTER_MARKERS = [
        r'(?i)tota.?\s*items',
        r'(?i)invoice\s*valu',
        r'(?i)grand\s*total',
        r'(?i)sub\s*total',
        r'(?i)sale\s*tax\s*break',
        r'(?i)round\s*[il1]ng',
        r'(?i)die?\s*scount.*[Rr][Ss]',
        r'(?i)discount.*[Rr][Ss]',
        r'(?i)\d+\s*/\s*\d+[,\.]\d',    # "5/5.00" or "5/5,00" total items
    ]

    # Total extraction patterns
    TOTAL_PATTERNS = [
        (r'(?i)(invoice\s*valu).*?[Rr][Ss]\.?\s*([\d,\.\s]+)', 'total'),
        (r'(?i)(grand\s*total).*?[Rr][Ss]\.?\s*([\d,\.\s]+)', 'total'),
        (r'(?i)(sub\s*total|subtotal).*?[Rr][Ss]\.?\s*([\d,\.\s]+)', 'subtotal'),
        (r'(?i)(sub\s*total|subtotal).*?[\$]?\s*(\d+[.,]\d{2})', 'subtotal'),
        (r'(?i)(order\s*total).*?[\$]?\s*(\d+[.,]\d{2})', 'total'),
        (r'(?i)(grand\s*total).*?[\$]?\s*(\d+[.,]\d{2})', 'total'),
        (r'(?i)\b(total)\b.*?[\$]?\s*(\d+[.,]\d{2})', 'total'),
        (r'(?i)(sales?\s*tax|tax|vat).*?[\$]?\s*(\d+[.,]\d{2})', 'tax'),
    ]

    CURRENCY_PATTERNS = [
        r'[Rr][Ss]\.?\s*',
        r'[\$\£\€\₹\¥]',
        r'PKR\s*',
        r'INR\s*',
        r'USD\s*',
    ]

    def __init__(self):
        self.receipt_metadata = {}
        self.detected_format = None

    # ─── NUMBER / PRICE UTILITIES ──────────────────────────────────

    def fix_ocr_number(self, raw: str) -> Optional[float]:
        """
        Fix OCR-mangled numbers where periods are read as commas.
        Handles: '250,00' → 250.00, '2,499,00' → 2499.00, '2 499 , 00' → 2499.00
        Also handles normal: '250.00' → 250.00, '2,499.00' → 2499.00
        """
        if not raw:
            return None

        # Remove all spaces within the number
        s = re.sub(r'\s+', '', raw.strip())

        # Remove any non-numeric chars except . and ,
        s = re.sub(r'[^\d.,]', '', s)

        if not s:
            return None

        # Count commas and periods
        commas = s.count(',')
        periods = s.count('.')

        # CASE 1: Has a period near the end → standard decimal (250.00, 2499.00)
        # "250.00" or "2,499.00"
        period_match = re.match(r'^([\d,]+)\.(\d{1,2})$', s)
        if period_match:
            integer_part = period_match.group(1).replace(',', '')
            decimal_part = period_match.group(2)
            try:
                return float(f"{integer_part}.{decimal_part}")
            except ValueError:
                pass

        # CASE 2: Ends with comma + 2 digits → comma is decimal separator (OCR read . as ,)
        # "250,00" or "2,499,00" or "2499,00"
        comma_dec_match = re.match(r'^([\d,\.]*\d),(\d{2})$', s)
        if comma_dec_match:
            integer_part = comma_dec_match.group(1).replace(',', '').replace('.', '')
            decimal_part = comma_dec_match.group(2)
            try:
                return float(f"{integer_part}.{decimal_part}")
            except ValueError:
                pass

        # CASE 3: Pure integer (no decimal at all)
        clean = s.replace(',', '').replace('.', '')
        if clean.isdigit():
            try:
                return float(clean)
            except ValueError:
                pass

        # CASE 4: Try simple float conversion
        try:
            return float(s.replace(',', ''))
        except ValueError:
            pass

        return None

    def extract_all_numbers(self, line: str) -> List[float]:
        """
        Extract all numbers from a line, handling OCR comma-as-decimal.
        Splits on whitespace chunks, then tries to parse each token as a number.
        """
        # First, remove Rs prefixes so "Rs250,00" becomes "250,00"
        cleaned = re.sub(r'[Rr][Ss][a-z]?\s*', '', line)

        # Split into tokens by 2+ spaces or tabs (receipt columns)
        tokens = re.split(r'\s{2,}|\t', cleaned)

        numbers = []
        for token in tokens:
            token = token.strip()
            if not token:
                continue

            # Try to extract number from token
            # First remove any surrounding text characters
            num_str = re.sub(r'^[^0-9,\.]*', '', token)  # Remove leading non-numeric
            num_str = re.sub(r'[^0-9,\.]*$', '', num_str)  # Remove trailing non-numeric

            if num_str:
                val = self.fix_ocr_number(num_str)
                if val is not None and val >= 0:
                    numbers.append(val)

        return numbers

    def extract_rs_amount(self, line: str) -> Optional[float]:
        """
        Try to extract Rs-prefixed amount from a line.
        Handles: Rs250,00  Rs2,499,00  Rs2, 499 , 00
        Also handles garbled: Rsz,999,00 (z→2 not possible, skip)
        """
        # Find Rs followed by digits/commas/periods/spaces
        matches = re.findall(r'[Rr][Ss]\.?\s*([\d,\.\s]+)', line)
        valid_amounts = []
        for m in matches:
            val = self.fix_ocr_number(m)
            if val is not None and val > 0:
                valid_amounts.append(val)

        if valid_amounts:
            return valid_amounts[-1]  # Last Rs amount is usually the total
        return None

    # ─── LINE CLASSIFICATION ──────────────────────────────────────

    def clean_text(self, text: str) -> str:
        return re.sub(r'\s+', ' ', text).strip()

    def should_skip_line(self, line: str) -> bool:
        cleaned = line.strip()
        if not cleaned or len(cleaned) < 3:
            return True
        for pattern in self.SKIP_PATTERNS:
            if re.search(pattern, cleaned):
                return True
        return False

    def is_footer_marker(self, line: str) -> bool:
        for pattern in self.FOOTER_MARKERS:
            if re.search(pattern, line):
                return True
        return False

    def find_footer_start(self, lines: List[str]) -> int:
        for i, line in enumerate(lines):
            if self.is_footer_marker(line):
                return i
        return len(lines)

    def is_weight_line(self, line: str) -> bool:
        patterns = [
            r'^\s*\d+\.?\d*\s*lb\s*@',
            r'^\s*\d+\.?\d*\s*@\s*\d',
            r'^\s*\d+\s*@\s*\d',
            r'^\s*/\s*lb',
            r'^\s*\d+\s+for\s+\d',
        ]
        for p in patterns:
            if re.search(p, line.strip(), re.IGNORECASE):
                return True
        return False

    def is_detail_line(self, line: str) -> bool:
        """Check if line looks like a detail/qty line: 1.00  250.00  0.00  Rs250.00"""
        numbers = self.extract_all_numbers(line)
        if len(numbers) >= 2:
            first = numbers[0]
            if 0 < first <= 100:  # Reasonable quantity
                return True
        return False

    # ─── DETAIL LINE PARSING ──────────────────────────────────────

    def parse_detail_line(self, line: str) -> Optional[Tuple[float, int, float]]:
        """
        Parse detail line: Qty  UnitPrice  Discount  RsTotal
        Returns: (total_price, quantity, unit_price) or None
        
        Strategy: Use UNIT PRICE (2nd number) as primary since OCR reads
        it more reliably than Rs amounts with garbled prefixes.
        """
        numbers = self.extract_all_numbers(line)
        rs_total = self.extract_rs_amount(line)

        if len(numbers) < 2:
            return None

        # First number = quantity (should be small integer like 1, 2, 3)
        qty_val = numbers[0]
        qty = max(1, int(qty_val)) if 0 < qty_val <= 100 else 1

        # Second number = unit price (most reliably OCR'd number)
        unit_price = numbers[1]

        if unit_price <= 0:
            return None

        # Calculate expected total
        expected_total = qty * unit_price

        # Use Rs total if available and reasonable, otherwise calculate from unit price
        if rs_total and rs_total > 0:
            # Check if Rs total roughly matches expected
            if abs(rs_total - expected_total) < expected_total * 0.15:
                final_total = rs_total
            else:
                # Rs amount is garbled, trust unit_price × qty
                final_total = expected_total
        else:
            final_total = expected_total

        return (round(final_total, 2), qty, round(unit_price, 2))

    # ─── FORMAT DETECTION ─────────────────────────────────────────

    def detect_format(self, lines: List[str]) -> str:
        rs_count = sum(1 for line in lines if re.search(r'[Rr][Ss]\.?\s*[\d,]', line))
        dollar_count = sum(1 for line in lines if re.search(r'\$\s*\d+', line))
        if rs_count >= 2:
            return 'south_asian'
        elif dollar_count >= 2:
            return 'us'
        return 'generic'

    def extract_totals(self, lines: List[str]) -> dict:
        totals = {}
        for line in lines:
            for pattern, key in self.TOTAL_PATTERNS:
                match = re.search(pattern, line)
                if match:
                    val = self.fix_ocr_number(match.group(2))
                    if val and val > 0:
                        totals[key] = val
        return totals

    def extract_store_info(self, lines: List[str]) -> dict:
        info = {}
        for line in lines[:5]:
            cleaned = line.strip()
            if cleaned and len(cleaned) > 2:
                if not re.match(r'^[\-=_\*\.]+$', cleaned):
                    if not re.match(r'^\d', cleaned):
                        if not re.match(r'(?i)^(user|usei|pos|trn|original|product|quantity|sales)', cleaned):
                            info['store_name'] = cleaned
                            break
        for line in lines:
            date_match = re.search(r'(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4})', line)
            if date_match:
                info['date'] = date_match.group(1)
                break
        return info

    # ─── SOUTH ASIAN FORMAT PARSER ────────────────────────────────

    def parse_south_asian_format(self, lines: List[str]) -> List[dict]:
        """
        Parse South Asian receipt format with footer boundary detection.
        Item Name on line 1, Qty/Price/Discount/Total on line 2.
        Uses UNIT PRICE as primary price source (more reliable than garbled Rs amounts).
        """
        items = []
        footer_start = self.find_footer_start(lines)

        i = 0
        while i < footer_start:
            line = lines[i].strip()

            if not line or self.should_skip_line(line):
                i += 1
                continue

            has_text = bool(re.search(r'[A-Za-z]{2,}', line))
            has_rs = bool(re.search(r'[Rr][Ss]', line))
            is_detail = self.is_detail_line(line)

            # CASE 1: Item name line (has text, not a detail line with Rs)
            if has_text and not is_detail:
                item_name = line.strip()

                # Look for detail line on the next line
                if i + 1 < footer_start:
                    next_line = lines[i + 1].strip()

                    # Check if next line is a detail line
                    if self.is_detail_line(next_line):
                        detail = self.parse_detail_line(next_line)

                        if detail:
                            total_price, qty, unit_price = detail

                            # Clean item name - remove stray numbers and special chars
                            clean_name = re.sub(r'^\s*[\-\*\.]+\s*', '', item_name)
                            clean_name = re.sub(r'\s*[\-\*\.]+\s*$', '', clean_name)
                            clean_name = re.sub(r'\b\d+[,\.]\d+\b', '', clean_name)  # remove embedded prices
                            clean_name = ' '.join(clean_name.split())

                            if (clean_name and len(clean_name) >= 2
                                    and re.search(r'[A-Za-z]{2,}', clean_name)
                                    and not re.search(r'(?i)(fbr|f.r)\s*(pos|p.s)', clean_name)
                                    and not re.search(r'(?i)pos\s*charge', clean_name)):
                                items.append({
                                    "name": clean_name,
                                    "price": unit_price,
                                    "quantity": qty,
                                    "total": total_price,
                                    "raw_line": f"{line} | {next_line}"
                                })

                            i += 2
                            continue

            # CASE 2: Merged line — item name + numbers + Rs all on one line
            # e.g.: "MybelI  2999,00  ine Fit Me Cmpct Pwdr #120  Rs2,999,00"
            if has_text and has_rs:
                numbers = self.extract_all_numbers(line)
                rs_amount = self.extract_rs_amount(line)

                if numbers and len(numbers) >= 1:
                    # Find the unit price (largest number that's not the Rs total)
                    # Remove name text to find prices
                    best_price = rs_amount if rs_amount else max(numbers)

                    # Extract name by removing all numbers and Rs parts
                    name = re.sub(r'[Rr][Ss][a-z]?\s*[\d,\.\s]*', '', line)
                    name = re.sub(r'\b\d+[,\.]\d+\b', '', name)
                    name = re.sub(r'\b\d+\b', '', name)
                    name = re.sub(r'[#@_]', ' ', name)
                    name = re.sub(r'\s+', ' ', name).strip()
                    name = re.sub(r'^[\s\-\*\.\,]+', '', name)
                    name = re.sub(r'[\s\-\*\.\,]+$', '', name)

                    if (name and len(name) >= 2 and best_price and best_price > 0
                            and re.search(r'[A-Za-z]{2,}', name)
                            and not re.search(r'(?i)(fbr|f.r)\s*(pos|p.s)', name)):
                        items.append({
                            "name": name,
                            "price": round(best_price, 2),
                            "quantity": 1,
                            "total": round(best_price, 2),
                            "raw_line": line
                        })

            i += 1

        return items

    # ─── US FORMAT PARSER ─────────────────────────────────────────

    def parse_us_format(self, lines: List[str]) -> List[dict]:
        merged_lines = self.merge_multiline_items(lines)
        items = []
        for line in merged_lines:
            cleaned = self.clean_text(line)
            if not cleaned or len(cleaned) < 3:
                continue
            if self.should_skip_line(cleaned):
                continue
            if self.is_weight_line(cleaned):
                continue

            price = self.extract_price_from_line(cleaned)
            if price is None:
                continue

            name = self.extract_item_name(cleaned, price)
            if not name or len(name) < 2 or not re.search(r'[A-Za-z]{2,}', name):
                continue

            items.append({
                "name": name,
                "price": round(price, 2),
                "quantity": 1,
                "total": round(price, 2),
                "raw_line": line.strip()
            })
        return items

    def extract_price_from_line(self, line: str) -> Optional[float]:
        normalized = self.normalize_price_string(line)
        cleaned = re.sub(r'\s+[FTNBX]\s*$', '', normalized.strip())

        decimal_prices = re.findall(r'(\d+\.\d{2})\b', cleaned)
        if decimal_prices:
            try:
                price = float(decimal_prices[-1])
                if 0.01 <= price <= 999999.99:
                    return price
            except ValueError:
                pass
        return None

    def normalize_price_string(self, text: str) -> str:
        normalized = text
        for pattern in self.CURRENCY_PATTERNS:
            normalized = re.sub(pattern, '', normalized)
        normalized = re.sub(r'(\d),(\d)', r'\1\2', normalized)
        return normalized

    def extract_item_name(self, line: str, price: float) -> str:
        name = line.strip()
        name = re.sub(r'\s+[FTNBX]\s*$', '', name)
        name = re.sub(r'[Rr][Ss]\.?\s*[\d,]+\.?\d*', '', name)
        price_str = f"{price:.2f}"
        name = name.replace(price_str, '')
        if price == int(price):
            name = name.replace(str(int(price)), '', 1)
        name = re.sub(r'\s+\d+[\.,]?\d*\s*$', '', name)
        for pattern in self.CURRENCY_PATTERNS:
            name = re.sub(pattern, '', name)
        name = re.sub(r'\d+\.?\d*\s*lb\s*@\s*\d+\.?\d*/?\s*lb', '', name, flags=re.IGNORECASE)
        name = re.sub(r'\d+\s*@\s*', '', name)
        name = re.sub(r'\d+\s+FOR\s+\d+\.?\d*', '', name, flags=re.IGNORECASE)
        name = re.sub(r'^[\s\-\*\.\,\#\@\d]+', '', name)
        name = re.sub(r'[\s\-\*\.\,\#]+$', '', name)
        name = ' '.join(name.split())
        return name.strip()

    def merge_multiline_items(self, lines: List[str]) -> List[str]:
        merged = []
        i = 0
        while i < len(lines):
            current = lines[i].strip()
            has_price = bool(re.search(r'\d+\.\d{2}', current))
            is_text = bool(re.search(r'[A-Za-z]{2,}', current))
            if is_text and not has_price and i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                next_price = re.search(r'(\d+[\.,]?\d*\.?\d{0,2})\s*[FTNBX]?\s*$', next_line)
                if next_price:
                    merged_line = f"{current}  {next_price.group(1)}"
                    flag_match = re.search(r'\s+([FTNBX])\s*$', next_line)
                    if flag_match:
                        merged_line += f" {flag_match.group(1)}"
                    merged.append(merged_line)
                    i += 2
                    continue
            merged.append(current)
            i += 1
        return merged

    # ─── MAIN ENTRY POINT ─────────────────────────────────────────

    def parse_receipt(self, raw_text: str, lines: List[str] = None) -> dict:
        if lines is None:
            lines = raw_text.split('\n')

        store_info = self.extract_store_info(lines)
        totals = self.extract_totals(lines)
        self.detected_format = self.detect_format(lines)

        if self.detected_format == 'south_asian':
            items = self.parse_south_asian_format(lines)
        else:
            items = self.parse_us_format(lines)

        # Fallback: try the other format
        if not items and self.detected_format == 'south_asian':
            items = self.parse_us_format(lines)
        elif not items:
            items = self.parse_south_asian_format(lines)

        calculated_total = sum(item["total"] for item in items)
        if "total" not in totals:
            totals["total"] = round(calculated_total, 2)
        if "subtotal" not in totals:
            totals["subtotal"] = round(calculated_total, 2)

        currency = "Rs" if self.detected_format == 'south_asian' else "$"
        for line in lines:
            if re.search(r'[Rr][Ss]\.?\s*\d', line):
                currency = "Rs"
                break
            if re.search(r'\$\s*\d', line):
                currency = "$"
                break

        return {
            "items": items,
            "totals": totals,
            "store_info": store_info,
            "raw_text": raw_text,
            "item_count": len(items),
            "calculated_total": round(calculated_total, 2),
            "detected_format": self.detected_format,
            "currency": currency,
        }
