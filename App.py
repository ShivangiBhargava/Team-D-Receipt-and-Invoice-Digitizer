import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px
from PIL import Image, ImageOps
import pytesseract
from datetime import datetime, date
import json
from groq import Groq
import io
from pdf2image import convert_from_bytes
import re
from contextlib import contextmanager
import hashlib
import os
from functools import lru_cache
import locale
from deep_translator import GoogleTranslator
from googletrans import Translator
import ast
import inspect
from functools import wraps

# ============================================================================
# MULTI-LANGUAGE TRANSLATION MODULE
# ============================================================================

class TranslationManager:
    """Centralized translation management with AI fallback"""
    
    SUPPORTED_LANGUAGES = {
        'en': 'English',
        'hi': 'हिन्दी (Hindi)',
        'fr': 'Français (French)',
        'es': 'Español (Spanish)',
        'de': 'Deutsch (German)',
        'zh': '中文 (Chinese)',
        'ja': '日本語 (Japanese)',
        'ru': 'Русский (Russian)',
        'ar': 'العربية (Arabic)',
        'pt': 'Português (Portuguese)',
        'it': 'Italiano (Italian)',
        'nl': 'Nederlands (Dutch)',
        'ko': '한국어 (Korean)',
        'tr': 'Türkçe (Turkish)',
        'pl': 'Polski (Polish)',
        'uk': 'Українська (Ukrainian)',
        'ta': 'தமிழ் (Tamil)',
        'te': 'తెలుగు (Telugu)',
        'bn': 'বাংলা (Bengali)',
        'mr': 'मराठी (Marathi)',
        'gu': 'ગુજરાતી (Gujarati)',
        'kn': 'ಕನ್ನಡ (Kannada)',
        'ml': 'മലയാളം (Malayalam)',
        'pa': 'ਪੰਜਾਬੀ (Punjabi)',
        'ur': 'اردو (Urdu)'
    }
    
    INDIAN_LANGUAGES = {
        'hi': 'हिन्दी',
        'bn': 'বাংলা',
        'te': 'తెలుగు',
        'mr': 'मराठी',
        'ta': 'தமிழ்',
        'ur': 'اردو',
        'gu': 'ગુજરાતી',
        'kn': 'ಕನ್ನಡ',
        'ml': 'മലയാളം',
        'pa': 'ਪੰਜਾਬੀ'
    }
    
    def __init__(self):
        self.cache_file = 'translation_cache.json'
        self.load_cache()
        self.translator = Translator()
        self.google_translator = GoogleTranslator()
        
    def load_cache(self):
        """Load translation cache from file"""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    self.cache = json.load(f)
            except:
                self.cache = {}
        else:
            self.cache = {}
    
    def save_cache(self):
        """Save translation cache to file"""
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)
        except:
            pass
    
    @lru_cache(maxsize=1000)
    def translate_text(self, text, target_lang='hi', source_lang='auto'):
        """Translate text with caching and multiple fallback options"""
        if not text or target_lang == 'en' or len(text.strip()) == 0:
            return text
        
        # Create cache key
        cache_key = f"{text}_{target_lang}_{source_lang}"
        cache_hash = hashlib.md5(cache_key.encode()).hexdigest()
        
        # Check cache
        if cache_hash in self.cache:
            return self.cache[cache_hash]
        
        try:
            # Try primary translator (Google Translate)
            translated = self.google_translator.translate(text, target=target_lang, source=source_lang)
        except:
            try:
                # Fallback to secondary translator
                translated = self.translator.translate(text, dest=target_lang, src=source_lang).text
            except:
                # If all fails, return original
                translated = text
        
        # Cache the result
        self.cache[cache_hash] = translated
        self.save_cache()
        
        return translated
    
    def detect_language(self, text):
        """Detect language of given text"""
        try:
            return self.translator.detect(text).lang
        except:
            return 'en'

# Initialize translation manager
translation_manager = TranslationManager()

# Language selection session state
if 'current_language' not in st.session_state:
    st.session_state['current_language'] = 'en'
if 'rtl_mode' not in st.session_state:
    st.session_state['rtl_mode'] = False

# ============================================================================
# TRANSLATION HELPER FUNCTIONS
# ============================================================================

def t(text, **kwargs):
    """
    Main translation function - translates text to current language
    Usage: t('Hello {name}', name='John')
    """
    if text is None:
        return ""
    
    current_lang = st.session_state.get('current_language', 'en')
    
    # Get translation
    translated = translation_manager.translate_text(str(text), target_lang=current_lang)
    
    # Format with kwargs if provided
    if kwargs and translated:
        try:
            translated = translated.format(**kwargs)
        except:
            pass
    
    return translated

def translate_ui(func):
    """Decorator to automatically translate UI elements"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

def language_selector():
    """Enhanced language selector with regional grouping"""
    
    st.markdown("### " + t("Select Language"))
    
    # Language categories
    tab1, tab2, tab3 = st.tabs([t("🌍 International"), t("🇮🇳 Indian Languages"), t("⚙️ Settings")])
    
    with tab1:
        intl_langs = {k: v for k, v in TranslationManager.SUPPORTED_LANGUAGES.items() 
                     if k not in TranslationManager.INDIAN_LANGUAGES and k != 'ur'}
        
        # Create grid of language buttons
        cols = st.columns(3)
        for i, (code, name) in enumerate(intl_langs.items()):
            with cols[i % 3]:
                if st.button(f"🌐 {name}", key=f"lang_intl_{code}", use_container_width=True):
                    st.session_state['current_language'] = code
                    st.session_state['rtl_mode'] = (code in ['ar'])
                    st.rerun()
    
    with tab2:
        cols = st.columns(3)
        for i, (code, name) in enumerate(TranslationManager.INDIAN_LANGUAGES.items()):
            with cols[i % 3]:
                if st.button(f"🪔 {name}", key=f"lang_ind_{code}", use_container_width=True):
                    st.session_state['current_language'] = code
                    st.session_state['rtl_mode'] = (code == 'ur')
                    st.rerun()
    
    with tab3:
        # Auto-detect language
        if st.button(t("🔍 Auto-detect Language"), use_container_width=True):
            try:
                # Try to detect from sample text
                detected = translation_manager.detect_language("Sample text for detection")
                st.session_state['current_language'] = detected
                st.rerun()
            except:
                st.error(t("Could not detect language"))
        
        # Current language indicator
        current = TranslationManager.SUPPORTED_LANGUAGES.get(
            st.session_state['current_language'],
            'English'
        )
        st.info(f"**{t('Current Language')}:** {current}")
        
        # RTL toggle for Arabic/Urdu
        if st.session_state['rtl_mode']:
            if st.button(t("📝 Toggle RTL Mode"), use_container_width=True):
                st.session_state['rtl_mode'] = not st.session_state['rtl_mode']
                st.rerun()

def apply_rtl_styles():
    """Apply RTL CSS for Arabic/Hebrew/etc"""
    if st.session_state.get('rtl_mode', False):
        st.markdown("""
        <style>
        .stApp {
            direction: rtl;
            text-align: right;
        }
        .css-1d391kg {
            direction: rtl;
        }
        .stTextInput input {
            text-align: right;
        }
        .stSelectbox div[data-baseweb="select"] {
            direction: rtl;
        }
        </style>
        """, unsafe_allow_html=True)

# ============================================================================
# DYNAMIC CONTENT TRANSLATOR
# ============================================================================

class DynamicTranslator:
    """Handles translation of dynamic content (receipts, items, etc.)"""
    
    def translate_receipt(self, receipt_data):
        """Translate all text fields in a receipt"""
        if not receipt_data:
            return receipt_data
            
        translated = receipt_data.copy() if isinstance(receipt_data, dict) else {}
        target_lang = st.session_state.get('current_language', 'en')
        
        if target_lang == 'en':
            return translated
        
        text_fields = ['merchant', 'category', 'invoice_number']
        for field in text_fields:
            if field in translated and translated[field]:
                translated[field] = translation_manager.translate_text(
                    str(translated[field]),
                    target_lang=target_lang
                )
        
        return translated
    
    def translate_line_items(self, items):
        """Translate line item names"""
        if not items:
            return items
            
        target_lang = st.session_state.get('current_language', 'en')
        if target_lang == 'en':
            return items
        
        translated_items = []
        for item in items:
            translated_item = item.copy() if isinstance(item, dict) else {}
            if 'name' in translated_item and translated_item['name']:
                translated_item['name'] = translation_manager.translate_text(
                    str(item['name']),
                    target_lang=target_lang
                )
            translated_items.append(translated_item)
        return translated_items
    
    def translate_dataframe(self, df, columns_to_translate=None):
        """Translate specific columns in a dataframe"""
        if df is None or df.empty:
            return df
            
        if columns_to_translate is None:
            columns_to_translate = ['merchant', 'category', 'name']
        
        target_lang = st.session_state.get('current_language', 'en')
        if target_lang == 'en':
            return df
        
        translated_df = df.copy()
        
        for col in columns_to_translate:
            if col in translated_df.columns:
                translated_df[col] = translated_df[col].apply(
                    lambda x: translation_manager.translate_text(str(x), target_lang) 
                    if pd.notna(x) else x
                )
        
        return translated_df

# Initialize dynamic translator
dynamic_translator = DynamicTranslator()

# ============================================================================
# LANGUAGE-SPECIFIC FORMATTERS
# ============================================================================

class LanguageFormatter:
    """Handles language-specific formatting (dates, numbers, currency)"""
    
    LOCALE_MAP = {
        'en': 'en_US.UTF-8',
        'hi': 'hi_IN.UTF-8',
        'fr': 'fr_FR.UTF-8',
        'es': 'es_ES.UTF-8',
        'de': 'de_DE.UTF-8',
        'zh': 'zh_CN.UTF-8',
        'ja': 'ja_JP.UTF-8',
        'ar': 'ar_AE.UTF-8',
        'ta': 'ta_IN.UTF-8',
        'te': 'te_IN.UTF-8',
        'bn': 'bn_IN.UTF-8',
        'ur': 'ur_PK.UTF-8'
    }
    
    @staticmethod
    def format_date(date_obj, lang=None):
        """Format date according to language preferences"""
        if lang is None:
            lang = st.session_state.get('current_language', 'en')
            
        if isinstance(date_obj, str):
            try:
                date_obj = datetime.strptime(date_obj, '%Y-%m-%d')
            except:
                return date_obj
        
        if not isinstance(date_obj, datetime):
            return str(date_obj)
        
        # Different date formats by language
        formats = {
            'en': '%B %d, %Y',
            'hi': '%d %B %Y',
            'fr': '%d %B %Y',
            'es': '%d de %B de %Y',
            'de': '%d. %B %Y',
            'zh': '%Y年%m月%d日',
            'ja': '%Y年%m月%d日',
            'ar': '%d %B %Y',
            'ta': '%d %B %Y',
            'te': '%d %B %Y',
            'bn': '%d %B %Y',
            'ur': '%d %B %Y'
        }
        
        # Get month names in target language if possible
        try:
            locale.setlocale(locale.LC_TIME, LanguageFormatter.LOCALE_MAP.get(lang, 'en_US.UTF-8'))
        except:
            pass
        
        return date_obj.strftime(formats.get(lang, '%Y-%m-%d'))
    
    @staticmethod
    def format_currency(amount, lang=None, symbol=True):
        """Format currency with proper symbols"""
        if lang is None:
            lang = st.session_state.get('current_language', 'en')
            
        try:
            amount = float(amount)
        except:
            return str(amount)
        
        currency_symbols = {
            'en': '$', 'hi': '₹', 'fr': '€', 'es': '€', 'de': '€',
            'zh': '¥', 'ja': '¥', 'ar': 'د.م.', 'ta': '₹', 'te': '₹',
            'bn': '₹', 'mr': '₹', 'gu': '₹', 'kn': '₹', 'ml': '₹',
            'pa': '₹', 'ur': '₹'
        }
        
        # Format number with proper thousands separators
        if lang in ['hi', 'ta', 'te', 'bn', 'mr', 'gu', 'kn', 'ml', 'pa', 'ur']:
            # Indian numbering system (lakhs, crores)
            amount_str = LanguageFormatter._format_indian_number(amount)
        else:
            # Western numbering system
            amount_str = f"{amount:,.2f}"
        
        if symbol:
            symbol = currency_symbols.get(lang, '$')
            # Symbol position varies by language
            if lang in ['fr', 'es', 'de', 'ar', 'ur']:
                return f"{amount_str} {symbol}"
            else:
                return f"{symbol}{amount_str}"
        
        return amount_str
    
    @staticmethod
    def _format_indian_number(amount):
        """Format numbers in Indian style (lakhs, crores)"""
        amount_str = f"{amount:.2f}"
        parts = amount_str.split('.')
        integer_part = parts[0]
        decimal_part = parts[1] if len(parts) > 1 else '00'
        
        # Format integer part in Indian style
        if len(integer_part) > 3:
            last_three = integer_part[-3:]
            other_numbers = integer_part[:-3]
            if other_numbers:
                # Group by twos from the right
                other_grouped = []
                for i in range(len(other_numbers), 0, -2):
                    start = max(0, i-2)
                    other_grouped.append(other_numbers[start:i])
                other_grouped.reverse()
                formatted_int = ','.join(other_grouped) + ',' + last_three
            else:
                formatted_int = last_three
        else:
            formatted_int = integer_part
        
        return f"{formatted_int}.{decimal_part}"

# ============================================================================
# AI-POWERED TRANSLATION ENHANCER
# ============================================================================

class AITranslationEnhancer:
    """Uses Groq AI for context-aware translations"""
    
    def __init__(self, api_key):
        self.client = Groq(api_key=api_key) if api_key else None
    
    def enhance_translation(self, text, context, target_lang):
        """Use AI for better context-aware translation"""
        if not self.client or not text:
            return translation_manager.translate_text(text, target_lang)
        
        prompt = f"""
        Translate the following text to {target_lang} with proper context.
        
        Context: This is {context}
        Original text: {text}
        
        Provide only the translated text, no explanations or additional text.
        """
        
        try:
            response = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="mixtral-8x7b-32768",
                temperature=0.3
            )
            return response.choices[0].message.content.strip()
        except:
            return translation_manager.translate_text(text, target_lang)

# ============================================================================
# CONFIGURATION
# ============================================================================

DB_NAME = 'receipt_vault_v6.db'
GROQ_MODEL = "llama-3.3-70b-versatile"

# Template definitions
VENDOR_TEMPLATES = {
    "coffee house": {
        "standardized_name": "Coffee House",
        "tax_rate": 0.0825,
        "aliases": ["coffee house inc", "the coffee house", "coffee house llc"]
    },
    "walmart": {
        "standardized_name": "Walmart",
        "tax_rate": None,
        "aliases": ["walmart supercenter", "wm supercenter"]
    },
    "coffee house inc.": {
         "standardized_name": "Coffee House",
         "tax_rate": 0.0825,
         "aliases": []
    }
}

# ============================================================================
# DATABASE FUNCTIONS
# ============================================================================

@contextmanager
def get_db_connection():
    """Context manager for database connections"""
    conn = sqlite3.connect(DB_NAME)
    try:
        yield conn
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()

def init_db():
    """Initialize database with optimized indexes"""
    with get_db_connection() as conn:
        c = conn.cursor()

        # Create tables
        c.execute('''CREATE TABLE IF NOT EXISTS receipts (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        merchant TEXT,
                        date TEXT,
                        invoice_number TEXT,
                        subtotal REAL,
                        tax REAL,
                        total_amount REAL,
                        filename TEXT,
                        upload_timestamp TEXT,
                        category TEXT DEFAULT 'Uncategorized'
                    )''')

        try:
            c.execute("ALTER TABLE receipts ADD COLUMN category TEXT DEFAULT 'Uncategorized'")
        except sqlite3.OperationalError:
            pass

        c.execute('''CREATE TABLE IF NOT EXISTS line_items (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        receipt_id INTEGER,
                        name TEXT,
                        qty INTEGER,
                        price REAL,
                        FOREIGN KEY (receipt_id) REFERENCES receipts (id)
                    )''')

        c.execute('''CREATE TABLE IF NOT EXISTS monthly_budgets (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        month_year TEXT UNIQUE,
                        budget_amount REAL
                    )''')

        # Create indexes
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_receipts_date ON receipts(date)",
            "CREATE INDEX IF NOT EXISTS idx_receipts_merchant ON receipts(merchant)",
            "CREATE INDEX IF NOT EXISTS idx_receipts_category ON receipts(category)",
            "CREATE INDEX IF NOT EXISTS idx_line_items_receipt_id ON line_items(receipt_id)"
        ]

        for idx in indexes:
            c.execute(idx)

def check_if_receipt_exists(merchant, date, total, invoice_num):
    """Optimized duplicate detection"""
    with get_db_connection() as conn:
        c = conn.cursor()

        query = """
            SELECT id, merchant, total_amount, invoice_number
            FROM receipts
            WHERE date = ?
            AND ABS(total_amount - ?) <= 0.05
        """
        params = [date, total]

        if invoice_num and invoice_num != "Unknown":
            query += " AND invoice_number = ?"
            params.append(invoice_num)

        c.execute(query, params)
        candidates = c.fetchall()

        for row in candidates:
            db_id, db_merch, db_total, db_inv = row

            if invoice_num and invoice_num != "Unknown" and db_inv and db_inv != "Unknown":
                if invoice_num == db_inv:
                    return True, 1, db_id

            m1 = merchant.lower().strip()
            m2 = db_merch.lower().strip()
            merchant_match = (m1 in m2 or m2 in m1)

            if merchant_match:
                return True, 1, db_id

        return False, 0, None

def save_receipt_to_db(data, filename, line_items_data):
    """Save receipt with line items"""
    with get_db_connection() as conn:
        c = conn.cursor()
        upload_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Insert receipt
        c.execute("""INSERT INTO receipts
                     (merchant, date, invoice_number, subtotal, tax, total_amount, filename, upload_timestamp, category)
                     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                  (data['merchant'], data['date'], data['invoice_number'],
                   data['subtotal'], data['tax'], data['total'], filename, upload_time,
                   data.get('category', 'Uncategorized')))
        receipt_id = c.lastrowid

        # Insert line items
        if line_items_data:
            line_items_values = [
                (receipt_id, item['name'], item['qty'], item['price'])
                for item in line_items_data
            ]
            c.executemany(
                "INSERT INTO line_items (receipt_id, name, qty, price) VALUES (?, ?, ?, ?)",
                line_items_values
            )

        return receipt_id

@st.cache_data(ttl=300)
def get_all_receipts():
    """Cached retrieval of all receipts"""
    with get_db_connection() as conn:
        try:
            df = pd.read_sql_query("SELECT * FROM receipts ORDER BY date DESC, id DESC", conn)
            if 'category' not in df.columns:
                df['category'] = 'Uncategorized'
        except:
            df = pd.DataFrame()
    return df

def get_receipt_by_id(receipt_id):
    """Get single receipt by ID"""
    with get_db_connection() as conn:
        try:
            df = pd.read_sql_query("SELECT * FROM receipts WHERE id = ?", conn, params=(receipt_id,))
            if not df.empty and 'category' not in df.columns:
                df['category'] = 'Uncategorized'
        except:
            df = pd.DataFrame()
    return df

@st.cache_data(ttl=60)
def get_line_items(receipt_id):
    """Cached retrieval of line items"""
    with get_db_connection() as conn:
        try:
            df = pd.read_sql_query(
                "SELECT name, qty, price FROM line_items WHERE receipt_id = ?",
                conn,
                params=(receipt_id,)
            )
        except:
            df = pd.DataFrame()
    return df

@st.cache_data(ttl=60)
def get_all_line_items_global():
    """Cached retrieval of all line items"""
    with get_db_connection() as conn:
        try:
            query = """
                SELECT li.name, li.qty, li.price, r.merchant, r.date, r.invoice_number, r.id as receipt_id
                FROM line_items li
                JOIN receipts r ON li.receipt_id = r.id
            """
            df = pd.read_sql_query(query, conn)
        except:
            df = pd.DataFrame()
    return df

def delete_receipt(receipt_id):
    """Delete receipt and clear cache"""
    with get_db_connection() as conn:
        c = conn.cursor()
        c.execute("DELETE FROM line_items WHERE receipt_id = ?", (receipt_id,))
        c.execute("DELETE FROM receipts WHERE id = ?", (receipt_id,))

    # Clear caches
    st.cache_data.clear()

def clear_database():
    """Clear all database tables"""
    with get_db_connection() as conn:
        c = conn.cursor()
        c.execute("DELETE FROM line_items")
        c.execute("DELETE FROM receipts")
        c.execute("DELETE FROM monthly_budgets")

    st.cache_data.clear()

def get_monthly_budget(month_year):
    """Get budget for a specific month"""
    with get_db_connection() as conn:
        c = conn.cursor()
        c.execute("SELECT budget_amount FROM monthly_budgets WHERE month_year = ?", (month_year,))
        result = c.fetchone()
    return result[0] if result else 0.0

def set_monthly_budget(month_year, amount):
    """Set or update monthly budget"""
    with get_db_connection() as conn:
        c = conn.cursor()
        c.execute("""INSERT INTO monthly_budgets (month_year, budget_amount)
                     VALUES (?, ?)
                     ON CONFLICT(month_year)
                     DO UPDATE SET budget_amount = ?""", (month_year, amount, amount))

def get_filtered_receipts(keyword=None, date_filter=None, month_filter=None, receipt_id=None):
    """Filter receipts with optimized queries"""
    with get_db_connection() as conn:
        base_query = """
            SELECT DISTINCT r.*
            FROM receipts r
            LEFT JOIN line_items li ON r.id = li.receipt_id
            WHERE 1=1
        """
        params = []

        if keyword:
            base_query += """
                AND (LOWER(r.merchant) LIKE ?
                OR LOWER(r.invoice_number) LIKE ?
                OR LOWER(COALESCE(r.category, '')) LIKE ?
                OR LOWER(li.name) LIKE ?)
            """
            keyword_pattern = f"%{keyword.lower()}%"
            params.extend([keyword_pattern] * 4)

        if receipt_id and receipt_id > 0:
            base_query += " AND r.id = ?"
            params.append(receipt_id)

        if date_filter:
            base_query += " AND r.date = ?"
            params.append(date_filter)

        if month_filter and month_filter != "All Months":
            base_query += " AND substr(r.upload_timestamp, 1, 7) = ?"
            params.append(month_filter)

        base_query += " ORDER BY r.date DESC, r.id DESC"

        try:
            df = pd.read_sql_query(base_query, conn, params=params)
            if not df.empty and 'category' not in df.columns:
                df['category'] = 'Uncategorized'
        except:
            df = pd.DataFrame()

    return df

@st.cache_data(ttl=60)
def get_available_months():
    """Get available months from receipts"""
    with get_db_connection() as conn:
        try:
            query = """
                SELECT DISTINCT strftime('%Y-%m', upload_timestamp) as month_year
                FROM receipts
                WHERE upload_timestamp IS NOT NULL
                ORDER BY month_year DESC
            """
            df = pd.read_sql_query(query, conn)
            months = df['month_year'].tolist()
        except:
            months = []
    return months

@st.cache_data(ttl=300)
def get_analytics_summary():
    """Analytics summary for dashboard"""
    with get_db_connection() as conn:
        query = """
            SELECT
                COUNT(*) as total_receipts,
                SUM(total_amount) as total_spend,
                AVG(total_amount) as avg_ticket,
                SUM(tax) as total_tax,
                MAX(date) as latest_date
            FROM receipts
        """
        result = pd.read_sql_query(query, conn)
    return result.iloc[0] if not result.empty else None

@st.cache_data(ttl=300)
def get_monthly_spending_summary():
    """Monthly spending by category"""
    with get_db_connection() as conn:
        query = """
            SELECT
                strftime('%Y-%m', date) as month_year,
                category,
                SUM(total_amount) as total_spend,
                COUNT(*) as receipt_count
            FROM receipts
            WHERE date IS NOT NULL
            GROUP BY month_year, category
            ORDER BY month_year DESC, total_spend DESC
        """
        df = pd.read_sql_query(query, conn)
    return df

@st.cache_data(ttl=300)
def get_time_series_data():
    """Time series data for trends"""
    with get_db_connection() as conn:
        query = """
            SELECT
                strftime('%Y-%m', date) as month_year,
                SUM(total_amount) as monthly_total,
                COUNT(*) as receipt_count,
                AVG(total_amount) as avg_receipt
            FROM receipts
            WHERE date IS NOT NULL
            GROUP BY month_year
            ORDER BY month_year ASC
        """
        df = pd.read_sql_query(query, conn)
    return df

@st.cache_data(ttl=300)
def get_day_of_week_spending():
    """Spending by day of week"""
    with get_db_connection() as conn:
        query = """
            SELECT
                CASE CAST(strftime('%w', date) AS INTEGER)
                    WHEN 0 THEN 'Sunday'
                    WHEN 1 THEN 'Monday'
                    WHEN 2 THEN 'Tuesday'
                    WHEN 3 THEN 'Wednesday'
                    WHEN 4 THEN 'Thursday'
                    WHEN 5 THEN 'Friday'
                    WHEN 6 THEN 'Saturday'
                END as day_name,
                SUM(total_amount) as total_spend,
                COUNT(*) as receipt_count
            FROM receipts
            WHERE date IS NOT NULL
            GROUP BY strftime('%w', date)
        """
        df = pd.read_sql_query(query, conn)
    return df

@st.cache_data(ttl=300)
def get_top_items(limit=10):
    """Top purchased items"""
    with get_db_connection() as conn:
        query = """
            SELECT
                name as item_name,
                COUNT(*) as purchase_count,
                SUM(qty * price) as total_value
            FROM line_items
            GROUP BY name
            ORDER BY purchase_count DESC
            LIMIT ?
        """
        df = pd.read_sql_query(query, conn, params=(limit,))
    return df

@st.cache_data(ttl=300)
def get_merchant_breakdown():
    """Merchant spending breakdown"""
    with get_db_connection() as conn:
        query = """
            SELECT
                merchant,
                SUM(total_amount) as total_spend,
                COUNT(*) as receipt_count,
                AVG(total_amount) as avg_receipt
            FROM receipts
            GROUP BY merchant
            ORDER BY total_spend DESC
        """
        df = pd.read_sql_query(query, conn)
    return df

def advanced_search_receipts(filters):
    """Advanced search with multiple criteria"""
    with get_db_connection() as conn:
        base_query = "SELECT DISTINCT r.* FROM receipts r"
        conditions = []
        params = []

        if filters.get('keyword'):
            keyword_pattern = f"%{filters['keyword'].lower()}%"
            base_query += " LEFT JOIN line_items li ON r.id = li.receipt_id"
            conditions.append("""(LOWER(r.merchant) LIKE ?
                                 OR LOWER(r.invoice_number) LIKE ?
                                 OR LOWER(COALESCE(r.category, '')) LIKE ?
                                 OR LOWER(li.name) LIKE ?) """)
            params.extend([keyword_pattern] * 4)

        if filters.get('date_from'):
            conditions.append("r.date >= ?")
            params.append(filters['date_from'])

        if filters.get('date_to'):
            conditions.append("r.date <= ?")
            params.append(filters['date_to'])

        if filters.get('amount_min') is not None:
            conditions.append("r.total_amount >= ?")
            params.append(filters['amount_min'])

        if filters.get('amount_max') is not None:
            conditions.append("r.total_amount <= ?")
            params.append(filters['amount_max'])

        if filters.get('category') and filters['category'] != 'All':
            conditions.append("r.category = ?")
            params.append(filters['category'])

        if filters.get('merchant') and filters['merchant'] != 'All':
            conditions.append("r.merchant = ?")
            params.append(filters['merchant'])

        if conditions:
            base_query += " WHERE " + " AND ".join(conditions)

        base_query += " ORDER BY r.date DESC, r.id DESC"

        try:
            df = pd.read_sql_query(base_query, conn, params=params)
            if not df.empty and 'category' not in df.columns:
                df['category'] = 'Uncategorized'
        except Exception as e:
            st.error(t(f"Search error: {e}"))
            df = pd.DataFrame()

    return df

@st.cache_data(ttl=60)
def get_unique_categories():
    """Get unique categories"""
    with get_db_connection() as conn:
        try:
            query = "SELECT DISTINCT category FROM receipts WHERE category IS NOT NULL ORDER BY category"
            df = pd.read_sql_query(query, conn)
            categories = df['category'].tolist()
        except:
            categories = []
    return ['All'] + categories

@st.cache_data(ttl=60)
def get_unique_merchants():
    """Get unique merchants"""
    with get_db_connection() as conn:
        try:
            query = "SELECT DISTINCT merchant FROM receipts WHERE merchant IS NOT NULL ORDER BY merchant"
            df = pd.read_sql_query(query, conn)
            merchants = df['merchant'].tolist()
        except:
            merchants = []
    return ['All'] + merchants

# ============================================================================
# PROCESSING FUNCTIONS
# ============================================================================

def preprocess_image(image):
    return ImageOps.grayscale(image)

def extract_text(image):
    return pytesseract.image_to_string(image)

def convert_pdf_to_images(pdf_file):
    return convert_from_bytes(pdf_file.getvalue())

def parse_with_groq(raw_text, api_key):
    client = Groq(api_key=api_key)
    prompt = f"""
    Extract structured data from this receipt text.
    Return ONLY a JSON object with these keys:
    'merchant', 'date' (YYYY-MM-DD), 'invoice_number', 'subtotal', 'tax', 'total',
    'category' (e.g., Groceries, Dining, Shopping, Transportation, Utilities, Healthcare, Entertainment, Other),
    and 'line_items' (a list of objects with 'name', 'qty', 'price').

    Do NOT infer tax if it is not explicitly stated. Set tax to 0 if not found.
    If 'subtotal' is missing but 'total' and 'tax' exist, calculate it.

    Text:
    {raw_text}
    """
    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=GROQ_MODEL,
            response_format={"type": "json_object"}
        )
        return json.loads(chat_completion.choices[0].message.content)
    except Exception as e:
        st.error(t(f"Groq Parsing Error: {e}"))
        return None

def calculate_parsing_score(data):
    """Calculate accuracy score based on completeness"""
    score = 0
    if data.get('merchant') and data['merchant'] != 'Unknown': score += 25
    if data.get('date'): score += 25
    if data.get('total') and data['total'] > 0: score += 25
    if data.get('tax') and data['tax'] > 0: score += 15
    if data.get('invoice_number') and data['invoice_number'] != 'Unknown': score += 10
    return min(max(score, 0), 100)

def find_matching_template(merchant_name):
    """Find template based on merchant name"""
    clean_name = merchant_name.lower().strip()

    if clean_name in VENDOR_TEMPLATES:
        return VENDOR_TEMPLATES[clean_name]

    for vendor_key, template_data in VENDOR_TEMPLATES.items():
        if clean_name == vendor_key or clean_name in template_data['aliases']:
            return template_data

    for vendor_key, template_data in VENDOR_TEMPLATES.items():
        if vendor_key in clean_name:
             return template_data

    return None

def apply_template_parsing(standard_data):
    """Apply vendor templates to parsed data"""
    refined_data = standard_data.copy()
    merchant = refined_data.get('merchant', 'Unknown')

    template = find_matching_template(merchant)
    template_applied = False

    if template:
        template_applied = True
        refined_data['merchant'] = template['standardized_name']

        total = refined_data.get('total', 0)
        tax = refined_data.get('tax', 0)
        known_rate = template.get('tax_rate')

        if total > 0 and (tax == 0 or tax is None) and known_rate is not None:
            subtotal = total / (1 + known_rate)
            calculated_tax = total - subtotal

            refined_data['tax'] = round(calculated_tax, 2)
            refined_data['subtotal'] = round(subtotal, 2)
            refined_data['tax_rate_applied'] = known_rate

    return refined_data, template_applied

def get_ai_budget_suggestions(total_spend, budget, category_breakdown, api_key):
    client = Groq(api_key=api_key)
    categories_text = "\n".join([f"- {cat}: ${amt:.2f}" for cat, amt in category_breakdown.items()])
    prompt = f"""
    You are a financial advisor. A user has exceeded their monthly budget.

    Monthly Budget: ${budget:.2f}
    Actual Spending: ${total_spend:.2f}
    Overspend: ${total_spend - budget:.2f}

    Category Breakdown:
    {categories_text}

    Provide 3-5 specific, actionable recommendations to reduce spending and stay within budget.
    Be practical and considerate. Format as a numbered list.
    """
    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=GROQ_MODEL
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return t(f"Unable to generate suggestions: {e}")

def validate_receipt(data, is_dup_bool, dup_id=None):
    """Validate receipt data"""
    results = {}

    sub = float(data.get('subtotal', 0))
    tax = float(data.get('tax', 0))
    total = float(data.get('total', 0))

    calculated_total = sub + tax

    if abs(calculated_total - total) <= 0.03:
        status_msg = t("Valid: {sub} + {tax} = {total}").format(
            sub=f"{sub:.2f}", tax=f"{tax:.2f}", total=f"{total:.2f}"
        )
        results['sum_check'] = (True, status_msg, f"{sub:.2f}+{tax:.2f}={total:.2f}")
    else:
        diff = calculated_total - total
        status_msg = t("Invalid: {sub} + {tax} != {total} (Diff: {diff:.2f})").format(
            sub=f"{sub:.2f}", tax=f"{tax:.2f}", total=f"{total:.2f}", diff=diff
        )
        results['sum_check'] = (False, status_msg, t("Exp: {calc}").format(calc=f"{calculated_total:.2f}"))

    if not is_dup_bool:
        results['dup'] = (True, t("No duplicate found"))
    else:
        results['dup'] = (False, t("Duplicate of Vault ID: {id}").format(id=dup_id))

    if sub > 0:
        rate = (tax / sub) * 100
        if 0 <= rate <= 30:
            results['tax_rate'] = (True, t("Tax Rate: {rate:.1f}% (Normal)").format(rate=rate))
        else:
            results['tax_rate'] = (False, t("Suspicious Tax Rate: {rate:.1f}%").format(rate=rate))
    else:
         results['tax_rate'] = (True, t("N/A (Subtotal is 0)"))

    missing = []
    if data['merchant'] == "Unknown": missing.append(t("Merchant"))
    if not data['date']: missing.append(t("Date"))
    if total == 0.0: missing.append(t("Total"))

    if not missing:
        results['fields'] = (True, t("All required fields present"))
    else:
        results['fields'] = (False, t("Missing: {fields}").format(fields=', '.join(missing)))

    return results

# ============================================================================
# CUSTOM CSS
# ============================================================================

def local_css():
    st.markdown("""
    <style>
    .parsing-header {
        background-color: #0E68CE;
        color: white;
        padding: 10px;
        border-radius: 5px 5px 0 0;
        font-weight: bold;
        display: flex;
        align-items: center;
    }
    .parsing-header svg { margin-right: 8px; }
    .comparison-container {
        border: 1px solid #ddd;
        border-top: none;
        padding: 15px;
        border-radius: 0 0 5px 5px;
        background-color: #f9f9f9;
    }
    .parsing-box {
        background-color: white;
        border: 1px solid #eee;
        border-radius: 8px;
        padding: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .parsing-title {
        font-weight: bold;
        font-size: 1.1em;
        margin-bottom: 10px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .accuracy-badge {
        background-color: #e3f2fd;
        color: #0d47a1;
        padding: 4px 8px;
        border-radius: 12px;
        font-size: 0.85em;
        font-weight: bold;
    }
    .field-row {
        display: flex;
        justify-content: space-between;
        margin-bottom: 8px;
        font-size: 0.95em;
    }
    .field-label { color: #666; }
    .field-value { font-weight: 500; color: #333; }
    .highlight-blue { color: #0E68CE; font-weight: bold; }
    .not-detected { color: #999; font-style: italic; }
    .feature-badges { margin-top: 15px; display: flex; gap: 10px; }
    .feature-badge {
        background-color: #e3f2fd;
        color: #0E68CE;
        padding: 5px 10px;
        border-radius: 15px;
        font-size: 0.85em;
        font-weight: 600;
        display: flex;
        align-items: center;
    }
    .feature-badge svg { margin-right: 5px; }
    </style>
    """, unsafe_allow_html=True)

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    st.set_page_config(page_title=t("Receipt & Invoice Digitizer"), layout="wide", page_icon="🧾")
    init_db()
    
    # Apply RTL styles if needed
    apply_rtl_styles()
    
    local_css()

    # Session State Initialization
    if 'current_receipt' not in st.session_state: st.session_state['current_receipt'] = None
    if 'current_line_items' not in st.session_state: st.session_state['current_line_items'] = []
    if 'validation_status' not in st.session_state: st.session_state['validation_status'] = None
    if 'is_key_valid' not in st.session_state: st.session_state['is_key_valid'] = False
    if 'pending_duplicate_save' not in st.session_state: st.session_state['pending_duplicate_save'] = False
    if 'duplicate_conflict_id' not in st.session_state: st.session_state['duplicate_conflict_id'] = None
    if 'last_uploaded_filename' not in st.session_state: st.session_state['last_uploaded_filename'] = ""
    if 'parsing_comparison' not in st.session_state: st.session_state['parsing_comparison'] = None
    if 'filters_applied' not in st.session_state: st.session_state['filters_applied'] = False
    if 'current_filters' not in st.session_state: st.session_state['current_filters'] = {}

    with st.sidebar:
        st.header(t("🔑 API Configuration"))
        
        # Language selector in sidebar
        with st.expander(t("🌐 Language / भाषा"), expanded=True):
            language_selector()
        
        user_groq_key = st.text_input(t("Enter Groq API Key"), type="password", help=t("Get your key at console.groq.com"))
        if user_groq_key:
            try:
                client = Groq(api_key=user_groq_key)
                client.models.list()
                st.session_state['is_key_valid'] = True
                st.success(t("✅ Valid API Key"), icon="🔐")
            except Exception as e:
                st.session_state['is_key_valid'] = False
                st.error(t("❌ Invalid API Key"), icon="⛔")
        else:
            st.session_state['is_key_valid'] = False

        st.divider()
        st.header(t("⚙️ Settings"))

        if st.button(t("🔄 Clear Cache")):
            st.cache_data.clear()
            st.toast(t("Cache cleared!"), icon="🔄")
            st.rerun()

        if st.button(t("🗑️ Clear Database")):
            clear_database()
            st.toast(t("Database cleared!"), icon="🗑️")
            st.rerun()

    st.title(t("🧾 Receipts & Invoice Digitizer"))

    tab_vault, tab_validation, tab_history, tab_analytics = st.tabs([
        t("📤 Upload & Process"), 
        t("✅ Extraction & Validation"), 
        t("📜 Bill History"), 
        t("📊 Analytics")
    ])

    # === TAB 1: UPLOAD & PROCESS ===
    with tab_vault:
        st.markdown(t("### 1. Document Ingestion"))
        uploaded_file = st.file_uploader(t("Upload Receipt"), type=["png", "jpg", "jpeg", "pdf"])

        if uploaded_file:
            st.session_state['last_uploaded_filename'] = uploaded_file.name
            if uploaded_file.type == "application/pdf":
                images = convert_pdf_to_images(uploaded_file)
                if images:
                    image = images[0]
                    st.info(t("PDF uploaded. Processing first page."))
                else:
                    st.error(t("Could not convert PDF to image."))
                    return
            else:
                image = Image.open(uploaded_file)

            cleaned_image = preprocess_image(image)
            st.subheader(t("Image Processing"))
            col_img1, col_img2 = st.columns(2)
            with col_img1:
                st.image(image, caption=t("Original Document"), use_container_width=True)
            with col_img2:
                st.image(cleaned_image, caption=t("Cleaned (Grayscale) for OCR"), use_container_width=True)

            st.divider()

            if st.button(t("🚀 Extract & Process"), type="primary", use_container_width=True):
                if not st.session_state['is_key_valid']:
                    st.error(t("Please enter a VALID Groq API Key in the sidebar first."))
                else:
                    with st.spinner(t("Running OCR, Standard AI Parsing, and Template Matching...")):
                        raw_text = extract_text(cleaned_image)

                        structured_data_standard = parse_with_groq(raw_text, user_groq_key)

                        if structured_data_standard:
                            receipt_data_standard = {
                                "merchant": structured_data_standard.get('merchant', 'Unknown'),
                                "date": structured_data_standard.get('date'),
                                "invoice_number": structured_data_standard.get('invoice_number', 'Unknown'),
                                "subtotal": float(structured_data_standard.get('subtotal') or 0),
                                "tax": float(structured_data_standard.get('tax') or 0),
                                "total": float(structured_data_standard.get('total') or 0),
                                "category": structured_data_standard.get('category', 'Uncategorized')
                            }
                            line_items = structured_data_standard.get('line_items', [])

                            standard_score = calculate_parsing_score(receipt_data_standard)
                            receipt_data_template, template_applied = apply_template_parsing(receipt_data_standard)
                            template_score = calculate_parsing_score(receipt_data_template)

                            st.session_state['parsing_comparison'] = {
                                'standard': receipt_data_standard,
                                'standard_score': standard_score,
                                'template': receipt_data_template,
                                'template_score': template_score,
                                'template_applied': template_applied
                            }

                            final_receipt_data = receipt_data_template
                            if not final_receipt_data['date']:
                                 final_receipt_data['date'] = datetime.now().strftime("%Y-%m-%d")

                            # Translate for display
                            translated_receipt = dynamic_translator.translate_receipt(final_receipt_data)
                            translated_items = dynamic_translator.translate_line_items(line_items)
                            
                            st.session_state['current_receipt'] = translated_receipt
                            st.session_state['current_line_items'] = translated_items

                            is_dup, _, conflict_id = check_if_receipt_exists(
                                final_receipt_data['merchant'], final_receipt_data['date'],
                                final_receipt_data['total'], final_receipt_data['invoice_number']
                            )

                            val_results = validate_receipt(translated_receipt, is_dup, conflict_id)
                            st.session_state['validation_status'] = val_results

                            if is_dup:
                                st.session_state['pending_duplicate_save'] = True
                                st.session_state['duplicate_conflict_id'] = conflict_id
                                st.warning(t("⚠️ Duplicate Detected! Matches Vault ID: {id}").format(id=conflict_id))
                            else:
                                st.session_state['pending_duplicate_save'] = False
                                st.session_state['duplicate_conflict_id'] = None
                                new_id = save_receipt_to_db(final_receipt_data, uploaded_file.name, line_items)

                                st.cache_data.clear()

                                st.success(t("Processing Complete! Added to Vault with ID: {id}").format(id=new_id))
                                st.toast(t("Receipt processed successfully!"), icon="✅")

                        else:
                            st.error(t("AI could not parse the receipt."))

            if st.session_state.get('pending_duplicate_save'):
                conflict_id = st.session_state.get('duplicate_conflict_id')
                st.error(t("This receipt duplicates Vault ID: {id} (Same Date, Vendor & Amount)").format(id=conflict_id))
                col_force, col_view = st.columns(2)
                with col_force:
                    if st.button(t("⚠️ Ignore & Force Save")):
                        r_data = st.session_state['current_receipt']
                        l_items = st.session_state['current_line_items']
                        f_name = st.session_state['last_uploaded_filename']
                        save_receipt_to_db(r_data, f_name, l_items)

                        st.cache_data.clear()

                        st.session_state['pending_duplicate_save'] = False
                        st.success(t("Forced save successful!"))
                        st.rerun()
                with col_view:
                    st.info(t("Go to 'Bill History' and search ID {id} to compare.").format(id=conflict_id))

    # === TAB 2: EXTRACTION & VALIDATION ===
    with tab_validation:
        st.markdown(f"""
        <div class="parsing-header">
            <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M20 14.66V20a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h5.34"></path><polygon points="18 2 22 6 12 16 8 16 8 12 18 2"></polygon></svg>
            {t('Template-Based Parsing')}
        </div>
        """, unsafe_allow_html=True)

        with st.container():
            st.markdown('<div class="comparison-container">', unsafe_allow_html=True)

            comparison_data = st.session_state.get('parsing_comparison')

            if comparison_data:
                std = comparison_data['standard']
                tmpl = comparison_data['template']
                std_score = comparison_data['standard_score']
                tmpl_score = comparison_data['template_score']
                score_diff = tmpl_score - std_score

                std_tax_display = f"${std.get('tax', 0):.2f}"
                if std.get('tax') == 0 or std.get('tax') is None:
                    std_tax_display = '<span class="not-detected">' + t('Not detected') + '</span>'

                tmpl_tax_val = tmpl.get('tax', 0)
                tmpl_tax_display = f"${tmpl_tax_val:.2f}"
                if tmpl.get('tax_rate_applied'):
                     rate_pct = tmpl['tax_rate_applied'] * 100
                     tmpl_tax_display = f'<span class="highlight-blue">${tmpl_tax_val:.2f} ({rate_pct:.2f}%)</span>'
                elif tmpl_tax_val == 0:
                     tmpl_tax_display = '<span class="not-detected">' + t('Not detected') + '</span>'

                c1, c2 = st.columns(2)

                with c1:
                    st.markdown(f"""
                    <div class="parsing-box">
                        <div class="parsing-title">
                            {t('Standard Parsing')}
                            <span class="accuracy-badge">{std_score}% {t('Accuracy')}</span>
                        </div>
                        <div class="field-row"><span class="field-label">{t('Date')}:</span> <span class="field-value">{std.get('date', 'N/A')}</span></div>
                        <div class="field-row"><span class="field-label">{t('Vendor')}:</span> <span class="field-value">{std.get('merchant', 'Unknown')}</span></div>
                        <div class="field-row"><span class="field-label">{t('Total')}:</span> <span class="field-value">${std.get('total', 0):.2f}</span></div>
                        <div class="field-row"><span class="field-label">{t('Tax')}:</span> <span class="field-value">{std_tax_display}</span></div>
                    </div>
                    """, unsafe_allow_html=True)

                with c2:
                     st.markdown(f"""
                    <div class="parsing-box">
                        <div class="parsing-title">
                            {t('Template Parsing')}
                            <span class="accuracy-badge">{tmpl_score}% {t('Accuracy')}</span>
                        </div>
                        <div class="field-row"><span class="field-label">{t('Date')}:</span> <span class="field-value">{tmpl.get('date', 'N/A')}</span></div>
                        <div class="field-row"><span class="field-label">{t('Vendor')}:</span> <span class="field-value" style="color: #0E68CE;">{tmpl.get('merchant', 'Unknown')}</span></div>
                        <div class="field-row"><span class="field-label">{t('Total')}:</span> <span class="field-value">${tmpl.get('total', 0):.2f}</span></div>
                        <div class="field-row"><span class="field-label">{t('Tax')}:</span> <span class="field-value">{tmpl_tax_display}</span></div>
                    </div>
                    """, unsafe_allow_html=True)

                if score_diff > 0:
                    st.markdown(f"""
                    <div class="feature-badge">
                        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="12" y1="5" x2="12" y2="19"></line><line x1="5" y1="12" x2="19" y2="12"></line></svg>
                        +{score_diff}% {t('Accuracy Improvement')}
                    </div>
                    """, unsafe_allow_html=True)

            else:
                st.info(t("Upload and process a document to see the template parsing comparison."))

            st.markdown('</div>', unsafe_allow_html=True)

        st.divider()

        st.markdown(t("### 🔍 Validation Details & Database Status"))
        if st.session_state['current_receipt']:
            data = st.session_state['current_receipt']
            items = st.session_state['current_line_items']
            val = st.session_state['validation_status']

            c_validate, c_db = st.columns(2)
            with c_validate:
                st.info(t("🔹 Parsing & Logic Validation"))
                if val:
                    res_sum = val['sum_check']
                    st.write(f"**{t('Sum Check')}**: {'✅' if res_sum[0] else '❌'} {res_sum[1]}")

                    res_dup = val['dup']
                    st.write(f"**{t('Duplicate Detected')}**: {'❌' if not res_dup[0] else '✅'} {res_dup[1]}")

                    res_tax = val['tax_rate']
                    st.write(f"**{t('Tax Logic')}**: {'✅' if res_tax[0] else '⚠️'} {res_tax[1]}")

                    res_fields = val['fields']
                    st.write(f"**{t('Required Fields')}**: {'✅' if res_fields[0] else '⚠️'} {res_fields[1]}")

            with c_db:
                st.info(t("🔹 Vault Status (Recent Entries)"))
                df_all = get_all_receipts()
                if not df_all.empty:
                    # Translate merchant and category for display
                    display_df = df_all[['id', 'merchant', 'date', 'total_amount']].head(5).copy()
                    if st.session_state['current_language'] != 'en':
                        display_df['merchant'] = display_df['merchant'].apply(
                            lambda x: translation_manager.translate_text(str(x), st.session_state['current_language'])
                        )
                    st.dataframe(display_df, hide_index=True, use_container_width=True,
                                column_config={
                                    "id": t("ID"),
                                    "merchant": t("Merchant"),
                                    "date": t("Date"),
                                    "total_amount": t("Amount")
                                })
                else:
                    st.write(t("Vault is empty."))
        else:
             st.warning(t("Please upload a document first to view validation details."))

    # === TAB 3: BILL HISTORY ===
    with tab_history:
        st.header(t("📜 Comprehensive Invoice Management & Analysis"))

        df_all_invoices = get_all_receipts()

        if df_all_invoices.empty:
            st.warning(t("⚠️ No invoices available in the vault yet. Upload some receipts to get started!"))
        else:
            st.info(t("📊 Total Invoices in Vault: **{count}**").format(count=len(df_all_invoices)))

            with st.expander(t("📂 Export Data (CSV / Excel)"), expanded=False):
                st.write(t("Download your vault data for accounting or external analysis."))
                df_items_export = get_all_line_items_global()
                col_exp1, col_exp2 = st.columns(2)

                with col_exp1:
                    st.subheader(t("📑 Receipts Summary"))
                    st.caption(t("One row per receipt (Totals, Dates, Merchants)."))
                    csv_receipts = df_all_invoices.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label=t("📥 Download CSV"),
                        data=csv_receipts,
                        file_name=f"receipts_summary_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv",
                        key='csv_rec'
                    )
                    buffer = io.BytesIO()
                    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                        df_all_invoices.to_excel(writer, sheet_name='Receipts', index=False)
                    buffer.seek(0)
                    st.download_button(
                        label=t("📥 Download Excel"),
                        data=buffer.getvalue(),
                        file_name=f"receipts_summary_{datetime.now().strftime('%Y%m%d')}.xlsx",
                        mime="application/vnd.ms-excel",
                        key='xls_rec'
                    )

                with col_exp2:
                    st.subheader(t("🛒 Itemized Details"))
                    st.caption(t("One row per line item (Product Name, Qty, Price + Receipt Info)."))
                    if not df_items_export.empty:
                        csv_items = df_items_export.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label=t("📥 Download CSV"),
                            data=csv_items,
                            file_name=f"line_items_detailed_{datetime.now().strftime('%Y%m%d')}.csv",
                            mime="text/csv",
                            key='csv_item'
                        )
                        buffer_items = io.BytesIO()
                        with pd.ExcelWriter(buffer_items, engine='xlsxwriter') as writer:
                            df_items_export.to_excel(writer, sheet_name='Line Items', index=False)
                        buffer_items.seek(0)
                        st.download_button(
                            label=t("📥 Download Excel"),
                            data=buffer_items.getvalue(),
                            file_name=f"line_items_detailed_{datetime.now().strftime('%Y%m%d')}.xlsx",
                            mime="application/vnd.ms-excel",
                            key='xls_item'
                        )
                    else:
                        st.info(t("No line items available to export."))

            st.divider()

            # Advanced Search Section
            with st.expander(t("🔍 Advanced Search & Filters"), expanded=True):
                st.markdown(t("### Multi-Criteria Search"))

                col_s1, col_s2 = st.columns(2)

                with col_s1:
                    search_keyword = st.text_input(
                        t("🔎 Keyword Search"),
                        placeholder=t("Search vendor, invoice, category, or item name"),
                        key="adv_keyword",
                        value=st.session_state.current_filters.get('keyword', '')
                    )

                    st.markdown(t("**📅 Date Range**"))
                    col_d1, col_d2 = st.columns(2)
                    date_from_val = None
                    if 'date_from' in st.session_state.current_filters:
                        try:
                            date_from_val = date.fromisoformat(st.session_state.current_filters['date_from'])
                        except ValueError:
                            date_from_val = None
                    with col_d1:
                        date_from = st.date_input(t("From"), value=date_from_val, key="date_from")

                    date_to_val = None
                    if 'date_to' in st.session_state.current_filters:
                        try:
                            date_to_val = date.fromisoformat(st.session_state.current_filters['date_to'])
                        except ValueError:
                            date_to_val = None
                    with col_d2:
                        date_to = st.date_input(t("To"), value=date_to_val, key="date_to")

                with col_s2:
                    categories = get_unique_categories()
                    # Translate category names for display
                    translated_categories = ['All'] + [
                        translation_manager.translate_text(cat, st.session_state['current_language']) 
                        if cat != 'All' else 'All' 
                        for cat in categories[1:]
                    ]
                    
                    current_category_idx = 0
                    if st.session_state.current_filters.get('category'):
                        try:
                            original_cat = st.session_state.current_filters['category']
                            if original_cat in categories:
                                current_category_idx = categories.index(original_cat)
                        except ValueError:
                            current_category_idx = 0
                    
                    selected_category_display = st.selectbox(
                        t("📁 Category"), 
                        translated_categories, 
                        key="filter_category_display", 
                        index=current_category_idx
                    )
                    
                    # Map back to original category for filtering
                    if selected_category_display != 'All':
                        cat_idx = translated_categories.index(selected_category_display)
                        selected_category = categories[cat_idx]
                    else:
                        selected_category = 'All'

                    merchants = get_unique_merchants()
                    # Translate merchant names for display
                    translated_merchants = ['All'] + [
                        translation_manager.translate_text(merch, st.session_state['current_language']) 
                        if merch != 'All' else 'All' 
                        for merch in merchants[1:]
                    ]
                    
                    current_merchant_idx = 0
                    if st.session_state.current_filters.get('merchant'):
                        try:
                            original_merch = st.session_state.current_filters['merchant']
                            if original_merch in merchants:
                                current_merchant_idx = merchants.index(original_merch)
                        except ValueError:
                            current_merchant_idx = 0
                    
                    selected_merchant_display = st.selectbox(
                        t("🏪 Merchant"), 
                        translated_merchants, 
                        key="filter_merchant_display", 
                        index=current_merchant_idx
                    )
                    
                    # Map back to original merchant for filtering
                    if selected_merchant_display != 'All':
                        merch_idx = translated_merchants.index(selected_merchant_display)
                        selected_merchant = merchants[merch_idx]
                    else:
                        selected_merchant = 'All'

                st.markdown(t("**💰 Amount Range**"))
                col_a1, col_a2 = st.columns(2)
                with col_a1:
                    amount_min = st.number_input(
                        t("Min Amount ($"), 
                        min_value=0.0, 
                        value=st.session_state.current_filters.get('amount_min', 0.0), 
                        step=10.0, 
                        key="amount_min"
                    )
                with col_a2:
                    amount_max = st.number_input(
                        t("Max Amount ($"), 
                        min_value=0.0, 
                        value=st.session_state.current_filters.get('amount_max', 10000.0), 
                        step=10.0, 
                        key="amount_max"
                    )

                col_apply, col_reset = st.columns([1, 1])
                with col_apply:
                    if st.button(t("🔍 Apply Filters"), type="primary", use_container_width=True):
                        st.session_state['filters_applied'] = True
                        temp_filters = {}
                        if search_keyword: temp_filters['keyword'] = search_keyword
                        if date_from: temp_filters['date_from'] = date_from.strftime("%Y-%m-%d")
                        if date_to: temp_filters['date_to'] = date_to.strftime("%Y-%m-%d")
                        if amount_min > 0: temp_filters['amount_min'] = amount_min
                        if amount_max < 10000: temp_filters['amount_max'] = amount_max
                        if selected_category != 'All': temp_filters['category'] = selected_category
                        if selected_merchant != 'All': temp_filters['merchant'] = selected_merchant
                        st.session_state['current_filters'] = temp_filters
                        st.rerun()

                with col_reset:
                    if st.button(t("🔄 Reset"), use_container_width=True):
                        st.session_state['filters_applied'] = False
                        st.session_state['current_filters'] = {}
                        st.rerun()

            # Get filtered results
            if st.session_state['filters_applied'] and st.session_state['current_filters']:
                df_filtered = advanced_search_receipts(st.session_state['current_filters'])
                st.success(t("✅ Found {count} matching receipts").format(count=len(df_filtered)))
            else:
                df_filtered = df_all_invoices

            st.divider()

            # Month filter
            available_months = get_available_months()
            if available_months:
                available_months.insert(0, "All Months")
            else:
                available_months = ["All Months"]
            
            translated_months = [t(month) if month != "All Months" else t("All Months") for month in available_months]
            selected_month_display = st.selectbox(t("📅 Filter by Month"), translated_months, key="month_filter_display")
            
            # Map back to original month value
            if selected_month_display != t("All Months"):
                month_idx = translated_months.index(selected_month_display)
                selected_month = available_months[month_idx]
            else:
                selected_month = "All Months"

            # Apply month filter
            if selected_month != "All Months":
                month_str = selected_month
                df_filtered = df_filtered[df_filtered['upload_timestamp'].str.startswith(month_str)]

            st.divider()

            # Budget Management
            if selected_month != "All Months":
                st.subheader(t(f"💰 Budget Management - {selected_month}"))
                monthly_total = df_filtered['total_amount'].sum() if not df_filtered.empty else 0
                monthly_budget = get_monthly_budget(selected_month)
                
                col_budget1, col_budget2, col_budget3 = st.columns(3)
                with col_budget1:
                    st.metric(t("Total Spending"), LanguageFormatter.format_currency(monthly_total))
                with col_budget2:
                    if monthly_budget > 0:
                        st.metric(t("Monthly Budget"), LanguageFormatter.format_currency(monthly_budget))
                    else:
                        st.metric(t("Monthly Budget"), t("Not Set"))
                with col_budget3:
                    if monthly_budget > 0:
                        budget_usage = (monthly_total / monthly_budget) * 100
                        st.metric(t("Budget Usage"), f"{budget_usage:.1f}%")
                
                if monthly_budget > 0:
                    st.progress(min(monthly_total / monthly_budget, 1.0))
                
                with st.expander(t("⚙️ Set/Update Monthly Budget")):
                    new_budget = st.number_input(
                        t(f"Budget for {selected_month}"),
                        min_value=0.0,
                        value=float(monthly_budget),
                        step=100.0,
                        key="budget_input"
                    )
                    if st.button(t("💾 Save Budget")):
                        set_monthly_budget(selected_month, new_budget)
                        st.success(t(f"Budget for {selected_month} set to {LanguageFormatter.format_currency(new_budget)}"))
                        st.rerun()
                
                st.divider()

                st.subheader(t("📊 Category-Wise Spending Analysis"))
                if 'category' in df_filtered.columns and not df_filtered.empty:
                    category_spending = df_filtered.groupby('category')['total_amount'].sum().reset_index()
                    category_spending.columns = [t('Category'), t('Total Spend')]
                    
                    # Translate categories
                    if st.session_state['current_language'] != 'en':
                        category_spending[t('Category')] = category_spending[t('Category')].apply(
                            lambda x: translation_manager.translate_text(str(x), st.session_state['current_language'])
                        )
                    
                    category_spending = category_spending.sort_values(t('Total Spend'), ascending=False)
                    st.dataframe(category_spending, use_container_width=True, hide_index=True)

                    if monthly_budget > 0:
                        if monthly_total <= monthly_budget:
                            remaining = monthly_budget - monthly_total
                            st.success(t("✅ You're within budget! Remaining: {amount}").format(
                                amount=LanguageFormatter.format_currency(remaining)
                            ))
                        else:
                            overspend = monthly_total - monthly_budget
                            st.warning(t("⚠️ Budget exceeded by {amount}").format(
                                amount=LanguageFormatter.format_currency(overspend)
                            ))
                            if st.session_state['is_key_valid']:
                                with st.expander(t("🤖 AI Budget Optimization Suggestions"), expanded=True):
                                    if st.button(t("Generate AI Recommendations")):
                                        with st.spinner(t("Analyzing your spending patterns...")):
                                            category_dict = dict(zip(
                                                category_spending[t('Category')], 
                                                category_spending[t('Total Spend')]
                                            ))
                                            suggestions = get_ai_budget_suggestions(
                                                monthly_total,
                                                monthly_budget,
                                                category_dict,
                                                user_groq_key
                                            )
                                            st.markdown(suggestions)
                            else:
                                st.info(t("💡 Enter a valid Groq API key in the sidebar to get AI-powered budget suggestions."))
                st.divider()

            st.subheader(t("📋 Invoice List"))
            if not df_filtered.empty:
                required_cols = ['id', 'merchant', 'date', 'total_amount', 'category', 'invoice_number']
                for col in required_cols:
                    if col not in df_filtered.columns:
                        if col == 'category':
                            df_filtered[col] = 'Uncategorized'
                        else:
                            df_filtered[col] = ''
                
                display_df = df_filtered[required_cols].copy()
                
                # Translate merchant and category for display
                if st.session_state['current_language'] != 'en':
                    display_df['merchant'] = display_df['merchant'].apply(
                        lambda x: translation_manager.translate_text(str(x), st.session_state['current_language'])
                    )
                    display_df['category'] = display_df['category'].apply(
                        lambda x: translation_manager.translate_text(str(x), st.session_state['current_language'])
                    )
                
                display_df.columns = [t('ID'), t('Vendor'), t('Date'), t('Amount'), t('Category'), t('Invoice #')]

                selected_id = None
                
                cols_w = [0.6, 0.8, 3.5, 1.2, 1.2, 1.5]
                header_cols = st.columns(cols_w)
                header_cols[0].markdown(f"**{t('Select')}**")
                header_cols[1].markdown(f"**{t('ID')}**")
                header_cols[2].markdown(f"**{t('Vendor')}**")
                header_cols[3].markdown(f"**{t('Date')}**")
                header_cols[4].markdown(f"**{t('Amount')}**")
                header_cols[5].markdown(f"**{t('Invoice #')}**")

                id_list = display_df[t('ID')].tolist()
                for _, row in display_df.iterrows():
                    rcols = st.columns(cols_w)
                    rid = int(row[t('ID')])
                    key = f"invoice_chk_{rid}"
                    if key not in st.session_state:
                        st.session_state[key] = False
                    checked = rcols[0].checkbox("", value=st.session_state.get(key, False), key=key)
                    if checked:
                        for other in id_list:
                            ok = f"invoice_chk_{int(other)}"
                            if ok != key and st.session_state.get(ok, False):
                                st.session_state[ok] = False
                        selected_id = rid
                    rcols[1].markdown(f"**{rid}**")
                    rcols[2].markdown(f"{row[t('Vendor')]}")
                    rcols[3].markdown(f"{row[t('Date')]}")
                    rcols[4].markdown(f"{LanguageFormatter.format_currency(row[t('Amount')])}")
                    rcols[5].markdown(f"{row[t('Invoice #')]}" if pd.notna(row[t('Invoice #')]) else "")

                st.markdown("---")
                if selected_id:
                    st.divider()
                    st.subheader(t(f"📄 Invoice Details - ID: {selected_id}"))
                    selected_receipt = get_receipt_by_id(selected_id)
                    if not selected_receipt.empty:
                        receipt_row = selected_receipt.iloc[0]
                        
                        # Translate for display
                        merchant_display = translation_manager.translate_text(
                            receipt_row['merchant'], 
                            st.session_state['current_language']
                        ) if st.session_state['current_language'] != 'en' else receipt_row['merchant']
                        
                        category_display = translation_manager.translate_text(
                            receipt_row.get('category', 'Uncategorized'), 
                            st.session_state['current_language']
                        ) if st.session_state['current_language'] != 'en' else receipt_row.get('category', 'Uncategorized')
                        
                        col_detail1, col_detail2, col_detail3, col_detail4 = st.columns(4)
                        col_detail1.metric(t("🏪 Vendor"), merchant_display)
                        col_detail2.metric(t("💵 Total Amount"), LanguageFormatter.format_currency(receipt_row['total_amount']))
                        col_detail3.metric(t("📁 Category"), category_display)
                        col_detail4.metric(t("📅 Date"), receipt_row['date'])
                        
                        st.markdown(f"**{t('Invoice Number')}:** {receipt_row['invoice_number']}")
                        st.markdown(f"**{t('Uploaded')}:** {receipt_row['upload_timestamp']}")
                        st.divider()

                        st.subheader(t("🛒 Itemized Breakdown"))
                        line_items_df = get_line_items(selected_id)
                        if not line_items_df.empty:
                            line_items_df['qty'] = pd.to_numeric(line_items_df['qty'], errors='coerce').fillna(0).astype(int)
                            line_items_df['price'] = pd.to_numeric(line_items_df['price'], errors='coerce').fillna(0.0)
                            
                            # Translate item names
                            if st.session_state['current_language'] != 'en':
                                line_items_df['name'] = line_items_df['name'].apply(
                                    lambda x: translation_manager.translate_text(str(x), st.session_state['current_language'])
                                )
                            
                            line_items_df[t('Total')] = line_items_df['qty'] * line_items_df['price']
                            line_items_df.columns = [t('Item Name'), t('Quantity'), t('Price'), t('Total')]
                            
                            # Format currency columns
                            line_items_df[t('Price')] = line_items_df[t('Price')].apply(
                                lambda x: LanguageFormatter.format_currency(x, symbol=True)
                            )
                            line_items_df[t('Total')] = line_items_df[t('Total')].apply(
                                lambda x: LanguageFormatter.format_currency(x, symbol=True)
                            )
                            
                            st.dataframe(line_items_df, use_container_width=True, hide_index=True)
                            
                            st.markdown(t("#### 📤 Export This Invoice"))
                            csv_single = line_items_df.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label=t("📥 Download Items as CSV"),
                                data=csv_single,
                                file_name=f"invoice_{selected_id}_items.csv",
                                mime="text/csv",
                                key=f"csv_single_{selected_id}"
                            )
                        else:
                            st.info(t("No detailed line items found for this invoice."))
                        st.divider()

                        st.markdown(t("### 🗑️ Delete Invoice"))
                        st.warning(t("⚠️ This action cannot be undone. All related line items will also be deleted."))
                        delete_confirm = st.checkbox(
                            t(f"I confirm I want to delete Invoice ID: {selected_id}"), 
                            key=f"del_confirm_{selected_id}"
                        )
                        if delete_confirm:
                            if st.button(t(f"🗑️ Permanently Delete Invoice {selected_id}"), type="primary"):
                                delete_receipt(selected_id)
                                st.success(t(f"✅ Invoice {selected_id} has been permanently deleted."))
                                st.rerun()
            else:
                st.info(t("No invoices match the current filters."))

    # === TAB 4: ANALYTICS ===
    with tab_analytics:
        st.header(t("📊 Spending Analytics"))

        summary = get_analytics_summary()

        if summary is not None and summary['total_receipts'] > 0:
            kpi1, kpi2, kpi3, kpi4 = st.columns(4)

            total_spend = summary['total_spend']
            avg_ticket = summary['avg_ticket']
            total_tax = summary['total_tax']
            latest_date = summary['latest_date']

            try:
                recent_date = LanguageFormatter.format_date(latest_date)
            except:
                recent_date = latest_date

            kpi1.metric(t("Total Spend"), LanguageFormatter.format_currency(total_spend))
            kpi2.metric(t("Avg Receipt"), LanguageFormatter.format_currency(avg_ticket))
            kpi3.metric(t("Total Tax Paid"), LanguageFormatter.format_currency(total_tax))
            kpi4.metric(t("Last Purchase"), recent_date)

            st.divider()

            st.subheader(t("🏢 Merchant Breakdown"))
            col_a, col_b = st.columns(2)

            merchant_df = get_merchant_breakdown()
            
            # Translate merchant names for display
            if st.session_state['current_language'] != 'en' and not merchant_df.empty:
                merchant_df['merchant'] = merchant_df['merchant'].apply(
                    lambda x: translation_manager.translate_text(str(x), st.session_state['current_language'])
                )

            with col_a:
                if not merchant_df.empty:
                    fig_bar = px.bar(merchant_df, x='merchant', y='total_spend', color='merchant',
                                 title=t("Total Spend per Vendor"),
                                 text_auto='.2s')
                    fig_bar.update_layout(xaxis_title=t("Merchant"), yaxis_title=t("Amount ($)"))
                    st.plotly_chart(fig_bar, use_container_width=True)
                    st.markdown(t("**Insight**: This chart shows how much money has been spent at each vendor. Identifying the top vendors can help in negotiating better deals or optimizing spending habits if certain merchants consume a significant portion of the budget."))

            with col_b:
                if not merchant_df.empty:
                    fig_pie = px.pie(merchant_df, values='total_spend', names='merchant',
                                 title=t("Share of Wallet (Spend %)"),
                                 hole=0.4)
                    st.plotly_chart(fig_pie, use_container_width=True)
                    st.markdown(t("**Insight**: This pie chart illustrates the proportional contribution of each merchant to the total spending. A large slice indicates a dominant vendor, which might be an area to explore for cost-saving opportunities or loyalty program benefits."))

            st.subheader(t("📅 Time Trends"))

            time_series_df = get_time_series_data()

            if not time_series_df.empty:
                col_c, col_d = st.columns(2)

                with col_c:
                    fig_line = px.line(time_series_df, x='month_year', y='monthly_total', markers=True,
                                   title=t("Monthly Spending Trend"),
                                   labels={'month_year': t('Month'), 'monthly_total': t('Amount ($)')})
                    st.plotly_chart(fig_line, use_container_width=True)
                    st.markdown(t("**Insight**: This line graph visualizes spending patterns over months. Upward trends could indicate increasing expenses, while downward trends might suggest cost-saving measures or seasonal variations in spending."))

                with col_d:
                    time_series_df['cumulative'] = time_series_df['monthly_total'].cumsum()
                    fig_area = px.area(time_series_df, x='month_year', y='cumulative',
                                   title=t("Cumulative Spending Over Time"),
                                   labels={'month_year': t('Month'), 'cumulative': t('Running Total ($)')})
                    st.plotly_chart(fig_area, use_container_width=True)
                    st.markdown(t("**Insight**: This area chart displays the total accumulated spending over time. A steep curve indicates rapid spending, whereas a flatter curve suggests more controlled expenditure. This helps in understanding overall financial progression."))

            st.subheader(t("🧠 Spending Behavior"))
            col_e, col_f = st.columns(2)

            with col_e:
                day_df = get_day_of_week_spending()
                if not day_df.empty:
                    # Translate day names
                    if st.session_state['current_language'] != 'en':
                        day_df['day_name'] = day_df['day_name'].apply(
                            lambda x: translation_manager.translate_text(x, st.session_state['current_language'])
                        )
                    
                    days_order = [t('Monday'), t('Tuesday'), t('Wednesday'), t('Thursday'), t('Friday'), t('Saturday'), t('Sunday')]
                    day_df['day_name'] = pd.Categorical(day_df['day_name'], categories=days_order, ordered=True)
                    day_df = day_df.sort_values('day_name')

                    fig_day = px.bar(day_df, x='day_name', y='total_spend',
                                 title=t("Spending by Day of Week"),
                                 color='total_spend', color_continuous_scale='Viridis')
                    fig_day.update_layout(xaxis_title=t("Day"), yaxis_title=t("Amount ($)"))
                    st.plotly_chart(fig_day, use_container_width=True)
                    st.markdown(t("**Insight**: This bar chart reveals spending habits by day of the week. Peaks on certain days might correspond to routine activities like weekend shopping or regular weekly expenses, offering insights into lifestyle spending."))

            with col_f:
                df = get_all_receipts()
                if not df.empty:
                    fig_hist = px.histogram(df, x='total_amount', nbins=20,
                                        title=t("Distribution of Receipt Amounts"),
                                        labels={'total_amount': t('Receipt Value ($)')})
                    st.plotly_chart(fig_hist, use_container_width=True)
                    st.markdown(t("**Insight**: This histogram shows the frequency distribution of receipt amounts. It helps identify common spending thresholds; for instance, many small transactions versus a few large purchases, which can inform budgeting strategies."))

            st.subheader(t("🛒 Item Analysis"))

            top_items_df = get_top_items(10)
            if not top_items_df.empty:
                # Translate item names
                if st.session_state['current_language'] != 'en':
                    top_items_df['item_name'] = top_items_df['item_name'].apply(
                        lambda x: translation_manager.translate_text(str(x), st.session_state['current_language'])
                    )
                
                fig_items = px.bar(top_items_df, x='purchase_count', y='item_name', orientation='h',
                               title=t("Top 10 Most Frequent Items"),
                               color='purchase_count',
                               labels={'purchase_count': t('Purchase Count'), 'item_name': t('Item Name')})
                fig_items.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig_items, use_container_width=True)
                st.markdown(t("**Insight**: This chart highlights the top 10 most frequently purchased items. This can reveal recurring needs or popular products, useful for inventory management if applicable, or for understanding regular consumption patterns."))

        else:
            st.info(t("No data in vault yet. Upload some receipts to see analytics!"))

if __name__ == "__main__":
    main()
