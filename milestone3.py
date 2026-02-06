import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px
from PIL import Image, ImageOps
import pytesseract
from datetime import datetime
import json
from groq import Groq
import io  # Required for file export buffers

# --- CONFIGURATION ---
DB_NAME = 'receipt_vault_v6.db'
GROQ_MODEL = "llama-3.3-70b-versatile" 

# --- DATABASE FUNCTIONS ---
def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS receipts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    merchant TEXT,
                    date TEXT,
                    invoice_number TEXT,
                    subtotal REAL,
                    tax REAL,
                    total_amount REAL,
                    filename TEXT,
                    upload_timestamp TEXT
                )''')
    c.execute('''CREATE TABLE IF NOT EXISTS line_items (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    receipt_id INTEGER,
                    name TEXT,
                    qty INTEGER,
                    price REAL,
                    FOREIGN KEY (receipt_id) REFERENCES receipts (id)
                )''')
    conn.commit()
    conn.close()

def check_if_receipt_exists(merchant, date, total, invoice_num):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT id, merchant, total_amount, invoice_number FROM receipts WHERE date = ?", (date,))
    candidates = c.fetchall()
    conn.close()
    
    for row in candidates:
        db_id, db_merch, db_total, db_inv = row
        if invoice_num and invoice_num != "Unknown" and db_inv and db_inv != "Unknown":
            if invoice_num == db_inv:
                return True, 1, db_id 
        
        price_match = abs(db_total - total) <= 0.05
        m1 = merchant.lower().strip()
        m2 = db_merch.lower().strip()
        merchant_match = (m1 in m2 or m2 in m1)
        
        if price_match and merchant_match:
            return True, 1, db_id
            
    return False, 0, None

def save_receipt_to_db(data, filename, line_items_data):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    upload_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute("""INSERT INTO receipts 
                 (merchant, date, invoice_number, subtotal, tax, total_amount, filename, upload_timestamp) 
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
              (data['merchant'], data['date'], data['invoice_number'], 
               data['subtotal'], data['tax'], data['total'], filename, upload_time))
    receipt_id = c.lastrowid
    for item in line_items_data:
        c.execute("INSERT INTO line_items (receipt_id, name, qty, price) VALUES (?, ?, ?, ?)",
                  (receipt_id, item['name'], item['qty'], item['price']))
    conn.commit()
    conn.close()
    return receipt_id

def get_all_receipts():
    conn = sqlite3.connect(DB_NAME)
    try:
        df = pd.read_sql_query("SELECT * FROM receipts ORDER BY id DESC", conn)
    except:
        df = pd.DataFrame()
    conn.close()
    return df

def get_receipt_by_id(receipt_id):
    conn = sqlite3.connect(DB_NAME)
    try:
        df = pd.read_sql_query("SELECT * FROM receipts WHERE id = ?", conn, params=(receipt_id,))
    except:
        df = pd.DataFrame()
    conn.close()
    return df

def get_line_items(receipt_id):
    conn = sqlite3.connect(DB_NAME)
    try:
        df = pd.read_sql_query("SELECT name, qty, price FROM line_items WHERE receipt_id = ?", conn, params=(receipt_id,))
    except:
        df = pd.DataFrame()
    conn.close()
    return df

def get_all_line_items_global():
    conn = sqlite3.connect(DB_NAME)
    try:
        query = """
            SELECT li.name, li.qty, li.price, r.merchant, r.date, r.invoice_number
            FROM line_items li
            JOIN receipts r ON li.receipt_id = r.id
        """
        df = pd.read_sql_query(query, conn)
    except:
        df = pd.DataFrame()
    conn.close()
    return df

def delete_receipt(receipt_id):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("DELETE FROM line_items WHERE receipt_id = ?", (receipt_id,))
    c.execute("DELETE FROM receipts WHERE id = ?", (receipt_id,))
    conn.commit()
    conn.close()

def clear_database():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("DELETE FROM line_items")
    c.execute("DELETE FROM receipts")
    conn.commit()
    conn.close()

# --- PROCESSING FUNCTIONS ---
def preprocess_image(image):
    return ImageOps.grayscale(image)

def extract_text(image):
    return pytesseract.image_to_string(image)

def parse_with_groq(raw_text, api_key):
    client = Groq(api_key=api_key)
    prompt = f"""
    Extract structured data from this receipt text. 
    Return ONLY a JSON object with these keys: 
    'merchant', 'date' (YYYY-MM-DD), 'invoice_number', 'subtotal', 'tax', 'total', 
    and 'line_items' (a list of objects with 'name', 'qty', 'price').
    
    If 'subtotal' is missing but 'total' and 'tax' exist, calculate it.
    If 'tax' is missing, try to infer it or set to 0.
    
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
        st.error(f"Groq Parsing Error: {e}")
        return None

# --- VALIDATION LOGIC ---
def validate_receipt(data, is_dup_bool, dup_id=None):
    results = {}
    
    # 1. Tax & Sum Logic Validation
    sub = float(data.get('subtotal', 0))
    tax = float(data.get('tax', 0))
    total = float(data.get('total', 0))
    
    calculated_total = sub + tax
    
    # Allow 3 cent rounding error
    if abs(calculated_total - total) <= 0.03:
        status_msg = f"Valid: {sub:.2f} + {tax:.2f} = {total:.2f}"
        results['sum_check'] = (True, status_msg, f"{sub:.2f}+{tax:.2f}={total:.2f}")
    else:
        diff = calculated_total - total
        status_msg = f"Invalid: {sub:.2f} + {tax:.2f} != {total:.2f} (Diff: {diff:.2f})"
        results['sum_check'] = (False, status_msg, f"Exp: {calculated_total:.2f}")

    # 2. Duplicate Validation
    if not is_dup_bool:
        results['dup'] = (True, f"No duplicate found")
    else:
        results['dup'] = (False, f"Duplicate of Vault ID: {dup_id}")

    # 3. Tax Rate Plausibility
    if sub > 0:
        rate = (tax / sub) * 100
        if 0 <= rate <= 30:
            results['tax_rate'] = (True, f"Tax Rate: {rate:.1f}% (Normal)")
        else:
            results['tax_rate'] = (False, f"Suspicious Tax Rate: {rate:.1f}%")
    else:
         results['tax_rate'] = (True, "N/A (Subtotal is 0)")

    # 4. Missing Fields
    missing = []
    if data['merchant'] == "Unknown": missing.append("Merchant")
    if not data['date']: missing.append("Date")
    if total == 0.0: missing.append("Total")
    
    if not missing:
        results['fields'] = (True, "All required fields present")
    else:
        results['fields'] = (False, f"Missing: {', '.join(missing)}")

    return results

# --- MAIN APP ---
def main():
    st.set_page_config(page_title="Receipt Vault & Validator", layout="wide", page_icon="🧾")
    init_db()

    if 'current_receipt' not in st.session_state: st.session_state['current_receipt'] = None
    if 'current_line_items' not in st.session_state: st.session_state['current_line_items'] = []
    if 'validation_status' not in st.session_state: st.session_state['validation_status'] = None
    if 'is_key_valid' not in st.session_state: st.session_state['is_key_valid'] = False
    
    if 'pending_duplicate_save' not in st.session_state: st.session_state['pending_duplicate_save'] = False
    if 'duplicate_conflict_id' not in st.session_state: st.session_state['duplicate_conflict_id'] = None
    if 'last_uploaded_filename' not in st.session_state: st.session_state['last_uploaded_filename'] = ""
    if 'view_receipt_id' not in st.session_state: st.session_state['view_receipt_id'] = None

    with st.sidebar:
        st.header("🔑 API Configuration")
        user_groq_key = st.text_input("Enter Groq API Key", type="password", help="Get your key at console.groq.com")
        
        if user_groq_key:
            try:
                client = Groq(api_key=user_groq_key)
                client.models.list() 
                st.session_state['is_key_valid'] = True
                st.success("✅ Valid API Key", icon="🔐")
            except Exception as e:
                st.session_state['is_key_valid'] = False
                st.error("❌ Invalid API Key", icon="⛔")
        else:
            st.session_state['is_key_valid'] = False

        st.divider()
        st.header("⚙️ Settings")
        if st.button("Clear Database"):
            clear_database()
            st.toast("Database cleared!", icon="🗑️")
            st.rerun()

    st.title("🧾 Receipt Vault System")
    
    tab_vault, tab_validation, tab_history, tab_analytics = st.tabs(["📤 Upload & Process", "✅ Extraction & Validation", "📜 Bill History", "📊 Analytics"])

    # === TAB 1: UPLOAD & PROCESS ===
    with tab_vault:
        st.markdown("### 1. Document Ingestion")
        uploaded_file = st.file_uploader("Upload Receipt", type=["png", "jpg", "jpeg"])
        
        if uploaded_file:
            st.session_state['last_uploaded_filename'] = uploaded_file.name
            image = Image.open(uploaded_file)
            cleaned_image = preprocess_image(image)

            st.subheader("Image Processing")
            col_img1, col_img2 = st.columns(2)
            with col_img1:
                st.image(image, caption="Original Receipt", use_container_width=True)
            with col_img2:
                st.image(cleaned_image, caption="Cleaned (Grayscale) for OCR", use_container_width=True)

            st.divider()

            if st.button("🚀 Extract & Process", type="primary", use_container_width=True):
                if not st.session_state['is_key_valid']:
                    st.error("Please enter a VALID Groq API Key in the sidebar first.")
                else:
                    with st.spinner("Running OCR & Groq AI Analysis..."):
                        raw_text = extract_text(cleaned_image)
                        structured_data = parse_with_groq(raw_text, user_groq_key)
                        
                        if structured_data:
                            receipt_data = {
                                "merchant": structured_data.get('merchant', 'Unknown'),
                                "date": structured_data.get('date', datetime.now().strftime("%Y-%m-%d")),
                                "invoice_number": structured_data.get('invoice_number', 'Unknown'),
                                "subtotal": float(structured_data.get('subtotal', 0)),
                                "tax": float(structured_data.get('tax', 0)),
                                "total": float(structured_data.get('total', 0))
                            }
                            line_items = structured_data.get('line_items', [])
                            
                            st.session_state['current_receipt'] = receipt_data
                            st.session_state['current_line_items'] = line_items
                            
                            # Check Exists
                            is_dup, _, conflict_id = check_if_receipt_exists(
                                receipt_data['merchant'], receipt_data['date'], 
                                receipt_data['total'], receipt_data['invoice_number']
                            )
                            
                            val_results = validate_receipt(receipt_data, is_dup, conflict_id)
                            st.session_state['validation_status'] = val_results
                            
                            if is_dup:
                                st.session_state['pending_duplicate_save'] = True 
                                st.session_state['duplicate_conflict_id'] = conflict_id
                                st.warning(f"⚠️ Duplicate Detected! Matches Vault ID: {conflict_id}")
                            else:
                                st.session_state['pending_duplicate_save'] = False
                                st.session_state['duplicate_conflict_id'] = None
                                new_id = save_receipt_to_db(receipt_data, uploaded_file.name, line_items)
                                st.success(f"Processing Complete! Added to Vault with ID: {new_id}")
                            
                            st.markdown("#### Quick Validation Check")
                            v1, v2, v3 = st.columns(3)
                            
                            # Sum Check Metric
                            sum_ok = val_results['sum_check'][0]
                            sum_help = val_results['sum_check'][2]
                            v1.metric("Sum Logic", "Pass" if sum_ok else "Fail", help=sum_help, delta_color="normal")
                            
                            if val_results['dup'][0]:
                                v2.metric("Duplicate", "None")
                            else:
                                v2.metric("Duplicate", f"ID {conflict_id}")

                            v3.metric("Tax Rate", "OK" if val_results['tax_rate'][0] else "Suspicious")
                        else:
                            st.error("AI could not parse the receipt.")

            if st.session_state.get('pending_duplicate_save'):
                conflict_id = st.session_state.get('duplicate_conflict_id')
                st.error(f"This receipt duplicates Vault ID: {conflict_id} (Same Date, Vendor & Amount).")
                col_force, col_view = st.columns(2)
                with col_force:
                    if st.button("⚠️ Ignore & Force Save"):
                        r_data = st.session_state['current_receipt']
                        l_items = st.session_state['current_line_items']
                        f_name = st.session_state['last_uploaded_filename']
                        save_receipt_to_db(r_data, f_name, l_items)
                        st.session_state['pending_duplicate_save'] = False
                        st.success("Forced save successful!")
                        st.rerun()
                with col_view:
                    st.info(f"Go to 'Bill History' and search ID {conflict_id} to compare.")

    # === TAB 2: DETAILED VALIDATION ===
    with tab_validation:
        st.markdown("## Field Extraction & Validation Details")
        
        if st.session_state['current_receipt']:
            data = st.session_state['current_receipt']
            items = st.session_state['current_line_items']
            val = st.session_state['validation_status']
            
            c_extract, c_validate, c_db = st.columns(3)
            with c_extract:
                st.info("🔹 Field Extraction")
                with st.container(border=True):
                    st.text_input("Vendor", value=data.get('merchant', ''), disabled=True)
                    st.text_input("Date", value=data.get('date', ''), disabled=True)
                    st.text_input("Invoice #", value=data.get('invoice_number', ''), disabled=True)
                    st.markdown("---")
                    c1, c2 = st.columns(2)
                    c1.text_input("Subtotal", value=f"{data.get('subtotal', 0):.2f}", disabled=True)
                    c2.text_input("Tax", value=f"{data.get('tax', 0):.2f}", disabled=True)
                    st.text_input("Total", value=f"{data.get('total', 0):.2f}", disabled=True)
                    if items:
                        st.dataframe(pd.DataFrame(items), hide_index=True, height=150)
            with c_validate:
                st.info("🔹 Validation Logic")
                if val:
                    res_sum = val['sum_check']
                    if res_sum[0]:
                        st.success(f"**Sum Check**: ✅ {res_sum[1]}")
                    else:
                        st.error(f"**Sum Check**: ❌ {res_sum[1]}")
                    
                    res_dup = val['dup']
                    st.write(f"**Duplicate**: {'✅' if res_dup[0] else '❌'} {res_dup[1]}")
                    
                    res_tax = val['tax_rate']
                    st.write(f"**Tax Rate**: {'✅' if res_tax[0] else '❌'} {res_tax[1]}")
            with c_db:
                st.info("🔹 Vault Status")
                df_all = get_all_receipts()
                if not df_all.empty:
                    st.metric("Total Vault Entries", len(df_all))
                    st.dataframe(df_all[['id', 'merchant', 'total_amount']].head(10), hide_index=True)
        else:
            st.warning("Please upload a document first.")

    # === TAB 3: BILL HISTORY (UPDATED WITH EXPORT) ===
    with tab_history:
        st.header("📜 Detailed Bill History & Management")
        
        # --- EXPORT SECTION ---
        with st.expander("📂 Export Data (CSV / Excel)", expanded=False):
            st.write("Download your vault data for accounting or external analysis.")
            
            # Prepare Dataframes
            df_receipts_export = get_all_receipts()
            df_items_export = get_all_line_items_global()
            
            col_exp1, col_exp2 = st.columns(2)
            
            # 1. Receipts Summary Export
            with col_exp1:
                st.subheader("📑 Receipts Summary")
                st.caption("One row per receipt (Totals, Dates, Merchants).")
                
                if not df_receipts_export.empty:
                    # CSV Button
                    csv_receipts = df_receipts_export.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download CSV",
                        data=csv_receipts,
                        file_name=f"receipts_summary_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv",
                        key='csv_rec'
                    )
                    
                    # Excel Button
                    buffer = io.BytesIO()
                    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                        df_receipts_export.to_excel(writer, sheet_name='Receipts', index=False)
                    
                    st.download_button(
                        label="Download Excel",
                        data=buffer.getvalue(),
                        file_name=f"receipts_summary_{datetime.now().strftime('%Y%m%d')}.xlsx",
                        mime="application/vnd.ms-excel",
                        key='xls_rec'
                    )
                else:
                    st.info("No receipts to export.")

            # 2. Itemized Export
            with col_exp2:
                st.subheader("🛒 Itemized Details")
                st.caption("One row per line item (Product Name, Qty, Price + Receipt Info).")
                
                if not df_items_export.empty:
                    # CSV Button
                    csv_items = df_items_export.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download CSV",
                        data=csv_items,
                        file_name=f"line_items_detailed_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv",
                        key='csv_item'
                    )
                    
                    # Excel Button
                    buffer_items = io.BytesIO()
                    with pd.ExcelWriter(buffer_items, engine='xlsxwriter') as writer:
                        df_items_export.to_excel(writer, sheet_name='Line Items', index=False)
                    
                    st.download_button(
                        label="Download Excel",
                        data=buffer_items.getvalue(),
                        file_name=f"line_items_detailed_{datetime.now().strftime('%Y%m%d')}.xlsx",
                        mime="application/vnd.ms-excel",
                        key='xls_item'
                    )
                else:
                    st.info("No line items available to export.")

        st.divider()

        df_history = get_all_receipts()
        
        col_search, col_rest = st.columns([1, 3])
        with col_search:
            search_id = st.number_input("🔍 Search by Receipt ID", min_value=0, step=1, value=0)
            if search_id > 0:
                receipt_match = get_receipt_by_id(search_id)
                if not receipt_match.empty:
                    st.session_state['view_receipt_id'] = search_id
                else:
                    st.error(f"ID {search_id} not found.")
        
        if not df_history.empty:
            col_list, col_detail = st.columns([1, 2])
            with col_list:
                st.subheader("Receipt List")
                df_history['label'] = df_history.apply(lambda x: f"ID: {x['id']} - {x['merchant']} (${x['total_amount']})", axis=1)
                
                default_index = 0
                if st.session_state['view_receipt_id']:
                     match_idx = df_history.index[df_history['id'] == st.session_state['view_receipt_id']].tolist()
                     if match_idx:
                         default_index = match_idx[0]

                try:
                    selected_label = st.selectbox("Select Receipt:", df_history['label'], index=default_index)
                except:
                    selected_label = st.selectbox("Select Receipt:", df_history['label'], index=0)

                selected_id = int(selected_label.split(" - ")[0].replace("ID: ", ""))
                st.session_state['view_receipt_id'] = selected_id
                
                st.divider()
                st.markdown("### Delete Bill")
                if st.button(f"Delete Bill ID: {selected_id}", type="primary"):
                    delete_receipt(selected_id)
                    st.toast(f"Receipt {selected_id} deleted!", icon="🗑️")
                    st.rerun()

            with col_detail:
                st.subheader(f"Details for ID: {selected_id}")
                selected_row = df_history[df_history['id'] == selected_id].iloc[0]
                with st.container(border=True):
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Merchant", selected_row['merchant'])
                    c2.metric("Date", selected_row['date'])
                    c3.metric("Total Amount", f"${selected_row['total_amount']:.2f}")
                    st.markdown(f"**Invoice #:** {selected_row['invoice_number']}")
                    st.markdown(f"**Uploaded:** {selected_row['upload_timestamp']}")

                st.subheader("🛒 Line Items")
                line_items_df = get_line_items(selected_id)
                if not line_items_df.empty:
                    st.dataframe(line_items_df, use_container_width=True, hide_index=True)
                else:
                    st.info("No detailed line items found for this receipt.")
        else:
            st.info("No receipts found in the database.")

    # === TAB 4: ANALYTICS ===
    with tab_analytics:
        st.header("📊 Spending Analytics")
        df = get_all_receipts()
        
        if not df.empty:
            # Data Pre-processing
            df['total_amount'] = pd.to_numeric(df['total_amount'], errors='coerce').fillna(0)
            df['tax'] = pd.to_numeric(df['tax'], errors='coerce').fillna(0)
            df['date_obj'] = pd.to_datetime(df['date'], errors='coerce')
            df_clean = df.dropna(subset=['date_obj']).copy().sort_values('date_obj')

            # --- KPI ROW ---
            kpi1, kpi2, kpi3, kpi4 = st.columns(4)
            total_spend = df['total_amount'].sum()
            avg_ticket = df['total_amount'].mean()
            total_tax = df['tax'].sum()
            recent_date = df_clean['date_obj'].max().strftime('%b %d, %Y') if not df_clean.empty else "N/A"

            kpi1.metric("Total Spend", f"${total_spend:,.2f}")
            kpi2.metric("Avg Receipt", f"${avg_ticket:,.2f}")
            kpi3.metric("Total Tax Paid", f"${total_tax:,.2f}")
            kpi4.metric("Last Purchase", recent_date)
            
            st.divider()

            # --- ROW 1: MERCHANT ANALYSIS ---
            st.subheader("🏢 Merchant Breakdown")
            col_a, col_b = st.columns(2)
            
            with col_a:
                fig_bar = px.bar(df, x='merchant', y='total_amount', color='merchant', 
                             title="Total Spend per Vendor", 
                             text_auto='.2s')
                st.plotly_chart(fig_bar, use_container_width=True)
            
            with col_b:
                fig_pie = px.pie(df, values='total_amount', names='merchant', 
                             title="Share of Wallet (Spend %)", 
                             hole=0.4)
                st.plotly_chart(fig_pie, use_container_width=True)

            # --- ROW 2: TIME SERIES ---
            st.subheader("📅 Time Trends")
            if not df_clean.empty:
                col_c, col_d = st.columns(2)
                
                with col_c:
                    # Monthly Trend
                    df_clean['month_year'] = df_clean['date_obj'].dt.strftime('%Y-%m')
                    monthly_spend = df_clean.groupby('month_year')['total_amount'].sum().reset_index()
                    fig_line = px.line(monthly_spend, x='month_year', y='total_amount', markers=True, 
                                   title="Monthly Spending Trend",
                                   labels={'month_year': 'Month', 'total_amount': 'Amount ($)'})
                    st.plotly_chart(fig_line, use_container_width=True)
                
                with col_d:
                    # Cumulative Spend
                    df_clean['cumulative'] = df_clean['total_amount'].cumsum()
                    fig_area = px.area(df_clean, x='date_obj', y='cumulative', 
                                   title="Cumulative Spending Over Time",
                                   labels={'date_obj': 'Date', 'cumulative': 'Running Total ($)'})
                    st.plotly_chart(fig_area, use_container_width=True)

            # --- ROW 3: BEHAVIORAL ---
            st.subheader("🧠 Spending Behavior")
            col_e, col_f = st.columns(2)

            with col_e:
                # Day of Week Analysis
                if not df_clean.empty:
                    df_clean['day_name'] = df_clean['date_obj'].dt.day_name()
                    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                    df_clean['day_name'] = pd.Categorical(df_clean['day_name'], categories=days_order, ordered=True)
                    day_counts = df_clean.groupby('day_name')['total_amount'].sum().reset_index()
                    
                    fig_day = px.bar(day_counts, x='day_name', y='total_amount', 
                                 title="Spending by Day of Week",
                                 color='total_amount', color_continuous_scale='Viridis')
                    st.plotly_chart(fig_day, use_container_width=True)

            with col_f:
                # Distribution of costs
                fig_hist = px.histogram(df, x='total_amount', nbins=20, 
                                    title="Distribution of Receipt Amounts",
                                    labels={'total_amount': 'Receipt Value ($)'})
                st.plotly_chart(fig_hist, use_container_width=True)

            # --- ROW 4: ITEM LEVEL ---
            st.subheader("🛒 Item Analysis")
            df_items = get_all_line_items_global()
            if not df_items.empty:
                # Top Items by Frequency
                top_items = df_items['name'].value_counts().head(10).reset_index()
                top_items.columns = ['Item Name', 'Count']
                
                fig_items = px.bar(top_items, x='Count', y='Item Name', orientation='h',
                               title="Top 10 Most Frequent Items",
                               color='Count')
                fig_items.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig_items, use_container_width=True)
            else:
                st.info("No detailed line item data available yet.")

        else:
            st.info("No data in vault yet. Upload some receipts to see analytics!")

if __name__ == "__main__":
    main()