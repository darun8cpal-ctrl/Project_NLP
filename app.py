import streamlit as st
import joblib
import sys
import nltk
import string
import yfinance as yf 
from nltk.corpus import stopwords


try:
    english_stopwords = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords')
    english_stopwords = set(stopwords.words('english'))

def fn1(mess):
    clean = [char.lower() for char in mess if char not in string.punctuation and char not in "1234567890"]
    clean = ''.join(clean)
    return [word for word in clean.split() if word.lower() not in english_stopwords]

sys.modules['__main__'].fn1 = fn1


st.set_page_config(page_title="NarrativeTracker Engine", layout="wide")

@st.cache_data
def get_company_name(ticker):
    clean_ticker = ticker.replace('$', '').strip().upper()
    
    demo_map = {
        "AAPL": "Apple Inc.",
        "TSLA": "Tesla, Inc.",
        "MSFT": "Microsoft Corporation",
        "QCOM": "Qualcomm Incorporated",
        "BP": "BP p.l.c.",
        "GMCR": "Keurig Green Mountain"
    }
    
    if clean_ticker in demo_map:
        return demo_map[clean_ticker]
    
    try:
        ticker_obj = yf.Ticker(clean_ticker)

        info = ticker_obj.info
        name = info.get('longName') or info.get('shortName')
        return name if name else f"Asset: {clean_ticker}"
    except Exception:
        return f"Asset: {clean_ticker}"


@st.cache_resource
def load_pipeline_components():
    model = joblib.load('stock_predictor_model.pkl')
    b_transformer = joblib.load('bag_of_words_transformer.pkl')
    tfidf_transformer = joblib.load('tfidf_weight_transformer.pkl')
    label_encoder = joblib.load('ticker_label_encoder.pkl')
    return model, b_transformer, tfidf_transformer, label_encoder

try:
    model, b_transformer, tfidf_transformer, label_encoder = load_pipeline_components()
except Exception as e:
    st.error(f"Error loading model files: {e}. Please ensure the .pkl files are in the repository.")
    st.stop()


historical_headlines = {
    "🇺🇸 US Markets": [
        "iPhone sales surge unexpected in Asian markets during Q3",
        "Tesla delivery targets face supply chain bottlenecks in Berlin",
        "Microsoft cloud infrastructure investments drive record revenue gains"
    ],
    "🇬🇧 UK Markets": [
        "Central bank increases interest rates by 25 basis points to combat inflation",
        "BP plc profits climb amid shifting global energy demands",
        "AstraZeneca wins regulatory clearance for targeted oncology treatment"
    ],
    "🇯🇵 Asian Markets": [
        "New electric vehicle factory opens in Texas with automated assembly lines",
        "Sony updates production targets for next-generation display hardware",
        "SoftBank group increases venture allocations toward regional AI startups"
    ]
}

st.title("Stock Head Tracker: Your Ultimate Guide for Stock and Risk Prediction")
st.caption("Empirical Bayesian Text-Quantification Engine | 73% Macro Precision Framework")
st.markdown("---")

sidebar, main_panel = st.columns([1, 2])

with sidebar:
    st.header("Global Market Feed")
    region = st.selectbox("Choose a Market Region:", list(historical_headlines.keys()))
    st.subheader("Sample Headlines")
    st.write("Click and copy any headline below to test the engine:")
    for headline in historical_headlines[region]:
        st.info(f" {headline}")

with main_panel:
    st.header("Real-Time Inference Scan")
    user_headline = st.text_input("Type or paste a financial headline below:", placeholder="Paste a market narrative here...")
    
    if st.button("Run Bayesian Prediction Scan"):
        if user_headline.strip() == "":
            st.warning("Please enter a valid headline text to analyze.")
        else:
            with st.spinner("Processing token distributions..."):
                bow_vector = b_transformer.transform([user_headline])
                tfidf_vector = tfidf_transformer.transform(bow_vector)
                predicted_encoded = model.predict(tfidf_vector)
                predicted_stock = label_encoder.inverse_transform(predicted_encoded)[0]
                
                st.success("### Analysis Complete")
                company_full_name = get_company_name(predicted_stock)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(label="Predicted Asset Focus", value=f"${predicted_stock}")
                with col2:
                    st.metric(label="Framework Baseline", value="Multinomial NB")

                st.markdown("---")
                st.markdown(f"""
                <div style="background-color:#f0f2f6; padding:18px; border-radius:10px; border-left: 6px solid #ff4b4b;">
                    <span style="color:#555555; text-transform: uppercase; font-size: 0.8rem; font-weight: bold; letter-spacing: 1px;">Corporate Asset Profile</span>
                    <h3 style="margin: 5px 0 0 0; color:#111111; font-family: sans-serif;">{company_full_name}</h3>
                </div>
                """, unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("#### **Linguistic Metrics**")
                st.write(f"**Extracted Features:** `{fn1(user_headline)}`")