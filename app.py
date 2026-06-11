import streamlit as st
import joblib
import sys
import nltk
import string
import os
import gdown  
from nltk.corpus import stopwords
import yfinance as yf 
FILE_IDS = {
    'stock_predictor_model.pkl': '1-q9Z2PBNRhTdo3zlbMsfUoaQw7DHtl3h',
    'bag_of_words_transformer.pkl': '1vGHBQSB7IRT8xzIdBRXYiUz6NwTsVc3z',  
    'tfidf_weight_transformer.pkl': '1W2STwnw_M_IpF00HwAtaEW0ziEcj9ZM3',  
    'ticker_label_encoder.pkl': '1FvTI7xZ_AgBT-a895G6kHXdri7gw-oGI'
}

@st.cache_resource
def download_pipeline_assets():
    for filename, file_id in FILE_IDS.items():
        if not os.path.exists(filename):
            with st.spinner(f"Downloading {filename} from secure cloud storage..."):
                url = f'https://drive.google.com/uc?id={file_id}'
                gdown.download(url, filename, quiet=False)
@st.cache_data
def get_company_name(ticker):
    try:
        stock_info = yf.Ticker(ticker).info
        return stock_info.get('longName', stock_info.get('shortName', "Market Asset Class"))
    except Exception:
        return "Registered Market Asset Class"

try:
    english_stopwords = set(stopwords.words('english'))
except LookupError:
    import nltk
    nltk.download('stopwords')
    english_stopwords = set(stopwords.words('english'))

def fn1(mess):
    clean = [char.lower() for char in mess if char not in string.punctuation and char not in "1234567890"]
    clean = ''.join(clean)
    
    return [word for word in clean.split() if word.lower() not in english_stopwords]
    

sys.modules['__main__'].fn1 = fn1


st.set_page_config(page_title="NarrativeTracker Engine", layout="wide")


download_pipeline_assets()
def load_pipeline_components():
    model = joblib.load('stock_predictor_model.pkl')
    b_transformer = joblib.load('bag_of_words_transformer.pkl')
    tfidf_transformer = joblib.load('tfidf_weight_transformer.pkl')
    label_encoder = joblib.load('ticker_label_encoder.pkl')
    return model, b_transformer, tfidf_transformer, label_encoder

try:
    model, b_transformer, tfidf_transformer, label_encoder = load_pipeline_components()
except FileNotFoundError:
    st.error("Model files (.pkl) not found. Make sure to run your Jupyter export cell first!")


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
    
    user_headline = st.text_input(
        "Type or paste a financial headline below:",
        placeholder="Paste a market narrative here..."
    )
    
    if st.button("Run Bayesian Prediction Scan"):
        if user_headline.strip() == "":
            st.warning("Please enter a valid headline text to analyze.")
        else:
            with st.spinner("Processing token distributions..."):
                bow_vector = b_transformer.transform([user_headline])
                tfidf_vector = tfidf_transformer.transform(bow_vector)
                
                predicted_encoded = model.predict(tfidf_vector)
                predicted_stock = label_encoder.inverse_transform(predicted_encoded)[0]
                
                st.success("###  Analysis Complete")

                with st.spinner("Fetching corporate asset profile..."):
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
                st.caption(
                    "The framework maps these token weights against the prior distribution probabilities "
                    "established across your dataset rows to resolve the target mapping."
                )