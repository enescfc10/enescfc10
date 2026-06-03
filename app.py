import streamlit as st
import numpy as np
import time
from PIL import Image
import os

try:
    import tensorflow as tf
    from tensorflow.keras.preprocessing import image
    HAS_TF = True
except ImportError:
    HAS_TF = False

# --- SAYFA YAPILANDIRMASI VE TASARIM ---
st.set_page_config(
    page_title="Yapay Zeka ile Atık Sınıflandırma",
    page_icon="♻️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Gelişmiş CSS tasarımı
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    h1 {
        color: #4CAF50 !important;
        text-align: center;
        font-weight: 800;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.4);
    }
    .st-emotion-cache-1y4p8pa {
        padding-top: 2rem;
    }
    .upload-text {
        text-align: center;
        color: #ddd;
        font-size: 18px;
    }
    .result-card {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        border-radius: 15px;
        padding: 25px;
        text-align: center;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        border: 1px solid rgba(255, 255, 255, 0.18);
        color: white;
        margin-top: 20px;
        animation: fadeIn 1s;
    }
    .result-class {
        font-size: 36px;
        font-weight: bold;
        color: #00FF7F;
        margin: 10px 0;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    .result-confidence {
        font-size: 20px;
        color: #f1f1f1;
    }
    @keyframes fadeIn {
        0% { opacity: 0; transform: translateY(20px); }
        100% { opacity: 1; transform: translateY(0); }
    }
    .footer {
        text-align: center;
        margin-top: 50px;
        color: #888;
        font-size: 14px;
    }
</style>
""", unsafe_allow_html=True)


# --- FONKSİYONLAR VE MODEL YETENEKLERİ ---
@st.cache_resource
def load_classification_model():
    if not HAS_TF:
        return None
    # Modeli önbelleğe (cache) alarak her yüklemede beklemenin önüne geçiyoruz
    try:
        model_path = 'trash_model.h5'
        if os.path.exists(model_path):
            return tf.keras.models.load_model(model_path)
        else:
            return None
    except Exception as e:
        return None

def predict_rubbish(img, model):
    if model is None or not HAS_TF:
        # Eğer model yüklenmemişse temsilci (mock) bir veri dön
        # Sınıflar: Kagit, Plastik, Cam, Metal vs (Notebook'tan alınan veri yapısına uygun)
        classes = ['Cam', 'Kağıt', 'Karton', 'Metal', 'Organik', 'Plastik']
        import random
        return random.choice(classes), round(random.uniform(70.0, 99.9), 2)
    
    # Gerçek model için önişleme
    img_resized = img.resize((224, 224))
    img_array = image.img_to_array(img_resized)
    img_array = tf.expand_dims(img_array, 0)
    
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    
    # Notebook'taki varsayılan 6 sınıf sırası. Eğer değişmişse buradan güncellenmeli.
    classes = ['Cam', 'Kağıt', 'Karton', 'Metal', 'Organik', 'Plastik']
    
    predicted_class = classes[np.argmax(score)]
    confidence = 100 * np.max(score)
    return predicted_class, round(float(confidence), 2)


# --- ANA UYGULAMA ---

st.markdown("<h1>♻️ Akıllı Atık Sınıflandırma Sistemi</h1>", unsafe_allow_html=True)
st.markdown("<p class='upload-text'>Yapay Zeka ile çöplerinizi otomatik tanıyın, doğayı korumaya katkı sağlayın!</p>", unsafe_allow_html=True)

model = load_classification_model()

if model is None:
    st.warning("⚠️ 'trash_model.h5' dosyası dizinde bulunamadı! Şu anda demo (temsili) sonuçlar gösterilmektedir. Jupyter notebook üzerinden model_v2.save('trash_model.h5') metoduyla gerçek modelinizi kaydedip buraya kopyalayabilirsiniz.")

# Dosya yükleme aracı
uploaded_file = st.file_uploader("Atık fotoğrafı yükleyin (JPG, PNG)", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Resmi Göster
    col1, col2 = st.columns([1,1])
    
    with col1:
        st.markdown("### Yüklenen Fotoğraf")
        image_to_process = Image.open(uploaded_file)
        # Görüntüyü düzgün boylama
        st.image(image_to_process, use_column_width=True)
    
    with col2:
        st.markdown("### Model Analizi")
        # Analiz süreci animasyonu
        with st.spinner('Yapay zeka resmi inceliyor... Lütfen bekleyin.'):
            time.sleep(1.5) # Gerçekçilik ve animasyonun görünmesi için kısa gecikme
            
            # Tahmin fonksiyonunu çağır
            predicted_class, confidence = predict_rubbish(image_to_process, model)
        
        # Sonucu Estetik Kart İle Yazdır
        st.markdown(f"""
        <div class="result-card">
            <div>Atık Türü Tespit Edildi:</div>
            <div class="result-class">{predicted_class}</div>
            <div class="result-confidence">Güven Oranı: <b>%{confidence}</b> 🎯</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Atık Türüne Göre Renkli Bildirimler Müzakeresi
        if predicted_class.lower() == 'plastik':
            st.info("💡 Plastikler geri dönüşüm kutusuna atılmalıdır. Doğada kaybolması yüzlerce yıl sürer.")
        elif predicted_class.lower() == 'cam':
            st.success("💡 Cam, kalitesini kaybetmeden %100 geri dönüştürülebilir!")
        elif predicted_class.lower() == 'organik':
            st.warning("💡 Organik atıklarınızı kompost (doğal gübre) yapımında kullanabilirsiniz.")
        elif predicted_class.lower() == 'kağıt' or predicted_class.lower() == 'karton':
            st.info("💡 Kağıt ve karton geri dönüştürülerek ağaç kesilmesinin önüne geçilir.")
            

st.markdown("<div class='footer'>Geliştirici: Enes Öner | Gücünü TensorFlow ve Streamlit'ten Alır.</div>", unsafe_allow_html=True)
