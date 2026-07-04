import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download

repo_id = "hessamedin/fuel-consumption-predictor"

model_city = joblib.load(hf_hub_download(repo_id, "city_model.pkl"))
model_highway = joblib.load(hf_hub_download(repo_id, "highway_model.pkl"))
model_combined = joblib.load(hf_hub_download(repo_id, "combined_model.pkl"))
df = pd.read_csv(hf_hub_download(repo_id, "Fuel_Consumption_Cleaned.csv"))

features_to_keep = ['VEHICLE CLASS', 'ENGINE SIZE', 'CYLINDERS', 'TRANSMISSION', 'FUEL']
X = df[features_to_keep]
X_encoded = pd.get_dummies(X, columns=['VEHICLE CLASS', 'TRANSMISSION', 'FUEL'], drop_first=True)
train_columns = X_encoded.columns

iranian_cars_db = {
    "پراید (111/131/132)": {"ENGINE SIZE": 1.3, "CYLINDERS": 4, "TRANSMISSION": "M5", "FUEL": "X", "VEHICLE CLASS": "SUBCOMPACT", "desc": "خودروی اقتصادی و کم‌مصرف شهری"},
    "تیبا (هاچ‌بک/صندوق‌دار)": {"ENGINE SIZE": 1.5, "CYLINDERS": 4, "TRANSMISSION": "M5", "FUEL": "X", "VEHICLE CLASS": "SUBCOMPACT", "desc": "نسخه ارتقا یافته پراید با موتور قوی‌تر"},
    "پژو 206 (تیپ 2 - 1.4 لیتری)": {"ENGINE SIZE": 1.4, "CYLINDERS": 4, "TRANSMISSION": "M5", "FUEL": "X", "VEHICLE CLASS": "SUBCOMPACT", "desc": "خودروی هاچ‌بک محبوب شهری"},
    "پژو 206 (تیپ 5 - 1.6 لیتری)": {"ENGINE SIZE": 1.6, "CYLINDERS": 4, "TRANSMISSION": "M5", "FUEL": "X", "VEHICLE CLASS": "SUBCOMPACT", "desc": "نسخه فول‌آپشن 206 با موتور TU5"},
    "پژو 206 SD (V8)": {"ENGINE SIZE": 1.6, "CYLINDERS": 4, "TRANSMISSION": "A4", "FUEL": "X", "VEHICLE CLASS": "COMPACT", "desc": "نسخه صندوق‌دار 206 با گیربکس اتوماتیک"},
    "پژو پارس (موتور XU7 - 1.8 لیتری)": {"ENGINE SIZE": 1.8, "CYLINDERS": 4, "TRANSMISSION": "M5", "FUEL": "X", "VEHICLE CLASS": "MID-SIZE", "desc": "سدان خانوادگی پرفروش با موتور قدیمی اما جان‌سخت"},
    "پژو پارس (موتور TU5 - 1.6 لیتری)": {"ENGINE SIZE": 1.6, "CYLINDERS": 4, "TRANSMISSION": "M5", "FUEL": "X", "VEHICLE CLASS": "MID-SIZE", "desc": "نسخه کم‌مصرف‌تر پژو پارس"},
    "سمند (LX/SE - 1.8 لیتری)": {"ENGINE SIZE": 1.8, "CYLINDERS": 4, "TRANSMISSION": "M5", "FUEL": "X", "VEHICLE CLASS": "MID-SIZE", "desc": "سدان ملی ایران با فضای کابین جادار"},
    "دنا پلاس (تنفس طبیعی)": {"ENGINE SIZE": 1.6, "CYLINDERS": 4, "TRANSMISSION": "A6", "FUEL": "X", "VEHICLE CLASS": "MID-SIZE", "desc": "سدان مدرن ایران‌خودرو با گیربکس 6 سرعته اتوماتیک"},
    "دنا پلاس (توربو)": {"ENGINE SIZE": 1.7, "CYLINDERS": 4, "TRANSMISSION": "A6", "FUEL": "X", "VEHICLE CLASS": "MID-SIZE", "desc": "نسخه پرقدرت دنا پلاس با موتور توربوشارژ"},
    "تارا (دنده‌ای)": {"ENGINE SIZE": 1.6, "CYLINDERS": 4, "TRANSMISSION": "M6", "FUEL": "X", "VEHICLE CLASS": "COMPACT", "desc": "جدیدترین سدان ایران‌خودرو بر پایه پلتفرم پژو 301"},
    "تارا (اتوماتیک)": {"ENGINE SIZE": 1.6, "CYLINDERS": 4, "TRANSMISSION": "A6", "FUEL": "X", "VEHICLE CLASS": "COMPACT", "desc": "نسخه اتوماتیک تارا با گیربکس 6 سرعته"},
    "شاهین (توربو)": {"ENGINE SIZE": 1.5, "CYLINDERS": 4, "TRANSMISSION": "M5", "FUEL": "X", "VEHICLE CLASS": "COMPACT", "desc": "سدان اسپرت سایپا با موتور 1.5 لیتری توربو"}
}

def calculate_fuel_cost(fuel_needed, already_used):
    cost = 0
    remaining_fuel = fuel_needed
    current_quota = already_used
    
    if current_quota < 60 and remaining_fuel > 0:
        tier1_available = 60 - current_quota
        tier1_used = min(remaining_fuel, tier1_available)
        cost += tier1_used * 1500
        remaining_fuel -= tier1_used
        current_quota += tier1_used
        
    if current_quota < 130 and remaining_fuel > 0:
        tier2_available = 130 - current_quota
        tier2_used = min(remaining_fuel, tier2_available)
        cost += tier2_used * 3000
        remaining_fuel -= tier2_used
        current_quota += tier2_used
        
    if remaining_fuel > 0:
        cost += remaining_fuel * 5000
        
    return cost

st.title('Vehicle Fuel Consumption Predictor')
st.write('Predict fuel consumption (L/100 km) using global data or popular Iranian car specs.')

mode = st.radio(
    "Select your preferred method / روش انتخاب خود را مشخص کنید:",
    ["🇮 خودروهای ایرانی", "Global / Manual Selection"]
)

st.markdown("---")

driving_mode = st.selectbox(
    "Driving Condition / نوع رانندگی",
    ["City (شهری)", "Highway (جاده‌ای)", "Combined (ترکیبی)"]
)

if driving_mode == "️ City (شهری)":
    current_model = model_city
elif driving_mode == "Highway (جاده‌ای)":
    current_model = model_highway
else:
    current_model = model_combined

def predict_fuel(input_data, model):
    input_encoded = pd.get_dummies(input_data, columns=['VEHICLE CLASS', 'TRANSMISSION', 'FUEL'], drop_first=True)
    input_encoded = input_encoded.reindex(columns=train_columns, fill_value=0)
    return model.predict(input_encoded)[0]

if mode == "🇮🇷 خودروهای ایرانی":
    st.markdown("<h3 style='text-align: right; direction: rtl;'>انتخاب خودروی ایرانی</h3>", unsafe_allow_html=True)
    
    selected_car = st.selectbox("مدل خودرو را انتخاب کنید", list(iranian_cars_db.keys()))
    st.markdown(f"<p style='text-align: right; direction: rtl; color: gray;'>{iranian_cars_db[selected_car]['desc']}</p>", unsafe_allow_html=True)
    
    distance_km = st.number_input('مسافت سفر را وارد کنید (کیلومتر)', min_value=0.0, value=100.0, step=10.0)
    
    already_used = st.number_input('میزان سوخت مصرف شده از سهمیه ماهانه (لیتر) / Fuel already used this month (L)', 
                                   min_value=0.0, value=0.0, step=10.0, 
                                   help="اگر از اول ماه تا الان سوخت زده‌اید، مقدار آن را وارد کنید. در غیر این صورت 0 بگذارید.")
    
    if st.button("محاسبه مصرف سوخت و هزینه"):
        specs = iranian_cars_db[selected_car]
        
        input_data = pd.DataFrame({
            'VEHICLE CLASS': [specs['VEHICLE CLASS']],
            'ENGINE SIZE': [specs['ENGINE SIZE']],
            'CYLINDERS': [specs['CYLINDERS']],
            'TRANSMISSION': [specs['TRANSMISSION']],
            'FUEL': [specs['FUEL']]
        })
        
        prediction = predict_fuel(input_data, current_model)
        total_fuel = round((prediction / 100) * distance_km, 2)
        total_cost = calculate_fuel_cost(total_fuel, already_used)
        
        st.success(f'### مصرف سوخت پیش‌بینی شده برای {selected_car}: {prediction:.2f} لیتر در هر ۱۰۰ کیلومتر')
        st.info(f'⛽ برای مسافت **{distance_km} کیلومتر**، این خودرو حدود **{total_fuel:.2f} لیتر** بنزین مصرف خواهد کرد.\n💰 **هزینه تقریبی سوخت برای این سفر:** **{total_cost:,.0f} تومان**')
        
        with st.expander("🔍 مشاهده مشخصات فنی استفاده شده در این پیش‌بینی"):
            st.write(f"**حجم موتور:** {specs['ENGINE SIZE']} لیتر")
            st.write(f"**تعداد سیلندر:** {specs['CYLINDERS']}")
            st.write(f"**نوع گیربکس:** {specs['TRANSMISSION']}")
            st.write(f"**نوع سوخت:** {specs['FUEL']} (بنزین معمولی)")
            st.write(f"**کلاس جهانی معادل:** {specs['VEHICLE CLASS']}")

else:
    st.subheader("Enter Vehicle Specifications Manually")
    
    vehicle_class = st.selectbox('Vehicle Class', sorted(df['VEHICLE CLASS'].unique()))
    engine_size = st.slider('Engine Size (L)', float(df['ENGINE SIZE'].min()), float(df['ENGINE SIZE'].max()), float(df['ENGINE SIZE'].mean()))
    cylinders = st.slider('Cylinders', int(df['CYLINDERS'].min()), int(df['CYLINDERS'].max()), int(df['CYLINDERS'].mode()[0]))
    
    transmission_help = """
    **Transmission Codes Explained:**
    - **A / AS**: Automatic (e.g., A4 = 4-speed Auto, AS6 = 6-speed Auto)
    - **AM**: Automated Manual (e.g., AM6, AM7)
    - **AV**: Automatic with Variable ratio / CVT
    - **M**: Manual (e.g., M5 = 5-speed Manual, M6 = 6-speed Manual)
    """
    transmission = st.selectbox('Transmission', sorted(df['TRANSMISSION'].unique()), help=transmission_help)
    
    fuel_help = """
    **Fuel Types Explained:**
    - **X**: Regular Gasoline
    - **Z**: Premium Gasoline
    - **E**: Ethanol (E85 Flex Fuel)
    - **D**: Diesel
    - **N**: Natural Gas (CNG)
    """
    fuel = st.selectbox('Fuel Type', sorted(df['FUEL'].unique()), help=fuel_help)
    
    distance_km = st.number_input('Travel Distance (km)', min_value=0.0, value=100.0, step=10.0)
    
    if st.button('Predict Fuel Consumption'):
        input_data = pd.DataFrame({
            'VEHICLE CLASS': [vehicle_class],
            'ENGINE SIZE': [engine_size],
            'CYLINDERS': [cylinders],
            'TRANSMISSION': [transmission],
            'FUEL': [fuel]
        })
        
        prediction = predict_fuel(input_data, current_model)
        total_fuel = round((prediction / 100) * distance_km, 2)
        
        st.success(f'### Predicted Fuel Consumption: {prediction:.2f} L/100 km')
        st.info(f' For a distance of **{distance_km} km**, this vehicle will consume approximately **{total_fuel:.2f} Liters** of fuel.')

st.markdown("---")
st.caption("**Note:** Predictions are estimates based on mechanical equivalents in the global dataset. Fuel costs (for Iranian cars) are calculated based on Iran's current 3-tier smart card pricing (1500T/3000T/5000T). / پیش‌بینی‌ها بر اساس معادل‌های جهانی هستند و هزینه سوخت (برای خودروهای ایرانی) بر اساس قانون سهمیه‌بندی سه نرخی کارت سوخت محاسبه می‌شود.")