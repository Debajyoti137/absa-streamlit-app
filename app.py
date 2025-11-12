# Save this code as app.py
import streamlit as st
from pyabsa import AspectTermExtraction as ATEPC

# -----------------------------------------------------------------
# Page Configuration
# -----------------------------------------------------------------
st.set_page_config(
    page_title="ABSA E-commerce Demo",
    page_icon="🛍️",
    layout="wide"
)

# -----------------------------------------------------------------
# Model Loading
# -----------------------------------------------------------------
# Use st.cache_resource to load the model only once.
# This is crucial for performance in a deployed app.
@st.cache_resource
def load_model():
    """
    Loads the pre-trained ABSA model.
    We use 'multilingual' as a robust, general-purpose model.
    """
    st.write("Loading ABSA model... This may take a moment on first run.")
    
    # 'auto_device=True' will use a GPU if available (on Streamlit Cloud)
    model = ATEPC.AspectExtractor('multilingual', auto_device=True)
    return model

classifier = load_model()

# -----------------------------------------------------------------
# Streamlit App UI (User Interface)
# -----------------------------------------------------------------
st.title("🛍️ Aspect-Based Sentiment Analysis for E-commerce")
st.markdown("""
This app analyzes customer reviews to identify sentiments (Positive, Negative, Neutral) 
for specific product aspects (like 'battery life' or 'camera quality').
""")

# --- Instructions Expander ---
with st.expander("ℹ️ How to use this app"):
    st.write("""
    1.  **Enter a review:** Type or paste a product review into the text box below.
    2.  **Click 'Analyze':** The app will process the text.
    3.  **View results:** See the extracted aspects and their corresponding sentiments in the table.
    
    **Example reviews:**
    * `The battery life is amazing, but the camera is a bit disappointing.`
    * `I love the screen resolution, but the speakers are too quiet.`
    * `Great performance and sleek design. The keyboard feels nice too.`
    """)

# --- Main App ---
st.subheader("Analyze a Review")

# Default text for the user
default_review = "The battery life is amazing, but the camera is a bit disappointing."

# Text area for user input
review_text = st.text_area(
    "Enter your review:",
    default_review,
    height=120,
    placeholder="e.g., The screen is great but the battery dies quickly."
)

# Analyze button
if st.button("Analyze Sentiment", type="primary"):
    if not review_text.strip():
        st.warning("Please enter a review to analyze.")
    else:
        st.subheader("Analysis Results:")
        
        # Run the model
        with st.spinner("🧠 Analyzing aspect sentiments..."):
            
            # The 'predict' function takes a list of texts
            result = classifier.predict(
                texts=[review_text],
                save_result=False,
                print_result=False
            )
            
            # The model returns a list of results; we take the first one
            analysis = result[0]
            
            # Prepare data for display
            display_data = []
            if not analysis['aspect']:
                st.warning("No specific aspects were detected in this review.")
            else:
                for i, aspect in enumerate(analysis['aspect']):
                    sentiment = analysis['sentiment'][i]
                    confidence = analysis['confidence'][i]
                    
                    # Add emojis for better visual feedback
                    if sentiment == "Positive":
                        sentiment_emoji = "😄 Positive"
                    elif sentiment == "Negative":
                        sentiment_emoji = "😞 Negative"
                    else:
                        sentiment_emoji = "😐 Neutral"
                    
                    display_data.append({
                        "Aspect": aspect,
                        "Sentiment": sentiment_emoji,
                        "Confidence": f"{confidence:.2%}" # Format as percentage
                    })
                
                # Display results in a clean table
                st.table(display_data)

                # Optional: Show the raw JSON output
                with st.expander("Show Raw Model Output (JSON)"):
                    st.json(analysis)