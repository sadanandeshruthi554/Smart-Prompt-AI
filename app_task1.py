import streamlit as st
import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()

st.set_page_config(
    page_title="Review Analyzer",
    page_icon="📊",
    layout="centered"
)

st.title("Reviews Classify by Gemini LLM")

# CHECK API KEY
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    st.error("GEMINI_API_KEY not found in .env file")
    st.stop()

# CONFIGURE GEMINI
genai.configure(api_key=api_key)

# CREATE MODEL
model = genai.GenerativeModel("gemini-2.5-flash-lite")

# SIDEBAR
st.sidebar.title("Settings")

temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.5)
max_tokens = st.sidebar.slider("Max Tokens", 1, 2048, 512)

# SYSTEM PROMPT
system_prompt = """
You are an expert sentiment analysis assistant.

Your task is to carefully analyze the given review step-by-step using Chain of Thought reasoning.

Follow these steps internally:

1. Identify all positive sentiment phrases.
2. Identify all negative sentiment phrases.
3. Detect whether the review contains mixed or contradictory opinions.
4. Analyze the overall emotional tone.
5. Decide the final sentiment label.

Sentiment labels:
- Positive
- Negative
- Neutral

Rules:
- If the review mainly contains praise, satisfaction, or happy opinions → Positive
- If the review mainly contains complaints, disappointment, or negative opinions → Negative
- If the review contains both positive and negative opinions OR unclear sentiment → Neutral

Final Sentiment:
Positive / Negative / Neutral

Reason:
Explain clearly why this label was chosen based on the review content.
"""

# INPUT
user_input = st.text_area("Enter Review")

# GENERATE BUTTON
if st.button("Generate Classification"):

    if user_input.strip() == "":
        st.warning("Please enter a review")

    else:
        with st.spinner("Analyzing Review..."):

            response = model.generate_content(
                f"{system_prompt}\n\nReview:\n{user_input}",
                generation_config=genai.types.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens
                )
            )

            result = response.text

        st.subheader("LLM Response")
        st.write(result)