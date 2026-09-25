%%writefile app.py

import io

import streamlit as st
from groq import Groq
from huggingface_hub import InferenceClient


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="ASCEND AI Studio",
    page_icon="🧬",
    layout="wide"
)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_ID = "black-forest-labs/FLUX.1-schnell"
GROQ_MODEL = "openai/gpt-oss-20b"


# ============================================================
# LOAD API KEYS FROM STREAMLIT SECRETS
# ============================================================

try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
    HF_TOKEN = st.secrets["HF_TOKEN"]

except Exception:
    st.error(
        "API keys are not configured. "
        "Please add GROQ_API_KEY and HF_TOKEN "
        "to your Streamlit Secrets."
    )
    st.stop()


# ============================================================
# API CLIENTS
# ============================================================

groq_client = Groq(
    api_key=GROQ_API_KEY
)

hf_client = InferenceClient(
    api_key=HF_TOKEN
)


# ============================================================
# PROMPT ENHANCEMENT
# ============================================================

def enhance_prompt(user_prompt, visual_style):

    system_prompt = """
You are the professional prompt enhancement assistant for
ASCEND AI Studio.

ASCEND stands for:
Advancing Sciences, Creative, Engineering and New Discoveries.

Your job is to transform a user's idea into a professional
prompt suitable for an AI image-generation model.

Rules:

1. Preserve the user's original idea and intended meaning.
2. If the user's prompt is already detailed and professional,
   make only useful improvements.
3. If the user's prompt is vague, add relevant visual details.
4. Consider subject, environment, composition, lighting,
   perspective and visual quality.
5. When the topic is scientific, use appropriate scientific
   visual terminology.
6. Do not invent specific scientific facts.
7. Do not add unnecessary slogans, logos, or written text
   inside the image.
8. The selected visual style should influence the final prompt.
9. Do not explain your changes.
10. Return ONLY the final image-generation prompt.

Selected visual style:
"""

    user_message = f"""
Visual style:
{visual_style}

User's idea:
{user_prompt}
"""

    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_message
            }
        ],
        temperature=0.7,
        max_tokens=500
    )

    return response.choices[0].message.content.strip()


# ============================================================
# IMAGE GENERATION
# ============================================================

def generate_image(prompt, width, height):

    image = hf_client.text_to_image(
        prompt=prompt,
        model=MODEL_ID,
        width=width,
        height=height
    )

    return image


# ============================================================
# HEADER
# ============================================================

st.title("🧬 ASCEND AI Studio")

st.subheader(
    "Advancing Sciences, Creative, Engineering and New Discoveries"
)

st.write(
    "Transform your ideas into professional AI-generated "
    "visuals for science, education, events and social media."
)

st.divider()


# ============================================================
# USER INPUT
# ============================================================

st.header("Create Your Visual")

user_prompt = st.text_area(
    "What would you like to create?",
    placeholder=(
        "Example: Create a professional visual for a "
        "biotechnology seminar about CRISPR gene editing."
    ),
    height=150
)


# ============================================================
# SETTINGS
# ============================================================

col1, col2 = st.columns(2)


with col1:

    visual_style = st.selectbox(
        "Visual Style",
        [
            "Professional Scientific",
            "Realistic",
            "Scientific Illustration",
            "Minimalist",
            "Futuristic",
            "Educational",
            "Cinematic"
        ]
    )


with col2:

    aspect_ratio = st.selectbox(
        "Aspect Ratio",
        [
            "1:1",
            "16:9",
            "9:16"
        ]
    )


# ============================================================
# GENERATE BUTTON
# ============================================================

generate_button = st.button(
    "✨ Generate Image",
    type="primary",
    use_container_width=True
)


# ============================================================
# GENERATION PROCESS
# ============================================================

if generate_button:

    if not user_prompt.strip():

        st.warning(
            "Please enter an idea before generating an image."
        )

    else:

        try:

            # ------------------------------------------------
            # STEP 1 — GROQ
            # ------------------------------------------------

            with st.spinner(
                "Improving your prompt..."
            ):

                improved_prompt = enhance_prompt(
                    user_prompt,
                    visual_style
                )


            st.subheader("Improved Prompt")

            st.info(improved_prompt)


            # ------------------------------------------------
            # STEP 2 — ASPECT RATIO
            # ------------------------------------------------

            if aspect_ratio == "1:1":

                width = 1024
                height = 1024

            elif aspect_ratio == "16:9":

                width = 1344
                height = 768

            else:

                width = 768
                height = 1344


            # ------------------------------------------------
            # STEP 3 — HUGGING FACE
            # ------------------------------------------------

            with st.spinner(
                "Generating your image..."
            ):

                image = generate_image(
                    improved_prompt,
                    width,
                    height
                )


            # ------------------------------------------------
            # STEP 4 — DISPLAY
            # ------------------------------------------------

            st.success(
                "Image generated successfully!"
            )

            st.subheader("Generated Image")

            st.image(
                image,
                use_container_width=True
            )


            # ------------------------------------------------
            # STEP 5 — DOWNLOAD
            # ------------------------------------------------

            image_bytes = io.BytesIO()

            image.save(
                image_bytes,
                format="PNG"
            )

            st.download_button(
                label="⬇️ Download Image",
                data=image_bytes.getvalue(),
                file_name="ASCEND_generated_image.png",
                mime="image/png",
                use_container_width=True
            )


        except Exception as error:

            st.error(
                "Something went wrong while generating "
                "your image."
            )

            st.exception(error)
