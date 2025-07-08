import streamlit as st
import kagglehub
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from googletrans import Translator

LANGUAGES = {
    'English': 'en', 'Spanish': 'es', 'French': 'fr', 'German': 'de',
    'Italian': 'it', 'Portuguese': 'pt', 'Russian': 'ru', 'Chinese': 'zh',
    'Japanese': 'ja', 'Korean': 'ko', 'Arabic': 'ar', 'Hindi': 'hi',
    'Dutch': 'nl', 'Swedish': 'sv', 'Polish': 'pl'
}

@st.cache_resource
def load_model():
    try:
        GEMMA_PATH = kagglehub.model_download("google/gemma-3n/transformers/gemma-3n-e2b")
        tokenizer = AutoTokenizer.from_pretrained(GEMMA_PATH)
        model = AutoModelForCausalLM.from_pretrained(GEMMA_PATH, torch_dtype="auto", device_map="auto")
        return tokenizer, model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

def translate_text(text, target_language):
    try:
        if target_language == 'English' or not text.strip():
            return text
        translator = Translator()
        target_code = LANGUAGES.get(target_language, 'en')
        translated = translator.translate(text, dest=target_code)
        return translated.text
    except Exception as e:
        return f"Translation error: {str(e)}"

def generate_content(prompt, template_type, num_outputs, temperature, max_tokens):
    try:
        if template_type != "Custom Prompt":
            if template_type == "Landing Page Headlines":
                prompt = f"Create a compelling headline for: {prompt}\n\nHeadline:"
            elif template_type == "Product Descriptions":
                prompt = f"Write a persuasive product description for: {prompt}\n\nDescription:"
            elif template_type == "Email Subject Lines":
                prompt = f"Generate an effective email subject line for: {prompt}\n\nSubject:"

        inputs = st.session_state.tokenizer(prompt, return_tensors="pt").to(st.session_state.model.device)
        results = []
        for i in range(num_outputs):
            outputs = st.session_state.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=0.95,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3,
                pad_token_id=st.session_state.tokenizer.eos_token_id
            )
            text = st.session_state.tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated = text.replace(prompt, "").strip()
            if generated:
                generated = generated.split('\n')[0].strip()
                if generated:
                    results.append(f"{i+1}. {generated}")
        return "\n\n".join(results)
    except Exception as e:
        return f"Error: {str(e)}"

st.set_page_config(page_title="PromptCoach", layout="wide")
st.title("🚀 PromptCoach – Powered by Gemma 3n")
st.markdown("Generate creative marketing content with multilingual support!")

if "tokenizer" not in st.session_state or "model" not in st.session_state:
    tokenizer, model = load_model()
    st.session_state.tokenizer = tokenizer
    st.session_state.model = model

with st.sidebar:
    st.header("⚙️ Settings")
    template = st.selectbox("Prompt Type", ["Custom Prompt", "Landing Page Headlines", "Product Descriptions", "Email Subject Lines"])
    num_outputs = st.slider("Number of Outputs", 1, 5, 3)
    temperature = st.slider("Creativity (Temperature)", 0.1, 2.0, 0.8, 0.1)
    max_tokens = st.slider("Max Token Length", 20, 200, 100, 10)
    translate_prompt_to = st.selectbox("Translate Prompt To", list(LANGUAGES.keys()), index=0)
    translate_output_to = st.selectbox("Translate Output To", list(LANGUAGES.keys()), index=0)

st.subheader("📝 Enter Your Prompt")
prompt_input = st.text_area("Your prompt", "AI productivity tool")

if st.button("🚀 Generate"):
    with st.spinner("Generating content..."):
        translated_prompt = translate_text(prompt_input, translate_prompt_to)
        output = generate_content(translated_prompt, template, num_outputs, temperature, max_tokens)
        translated_output = translate_text(output, translate_output_to)
        st.success("Done!")
        st.subheader("📄 Output")
        st.code(translated_output)