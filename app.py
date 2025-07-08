import streamlit as st
import torch
import time
from deep_translator import GoogleTranslator

# Check if running on Streamlit Cloud
try:
    import kagglehub
    from transformers import AutoTokenizer, AutoModelForCausalLM
    STREAMLIT_CLOUD = False
except ImportError:
    STREAMLIT_CLOUD = True

LANGUAGES = {
    'English': 'en', 'Spanish': 'es', 'French': 'fr', 'German': 'de',
    'Italian': 'it', 'Portuguese': 'pt', 'Russian': 'ru', 'Chinese': 'zh',
    'Japanese': 'ja', 'Korean': 'ko', 'Arabic': 'ar', 'Hindi': 'hi',
    'Dutch': 'nl', 'Swedish': 'sv', 'Polish': 'pl'
}

@st.cache_resource
def load_model():
    """Load Gemma 3n model with fallback for Streamlit Cloud"""
    if STREAMLIT_CLOUD:
        st.warning("⚠️ Running in demo mode - Gemma 3n model not available on Streamlit Cloud")
        return None, None
    
    try:
        with st.spinner("🔄 Loading Gemma 3n model... This may take a few minutes on first run."):
            # Download the E2B model (smaller, 2GB memory footprint)
            GEMMA_PATH = kagglehub.model_download("google/gemma-3n/transformers/gemma-3n-e2b")
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(GEMMA_PATH)
            
            # Load model with optimizations for lower memory usage
            model = AutoModelForCausalLM.from_pretrained(
                GEMMA_PATH,
                torch_dtype=torch.float16,  # Use half precision
                device_map="auto",
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )
            
            st.success("✅ Gemma 3n model loaded successfully!")
            return tokenizer, model
            
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.info("💡 This might be due to memory constraints. Try running locally or on a machine with more RAM.")
        return None, None

def translate_text(text, target_language):
    """Translate text using deep-translator"""
    try:
        if target_language == 'English' or not text.strip():
            return text
        
        target_code = LANGUAGES.get(target_language, 'en')
        if target_code == 'en':
            return text
            
        translator = GoogleTranslator(source='auto', target=target_code)
        translated = translator.translate(text)
        return translated
    except Exception as e:
        return f"Translation error: {str(e)}"

def generate_with_gemma(prompt, template_type, num_outputs, temperature, max_tokens, tokenizer, model):
    """Generate content using Gemma 3n model"""
    try:
        # Apply marketing-specific prompts
        if template_type == "Landing Page Headlines":
            system_prompt = "You are a skilled marketing copywriter. Create compelling, attention-grabbing headlines that drive conversions. Focus on benefits, urgency, and emotional appeal."
            prompt = f"{system_prompt}\n\nCreate a powerful headline for: {prompt}\n\nHeadline:"
        elif template_type == "Product Descriptions":
            system_prompt = "You are an expert product copywriter. Write persuasive descriptions that highlight benefits, features, and value propositions to drive sales."
            prompt = f"{system_prompt}\n\nWrite a compelling product description for: {prompt}\n\nDescription:"
        elif template_type == "Email Subject Lines":
            system_prompt = "You are an email marketing specialist. Create subject lines that maximize open rates with urgency, curiosity, and clear value."
            prompt = f"{system_prompt}\n\nGenerate an effective email subject line for: {prompt}\n\nSubject:"
        elif template_type == "Social Media Posts":
            system_prompt = "You are a social media content creator. Write engaging posts that drive engagement, shares, and conversions."
            prompt = f"{system_prompt}\n\nCreate a social media post for: {prompt}\n\nPost:"
        elif template_type == "Ad Copy":
            system_prompt = "You are a digital advertising specialist. Write compelling ad copy that drives clicks and conversions."
            prompt = f"{system_prompt}\n\nWrite ad copy for: {prompt}\n\nAd:"
        
        # Tokenize the input
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        results = []
        for i in range(num_outputs):
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=0.95,
                    top_k=50,
                    repetition_penalty=1.1,
                    no_repeat_ngram_size=3,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            
            # Decode the output
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the new generated content
            new_content = generated_text.replace(prompt, "").strip()
            
            # Clean up the output
            if new_content:
                # Take the first line/sentence for cleaner output
                lines = new_content.split('\n')
                clean_content = lines[0].strip() if lines else new_content
                
                if clean_content and len(clean_content) > 10:  # Ensure meaningful content
                    results.append(f"**Variation {i+1}:**\n{clean_content}")
        
        return "\n\n".join(results) if results else "No suitable content generated. Try adjusting your prompt or settings."
        
    except Exception as e:
        return f"Generation error: {str(e)}"

def demo_generate_content(prompt, template_type, num_outputs):
    """Demo content generation when model is not available"""
    demo_content = {
        "Landing Page Headlines": [
            f"🚀 Transform Your Business with {prompt} - See Results in 24 Hours!",
            f"💡 The Ultimate {prompt} Solution That Actually Works",
            f"⚡ Revolutionary {prompt} - Join 10,000+ Happy Customers",
            f"🎯 Stop Struggling with {prompt} - We've Got You Covered",
            f"🔥 The {prompt} Everyone's Talking About - Try It Free!"
        ],
        "Product Descriptions": [
            f"Experience the next generation of {prompt} designed for modern professionals. Our innovative solution delivers exceptional results while saving you time and effort. Perfect for busy teams who demand quality and efficiency.",
            f"Transform your workflow with our premium {prompt}. Built with cutting-edge technology and user-friendly design for maximum productivity. Trusted by industry leaders worldwide.",
            f"Discover why thousands choose our {prompt}. Reliable, efficient, and backed by 24/7 support for your peace of mind. Start your free trial today.",
            f"Elevate your business with our award-winning {prompt}. Proven results, exceptional quality, and unmatched customer satisfaction guaranteed.",
            f"The ultimate {prompt} for ambitious professionals. Streamline operations, boost efficiency, and achieve your goals faster than ever before."
        ],
        "Email Subject Lines": [
            f"🎯 Your {prompt} Strategy Needs This (Open Now)",
            f"URGENT: {prompt} Deadline Approaching - Act Fast!",
            f"✨ New {prompt} Feature Just Launched - Exclusive Access",
            f"📈 Boost Your {prompt} Results by 300% (Proven Method)",
            f"🔥 Limited Time: {prompt} Offer Ends Tonight"
        ],
        "Social Media Posts": [
            f"Just discovered the game-changing {prompt} that's revolutionizing how we work! 🚀 Who else is ready to level up? #productivity #innovation",
            f"💡 Pro tip: The secret to mastering {prompt} isn't what you think... Here's what actually works 👇 #tips #success",
            f"🔥 Hot take: If you're not using {prompt} in 2025, you're already behind. Here's why it matters... #trendingnow",
            f"✨ Transform your {prompt} game with this simple trick. Results speak for themselves! 📈 #results #growth",
            f"🎯 Anyone else obsessed with optimizing their {prompt}? Share your best tips below! 👇 #community #sharing"
        ],
        "Ad Copy": [
            f"Ready to revolutionize your {prompt}? Our proven system delivers results in just 24 hours. Join thousands of satisfied customers. Click now!",
            f"Stop wasting time on {prompt} that doesn't work. Our solution is trusted by industry leaders. Try it risk-free today!",
            f"Transform your {prompt} with our award-winning platform. 30-day money-back guarantee. Start your free trial now!",
            f"The #{prompt} solution that actually works. Proven results, happy customers, unbeatable support. Get started today!",
            f"Breakthrough {prompt} technology that's changing everything. Limited time offer - 50% off for new users. Act now!"
        ]
    }
    
    selected_content = demo_content.get(template_type, demo_content["Landing Page Headlines"])
    return "\n\n".join([f"**Variation {i+1}:**\n{content}" for i, content in enumerate(selected_content[:num_outputs])])

# App Configuration
st.set_page_config(
    page_title="PromptCoach - Gemma 3n Marketing Assistant",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }
    
    .feature-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    
    .output-container {
        background: #ffffff;
        padding: 2rem;
        border-radius: 15px;
        border: 2px solid #e9ecef;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
    }
    
    .demo-warning {
        background: #fff3cd;
        border: 1px solid #ffeaa7;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>🚀 PromptCoach - Powered by Gemma 3n</h1>
    <p style="font-size: 1.2rem; margin-top: 1rem;">AI-Powered Marketing Content Generator for the Modern Marketer</p>
    <p style="font-size: 1rem; opacity: 0.9;">Built for Google Gemma 3n Hackathon | Multilingual Support | Real-time Generation</p>
</div>
""", unsafe_allow_html=True)

# Model Loading
if "tokenizer" not in st.session_state or "model" not in st.session_state:
    tokenizer, model = load_model()
    st.session_state.tokenizer = tokenizer
    st.session_state.model = model

# Sidebar Configuration
with st.sidebar:
    st.markdown("## ⚙️ Generation Settings")
    
    template = st.selectbox(
        "Content Type",
        ["Landing Page Headlines", "Product Descriptions", "Email Subject Lines", "Social Media Posts", "Ad Copy", "Custom Prompt"],
        help="Choose the type of marketing content you want to generate"
    )
    
    num_outputs = st.slider(
        "Number of Variations",
        1, 5, 3,
        help="Generate multiple variations for A/B testing"
    )
    
    if not STREAMLIT_CLOUD and st.session_state.model is not None:
        temperature = st.slider(
            "Creativity Level",
            0.1, 1.5, 0.8, 0.1,
            help="Higher values = more creative output"
        )
        
        max_tokens = st.slider(
            "Max Output Length",
            50, 300, 150, 25,
            help="Maximum number of tokens to generate"
        )
    else:
        temperature = 0.8
        max_tokens = 150
    
    st.markdown("---")
    st.markdown("## 🌐 Translation")
    
    translate_output_to = st.selectbox(
        "Translate Output To",
        list(LANGUAGES.keys()),
        index=0,
        help="Translate generated content to another language"
    )
    
    st.markdown("---")
    st.markdown("## 📊 Model Info")
    
    if STREAMLIT_CLOUD:
        st.markdown("""
        <div class="demo-warning">
            <strong>Demo Mode Active</strong><br>
            Running template-based generation.<br>
            Deploy locally to use Gemma 3n model.
        </div>
        """, unsafe_allow_html=True)
    elif st.session_state.model is not None:
        st.success("✅ Gemma 3n E2B Model Loaded")
        st.info("🔧 Using 2GB memory footprint")
    else:
        st.error("❌ Model not loaded")

# Main Content
col1, col2 = st.columns([3, 1])

with col1:
    st.markdown("## 📝 Enter Your Marketing Prompt")
    
    # Dynamic placeholders based on content type
    placeholders = {
        "Landing Page Headlines": "AI-powered project management tool for remote teams",
        "Product Descriptions": "Wireless noise-cancelling headphones with 30-hour battery life",
        "Email Subject Lines": "New product launch announcement with early bird discount",
        "Social Media Posts": "Behind-the-scenes content about our company culture",
        "Ad Copy": "Online course teaching digital marketing to beginners",
        "Custom Prompt": "Write a compelling call-to-action for my SaaS landing page"
    }
    
    prompt_input = st.text_area(
        "Your Marketing Prompt",
        placeholder=placeholders.get(template, "Enter your marketing prompt here..."),
        height=120,
        help="Be specific about your product, target audience, and key benefits"
    )
    
    # Content type specific tips
    tips = {
        "Landing Page Headlines": "💡 **Tip**: Include your target audience, main benefit, and create urgency or curiosity",
        "Product Descriptions": "💡 **Tip**: Focus on benefits over features, include social proof, and address pain points",
        "Email Subject Lines": "💡 **Tip**: Keep it under 50 characters, create urgency, and personalize when possible",
        "Social Media Posts": "💡 **Tip**: Include hashtags, engage with questions, and use emojis strategically",
        "Ad Copy": "💡 **Tip**: Include a clear value proposition, call-to-action, and address objections"
    }
    
    st.markdown(tips.get(template, "💡 **Tip**: Be clear and specific about what you want to create"))

with col2:
    st.markdown("## 🎯 Quick Templates")
    
    template_options = {
        "💼 B2B SaaS": "cloud-based CRM software for small businesses",
        "🛍️ E-commerce": "eco-friendly water bottles with temperature control",
        "📱 Mobile App": "meditation app with AI-powered personalized sessions",
        "🎓 Online Course": "complete web development bootcamp for beginners",
        "🏥 Healthcare": "telemedicine platform connecting patients with specialists",
        "🎮 Gaming": "multiplayer strategy game with blockchain rewards"
    }
    
    for button_text, template_prompt in template_options.items():
        if st.button(button_text, use_container_width=True):
            st.session_state.quick_prompt = template_prompt
            st.rerun()
    
    if 'quick_prompt' in st.session_state:
        prompt_input = st.session_state.quick_prompt

# Generation Button
if st.button("🚀 Generate Marketing Content", type="primary", use_container_width=True):
    if not prompt_input.strip():
        st.error("Please enter a marketing prompt to generate content.")
    else:
        with st.spinner("🎨 Generating your marketing content..."):
            start_time = time.time()
            
            # Generate content
            if not STREAMLIT_CLOUD and st.session_state.model is not None:
                output = generate_with_gemma(
                    prompt_input, 
                    template, 
                    num_outputs, 
                    temperature, 
                    max_tokens,
                    st.session_state.tokenizer,
                    st.session_state.model
                )
                model_used = "Gemma 3n E2B"
            else:
                output = demo_generate_content(prompt_input, template, num_outputs)
                model_used = "Demo Mode"
            
            # Translate output if needed
            if translate_output_to != 'English':
                with st.spinner("🌐 Translating output..."):
                    translated_output = translate_text(output, translate_output_to)
            else:
                translated_output = output
            
            generation_time = time.time() - start_time
            
            # Display results
            st.success(f"✅ Content generated in {generation_time:.2f} seconds!")
            
            # Results container
            st.markdown(f"""
            <div class="output-container">
                <h3>📄 Generated {template}</h3>
                <p><strong>Prompt:</strong> {prompt_input}</p>
                <p><strong>Model:</strong> {model_used} | <strong>Language:</strong> {translate_output_to}</p>
                <hr>
                <div style="white-space: pre-wrap; font-family: system-ui;">
{translated_output}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Action buttons
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("🔄 Generate More", use_container_width=True):
                    st.rerun()
            
            with col2:
                st.download_button(
                    label="📥 Download Content",
                    data=translated_output,
                    file_name=f"{template.lower().replace(' ', '_')}_content.txt",
                    mime="text/plain",
                    use_container_width=True
                )
            
            with col3:
                if st.button("📊 Analytics", use_container_width=True):
                    st.info("📈 **Content Analytics**: Word count, readability score, and engagement predictions coming soon!")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 2rem; background: #f8f9fa; border-radius: 10px;">
    <h4>🚀 PromptCoach - Gemma 3n Marketing Assistant</h4>
    <p>Built for <strong>Google Gemma 3n Hackathon</strong> | Powered by <strong>Gemma 3n E2B Model</strong></p>
    <p><em>Generate compelling marketing content with AI-powered creativity and multilingual support</em></p>
    <p>💡 <strong>Features:</strong> Real-time generation | A/B testing | Multi-language support | Mobile-optimized AI</p>
</div>
""", unsafe_allow_html=True)
