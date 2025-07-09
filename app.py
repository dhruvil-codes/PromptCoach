import streamlit as st
import torch
import time
import json
from typing import Dict, List, Tuple

# Check if running locally with Gemma 3n access
try:
    import kagglehub
    from transformers import AutoTokenizer, AutoModelForCausalLM
    GEMMA_AVAILABLE = True
except ImportError:
    GEMMA_AVAILABLE = False

# Marketing prompt categories
PROMPT_CATEGORIES = {
    "Landing Page Headlines": {
        "description": "Compelling headlines that grab attention and drive conversions",
        "example": "AI-powered project management tool for remote teams"
    },
    "Product Descriptions": {
        "description": "Persuasive descriptions that highlight benefits and drive sales",
        "example": "Wireless noise-cancelling headphones with 30-hour battery"
    },
    "Email Subject Lines": {
        "description": "Subject lines that maximize open rates and engagement",
        "example": "New product launch with early bird discount"
    },
    "Social Media Posts": {
        "description": "Engaging posts that drive interaction and sharing",
        "example": "Behind-the-scenes content about company culture"
    },
    "Ad Copy": {
        "description": "Compelling ad copy that drives clicks and conversions",
        "example": "Online course teaching digital marketing"
    }
}

@st.cache_resource
def load_gemma_model():
    """Load Gemma 3n model from Kaggle Hub"""
    if not GEMMA_AVAILABLE:
        return None, None
    
    try:
        with st.spinner("Loading Gemma 3n model..."):
            # Download the Gemma 3n E2B model
            path = kagglehub.model_download("google/gemma-3n/transformers/gemma-3n-e2b")
            
            # Load tokenizer and model
            tokenizer = AutoTokenizer.from_pretrained(path)
            model = AutoModelForCausalLM.from_pretrained(
                path,
                torch_dtype=torch.float16,
                device_map="auto",
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )
            
            return tokenizer, model
            
    except Exception as e:
        st.error(f"Error loading Gemma 3n: {str(e)}")
        return None, None

def evaluate_prompt(prompt: str, category: str, tokenizer, model) -> Dict:
    """Evaluate a marketing prompt and provide feedback"""
    
    evaluation_prompt = f"""
You are PromptCoach, an expert marketing prompt evaluator. Analyze this {category.lower()} prompt and provide constructive feedback.

PROMPT TO EVALUATE: "{prompt}"
CATEGORY: {category}

Please provide:
1. SCORE (1-10): Rate the prompt's effectiveness
2. STRENGTHS: What works well in this prompt
3. WEAKNESSES: What could be improved
4. IMPROVED VERSION: Rewrite the prompt to be more effective
5. A/B VARIATION: Provide an alternative version for testing

Format your response as:
SCORE: [number]
STRENGTHS: [strengths]
WEAKNESSES: [weaknesses]
IMPROVED VERSION: [improved prompt]
A/B VARIATION: [alternative prompt]
"""

    try:
        inputs = tokenizer(evaluation_prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=400,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                top_k=50,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_text = response.replace(evaluation_prompt, "").strip()
        
        # Parse the response
        return parse_evaluation_response(generated_text)
        
    except Exception as e:
        return {"error": f"Evaluation failed: {str(e)}"}

def parse_evaluation_response(response: str) -> Dict:
    """Parse the evaluation response into structured data"""
    try:
        lines = response.split('\n')
        result = {}
        
        for line in lines:
            if line.startswith('SCORE:'):
                score_text = line.replace('SCORE:', '').strip()
                try:
                    result['score'] = int(score_text.split()[0])
                except:
                    result['score'] = 5
            elif line.startswith('STRENGTHS:'):
                result['strengths'] = line.replace('STRENGTHS:', '').strip()
            elif line.startswith('WEAKNESSES:'):
                result['weaknesses'] = line.replace('WEAKNESSES:', '').strip()
            elif line.startswith('IMPROVED VERSION:'):
                result['improved'] = line.replace('IMPROVED VERSION:', '').strip()
            elif line.startswith('A/B VARIATION:'):
                result['ab_variation'] = line.replace('A/B VARIATION:', '').strip()
        
        return result
        
    except Exception:
        return {
            'score': 5,
            'strengths': 'Analysis pending...',
            'weaknesses': 'Analysis pending...',
            'improved': 'Improved version pending...',
            'ab_variation': 'A/B variation pending...'
        }

def demo_evaluation(prompt: str, category: str) -> Dict:
    """Demo evaluation when Gemma 3n is not available"""
    return {
        'score': 7,
        'strengths': f"Clear product positioning and target audience identification for {category.lower()}",
        'weaknesses': "Could benefit from stronger emotional triggers and more specific value propositions",
        'improved': f"Transform your business with our revolutionary {prompt} - Join 10,000+ satisfied customers who've seen 300% growth in just 30 days!",
        'ab_variation': f"Finally, a {prompt} that actually works - See results in 24 hours or your money back!"
    }

# App Configuration
st.set_page_config(
    page_title="PromptCoach - Gemma 3n Marketing Assistant",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for minimalist design with Inter font
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif !important;
    }
    
    .main {
        padding: 1rem 2rem;
        max-width: 800px;
        margin: 0 auto;
    }
    
    .header {
        text-align: center;
        margin-bottom: 3rem;
        padding: 2rem 0;
        border-bottom: 1px solid #e5e7eb;
    }
    
    .header h1 {
        font-size: 2.5rem;
        font-weight: 700;
        color: #111827;
        margin-bottom: 0.5rem;
    }
    
    .header p {
        font-size: 1.1rem;
        color: #6b7280;
        font-weight: 400;
    }
    
    .input-section {
        background: #f9fafb;
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        border: 1px solid #e5e7eb;
    }
    
    .results-section {
        background: white;
        padding: 2rem;
        border-radius: 12px;
        border: 1px solid #e5e7eb;
        margin-bottom: 2rem;
    }
    
    .score-display {
        display: flex;
        align-items: center;
        gap: 1rem;
        margin-bottom: 1.5rem;
    }
    
    .score-circle {
        width: 60px;
        height: 60px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.5rem;
        font-weight: 700;
        color: white;
    }
    
    .score-good { background: #10b981; }
    .score-medium { background: #f59e0b; }
    .score-poor { background: #ef4444; }
    
    .feedback-item {
        margin-bottom: 1.5rem;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #e5e7eb;
    }
    
    .feedback-item h4 {
        font-size: 1rem;
        font-weight: 600;
        color: #111827;
        margin-bottom: 0.5rem;
    }
    
    .feedback-item p {
        color: #4b5563;
        line-height: 1.6;
        margin: 0;
    }
    
    .strengths { border-left-color: #10b981; background: #f0fdf4; }
    .weaknesses { border-left-color: #ef4444; background: #fef2f2; }
    .improved { border-left-color: #3b82f6; background: #eff6ff; }
    .variation { border-left-color: #8b5cf6; background: #f5f3ff; }
    
    .stButton button {
        width: 100%;
        background: #111827;
        color: white;
        border: none;
        padding: 0.75rem 1.5rem;
        border-radius: 8px;
        font-weight: 500;
        font-size: 1rem;
        transition: all 0.2s;
    }
    
    .stButton button:hover {
        background: #1f2937;
        transform: translateY(-1px);
    }
    
    .stSelectbox > div > div {
        border-radius: 8px;
        border: 1px solid #d1d5db;
    }
    
    .stTextArea > div > div > textarea {
        border-radius: 8px;
        border: 1px solid #d1d5db;
        font-family: 'Inter', sans-serif;
    }
    
    .demo-notice {
        background: #fef3c7;
        border: 1px solid #f59e0b;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 2rem;
        text-align: center;
    }
    
    .demo-notice p {
        color: #92400e;
        font-weight: 500;
        margin: 0;
    }
</style>
""", unsafe_allow_html=True)

# Load model
if "tokenizer" not in st.session_state or "model" not in st.session_state:
    tokenizer, model = load_gemma_model()
    st.session_state.tokenizer = tokenizer
    st.session_state.model = model

# Header
st.markdown("""
<div class="header">
    <h1>🧠 PromptCoach</h1>
    <p>AI-powered marketing prompt evaluator built with Gemma 3n</p>
</div>
""", unsafe_allow_html=True)

# Demo notice if Gemma not available
if not GEMMA_AVAILABLE or st.session_state.model is None:
    st.markdown("""
    <div class="demo-notice">
        <p>⚠️ Demo Mode: Install kagglehub and transformers to use Gemma 3n locally</p>
    </div>
    """, unsafe_allow_html=True)

# Input Section
st.markdown('<div class="input-section">', unsafe_allow_html=True)

# Category selection
category = st.selectbox(
    "Select Content Type",
    list(PROMPT_CATEGORIES.keys()),
    help="Choose the type of marketing content you want to evaluate"
)

# Display category info
if category:
    st.markdown(f"**{category}**: {PROMPT_CATEGORIES[category]['description']}")
    st.markdown(f"*Example: {PROMPT_CATEGORIES[category]['example']}*")

# Prompt input
prompt_input = st.text_area(
    "Enter your marketing prompt to evaluate",
    height=120,
    placeholder=f"Example: {PROMPT_CATEGORIES[category]['example']}",
    help="Paste your current marketing prompt here for evaluation and improvement"
)

# Evaluate button
evaluate_button = st.button("🔍 Evaluate Prompt", type="primary")

st.markdown('</div>', unsafe_allow_html=True)

# Results Section
if evaluate_button:
    if not prompt_input.strip():
        st.error("Please enter a marketing prompt to evaluate.")
    else:
        with st.spinner("Analyzing your prompt with Gemma 3n..."):
            start_time = time.time()
            
            # Evaluate prompt
            if GEMMA_AVAILABLE and st.session_state.model is not None:
                result = evaluate_prompt(
                    prompt_input, 
                    category, 
                    st.session_state.tokenizer, 
                    st.session_state.model
                )
                model_used = "Gemma 3n E2B"
            else:
                result = demo_evaluation(prompt_input, category)
                model_used = "Demo Mode"
            
            analysis_time = time.time() - start_time
            
            if "error" in result:
                st.error(result["error"])
            else:
                # Display results
                st.markdown('<div class="results-section">', unsafe_allow_html=True)
                
                # Score display
                score = result.get('score', 5)
                if score >= 8:
                    score_class = "score-good"
                elif score >= 6:
                    score_class = "score-medium"
                else:
                    score_class = "score-poor"
                
                st.markdown(f"""
                <div class="score-display">
                    <div class="score-circle {score_class}">{score}</div>
                    <div>
                        <h3 style="margin: 0; color: #111827;">Prompt Score</h3>
                        <p style="margin: 0; color: #6b7280;">Analyzed in {analysis_time:.2f}s with {model_used}</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Feedback sections
                feedback_sections = [
                    ("strengths", "✅ Strengths", "What's working well in your prompt"),
                    ("weaknesses", "⚠️ Areas for Improvement", "What could be enhanced"),
                    ("improved", "🚀 Improved Version", "Enhanced version of your prompt"),
                    ("variation", "🔄 A/B Test Variation", "Alternative version for testing")
                ]
                
                for key, title, description in feedback_sections:
                    if key in result:
                        st.markdown(f"""
                        <div class="feedback-item {key}">
                            <h4>{title}</h4>
                            <p>{result[key]}</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Action buttons
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("🔄 Analyze Another"):
                        st.rerun()
                with col2:
                    # Create downloadable report
                    report = f"""PromptCoach Analysis Report
Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}
Category: {category}

Original Prompt: {prompt_input}

Score: {score}/10

Strengths: {result.get('strengths', 'N/A')}

Areas for Improvement: {result.get('weaknesses', 'N/A')}

Improved Version: {result.get('improved', 'N/A')}

A/B Variation: {result.get('variation', 'N/A')}

Generated with: {model_used}
"""
                    st.download_button(
                        label="📥 Download Report",
                        data=report,
                        file_name=f"promptcoach_analysis_{int(time.time())}.txt",
                        mime="text/plain"
                    )

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 2rem; color: #6b7280;">
    <p><strong>PromptCoach</strong> - Built for Google Gemma 3n Hackathon</p>
    <p>Powered by Gemma 3n E2B | Designed for marketers and founders</p>
</div>
""", unsafe_allow_html=True)
