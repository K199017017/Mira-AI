# main.py
"""
ساده‌ترین پروژه AI تستی برای GitHub
مدل طبقه‌بندی متن با Hugging Face (Sentiment Analysis)
"""

from transformers import pipeline
import gradio as gr

def analyze_sentiment(text: str):
    """تحلیل احساسات متن (مثبت، منفی یا خنثی)"""
    sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    result = sentiment_pipeline(text)[0]
    
    label = "مثبت 😊" if result['label'] == "POSITIVE" else "منفی 😞"
    score = round(result['score'] * 100, 2)
    
    return f"**{label}**\nاطمینان: {score}%"

# رابط کاربری ساده با Gradio
iface = gr.Interface(
    fn=analyze_sentiment,
    inputs=gr.Textbox(lines=5, placeholder="متن خود را اینجا بنویسید..."),
    outputs=gr.Markdown(),
    title="🔥 تحلیل احساسات متن (AI Demo)",
    description="یک پروژه تستی ساده با Hugging Face + Gradio",
    examples=[
        ["این محصول عالی بود، خیلی راضی‌ام!"],
        ["خیلی بد بود، اصلاً پیشنهاد نمی‌کنم."],
        ["محصول معمولی بود."]
    ]
)

if __name__ == "__main__":
    print("🚀 اپلیکیشن در حال اجرا...")
    iface.launch(share=True)  # share=True برای لینک عمومی
