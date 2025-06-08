from flask import Flask, request
from openai import OpenAI
from bs4 import BeautifulSoup
import os, json, requests
from collections import defaultdict
from datetime import datetime

app = Flask(__name__)

# ✅ ENVIRONMENT CONFIG
TWILIO_SID = os.getenv("TWILIO_SID")
TWILIO_AUTH = os.getenv("TWILIO_AUTH")
TWILIO_NUMBER = 'whatsapp:+14155238886'
SHIP24_API_KEY = os.getenv("SHIP24_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)

# ✅ DIRECTORIES
os.makedirs("logs", exist_ok=True)
os.makedirs("vectordb", exist_ok=True)

# ✅ WHATSAPP ROUTE
@app.route('/whatsapp', methods=['POST'])
def whatsapp():
    msg = request.form.get('Body', '').strip()
    sender = request.form.get('From')
    user_id = sender.replace(":", "").replace("+", "")

    if not user_id:
        return "❌ Invalid user ID", 400

    if msg.lower().startswith("track "):
        tracking_number = msg.split("track ", 1)[1].strip()
        response = track_package(tracking_number)

    elif msg.lower().startswith("product "):
        keyword = msg.split("product ", 1)[1].strip()
        response = search_product_vector(keyword)

    elif msg.lower() == "form":
        response = get_latest_form_data()

    elif msg.lower() == "refresh":
        response = refresh_vector_db()

    else:
        response = gpt_reply(user_id, msg)

    send_whatsapp_message(sender, response)
    return "OK", 200


# ✅ SEND MESSAGE
def send_whatsapp_message(to, body):
    url = f'https://api.twilio.com/2010-04-01/Accounts/{TWILIO_SID}/Messages.json'
    auth = (TWILIO_SID, TWILIO_AUTH)
    data = {
        'From': TWILIO_NUMBER,
        'To': to,
        'Body': body
    }
    requests.post(url, data=data, auth=auth)


# ✅ GPT RESPONSE
def gpt_reply(user_id, user_msg):
    save_convo(user_id, "user", user_msg)
    history = load_convo(user_id)[-6:]

    prompt = [
        {
            "role": "system",
            "content": (
                "You are a friendly, helpful customer support assistant for a stylish brand called Factory43. "
                "You're chatting on WhatsApp to assist users with their orders, products, and tracking. "
                "You know the store URL is https://shop.factory43.com/. Never say you're an AI. "
                "Act like you're part of the Factory43 team."
            )
        }
    ] + history

    try:
        res = client.chat.completions.create(
            model="gpt-4",
            messages=prompt,
            temperature=0.6,
            max_tokens=200
        )
        reply = res.choices[0].message.content.strip()
        save_convo(user_id, "assistant", reply)
        return reply
    except Exception as e:
        return f"⚠️ GPT error: {str(e)}"


# ✅ MEMORY (FILE-BASED)
def log_path(user_id):
    return f"logs/{user_id}.json"

def save_convo(user_id, role, content):
    log = load_convo(user_id)
    log.append({"role": role, "content": content})
    with open(log_path(user_id), "w") as f:
        json.dump(log, f)

def load_convo(user_id):
    try:
        with open(log_path(user_id), "r") as f:
            return json.load(f)
    except:
        return []


# ✅ TRACKING FUNCTION
def track_package(tracking_number):
    url = f"https://api.ship24.com/public/v1/trackers/{tracking_number}"
    headers = {'Authorization': f'Bearer {SHIP24_API_KEY}'}
    res = requests.get(url, headers=headers).json()

    try:
        event = res['data']['trackers'][0]['events'][0]
        return f"📦 {event['description']} at {event['location']} on {event['datetime']}"
    except:
        return "❌ Couldn't find tracking info. Please check your tracking number."


# ✅ FORM DATA
def get_latest_form_data():
    # Placeholder
    return "📝 Latest form submission: Name: John Doe, Email: john@example.com"


# ✅ VECTOR SEARCH PRODUCT
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

def refresh_vector_db():
    try:
        site = requests.get("https://shop.factory43.com/").text
        soup = BeautifulSoup(site, "html.parser")
        texts = [x.get_text() for x in soup.find_all(["h1", "h2", "p", "li"])]
        corpus = list(set(filter(lambda t: len(t) > 30, texts)))

        vectorizer = TfidfVectorizer().fit(corpus)
        vectors = vectorizer.transform(corpus)

        with open("vectordb/products.pkl", "wb") as f:
            pickle.dump({"corpus": corpus, "vectorizer": vectorizer, "vectors": vectors}, f)

        return f"✅ Vector DB refreshed with {len(corpus)} entries"
    except Exception as e:
        return f"❌ Refresh failed: {str(e)}"


def search_product_vector(query):
    try:
        with open("vectordb/products.pkl", "rb") as f:
            data = pickle.load(f)

        vec = data["vectorizer"].transform([query])
        sims = (vec @ data["vectors"].T).toarray()[0]
        best = sims.argmax()

        if sims[best] < 0.1:
            return f"❌ No matching product found for '{query}'."

        return f"🛍️ Product info: {data['corpus'][best]}"
    except Exception as e:
        return f"⚠️ Vector DB error: {str(e)}"
