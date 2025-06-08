from flask import Flask, request
import requests, os, json
from openai import OpenAI
from collections import defaultdict
from datetime import datetime

app = Flask(__name__)

# ✅ ENV VARS
TWILIO_SID = os.getenv("TWILIO_SID")
TWILIO_AUTH = os.getenv("TWILIO_AUTH")
TWILIO_NUMBER = 'whatsapp:+14155238886'
SHIP24_API_KEY = os.getenv("SHIP24_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ✅ OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# ✅ Ensure directories exist
os.makedirs("logs", exist_ok=True)
os.makedirs("vectordb", exist_ok=True)

@app.route('/whatsapp', methods=['POST'])
def whatsapp():
    msg = request.form.get('Body', '').strip()
    sender = request.form.get('From')

    if not sender:
        return "Missing sender ID", 400

    response_text = None

    # COMMAND PARSING
    if msg.lower().startswith("track "):
        tracking_number = msg.split("track ", 1)[1].strip()
        response_text = track_package(tracking_number)

    elif msg.lower().startswith("product "):
        keyword = msg.split("product ", 1)[1].strip()
        response_text = search_product(keyword)

    elif msg.lower() == "form":
        response_text = form_reply()

    else:
        response_text = gpt_reply(sender, msg)

    send_whatsapp_message(sender, response_text)
    return "OK", 200

def send_whatsapp_message(to, body):
    url = f'https://api.twilio.com/2010-04-01/Accounts/{TWILIO_SID}/Messages.json'
    auth = (TWILIO_SID, TWILIO_AUTH)
    data = {
        'From': TWILIO_NUMBER,
        'To': to,
        'Body': body
    }
    requests.post(url, data=data, auth=auth)

def track_package(tracking_number):
    url = f"https://api.ship24.com/public/v1/trackers/{tracking_number}"
    headers = {'Authorization': f'Bearer {SHIP24_API_KEY}'}
    res = requests.get(url, headers=headers).json()

    try:
        event = res['data']['trackers'][0]['events'][0]
        return f"📦 {event['description']} at {event['location']} on {event['datetime']}"
    except:
        return "❌ Couldn't find tracking info. Please double-check your number."

def search_product(keyword):
    try:
        site = requests.get("https://shop.factory43.com").text
        return f"✅ We have '{keyword}' available!" if keyword.lower() in site.lower() else f"❌ '{keyword}' not found."
    except:
        return f"⚠️ Failed to load site. Please retry later."

def form_reply():
    return "📝 Latest form response: Name: John Doe, Email: john@example.com"

def gpt_reply(user_id, user_msg):
    save_convo(user_id, 'user', user_msg)
    history = load_convo(user_id)[-6:]

    messages = [{"role": "system", "content": "You are a helpful assistant for an online store."}] + history

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.6,
            max_tokens=150
        )
        reply = response.choices[0].message.content.strip()
        save_convo(user_id, 'assistant', reply)
        return reply
    except Exception as e:
        print(f"⚠️ GPT error: {e}")
        return "⚠️ GPT failed. Please try again."

def save_convo(user_id, role, content):
    log_path = f"logs/{user_id}.json"
    convo = load_convo(user_id)
    convo.append({"role": role, "content": content, "timestamp": datetime.utcnow().isoformat()})
    with open(log_path, "w") as f:
        json.dump(convo, f)

def load_convo(user_id):
    log_path = f"logs/{user_id}.json"
    if os.path.exists(log_path):
        with open(log_path, "r") as f:
            return json.load(f)
    return []
