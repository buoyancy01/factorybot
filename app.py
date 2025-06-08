from flask import Flask, request
import requests, os, json, openai
from collections import defaultdict

app = Flask(__name__)

# ✅ Create required folders if missing
os.makedirs("logs", exist_ok=True)
os.makedirs("vectordb", exist_ok=True)

# ✅ ENV VARS
TWILIO_SID = os.getenv("TWILIO_SID")
TWILIO_AUTH = os.getenv("TWILIO_AUTH")
TWILIO_NUMBER = 'whatsapp:+14155238886'
SHIP24_API_KEY = os.getenv("SHIP24_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

@app.route('/whatsapp', methods=['POST'])
def whatsapp():
    msg = request.form.get('Body', '').strip()
    sender = request.form.get('From')

    response_text = None

    try:
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

    except Exception as e:
        print(f"❌ Error: {e}")
        return "Internal Error", 500


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
    # Placeholder: Replace with Tally API or webhook logic
    return "📝 Latest form response: Name: John Doe, Email: john@example.com"


def gpt_reply(user_id, user_msg):
    """Use GPT and feed it conversation history"""
    save_convo(user_id, 'user', user_msg)

    history = load_convo(user_id)[-6:]
    prompt = [{"role": "system", "content": "You are a helpful assistant for an online store."}] + history

    try:
        res = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=prompt,
            temperature=0.6,
            max_tokens=150
        )
        reply = res['choices'][0]['message']['content'].strip()
        save_convo(user_id, 'assistant', reply)
        return reply
    except Exception as e:
        print(f"⚠️ GPT error: {e}")
        return "⚠️ GPT failed. Please try again."


def convo_log_path(user_id):
    """Return file path for user memory"""
    filename = user_id.replace(":", "").replace("+", "") + ".json"
    return os.path.join("logs", filename)


def load_convo(user_id):
    """Load previous conversation if exists"""
    path = convo_log_path(user_id)
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return []


def save_convo(user_id, role, content):
    """Append to user history and save"""
    path = convo_log_path(user_id)
    convo = load_convo(user_id)
    convo.append({"role": role, "content": content})
    with open(path, "w") as f:
        json.dump(convo, f, indent=2)
