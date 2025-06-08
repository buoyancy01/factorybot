from flask import Flask, request
import requests, os, openai, json
from collections import defaultdict
import faiss
import numpy as np
from bs4 import BeautifulSoup

app = Flask(__name__)

# ✅ ENV VARS
TWILIO_SID = os.getenv("TWILIO_SID")
TWILIO_AUTH = os.getenv("TWILIO_AUTH")
TWILIO_NUMBER = 'whatsapp:+14155238886'
SHIP24_API_KEY = os.getenv("SHIP24_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

# ✅ MEMORY STORAGE
user_memory = defaultdict(list)  # Can be swapped with file/db/memcache

# ✅ VECTOR DB
index = None
product_texts = []

@app.route('/whatsapp', methods=['POST'])
def whatsapp():
    msg = request.form.get('Body', '').strip()
    sender = request.form.get('From')
    response_text = None

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

def save_convo(user_id, role, content):
    user_memory[user_id].append({"role": role, "content": content})
    log_path = f"logs/{user_id}.json"
    history = user_memory[user_id][-20:]
    with open(log_path, "w") as f:
        json.dump(history, f, indent=2)

def embed(text):
    response = openai.embeddings.create(input=[text], model="text-embedding-ada-002")
    return np.array(response.data[0].embedding, dtype='float32')

def query_vector_db(query):
    global index, product_texts
    if index is None or not product_texts:
        return ""
    q_emb = embed(query).astype('float32')
    D, I = index.search(np.array([q_emb]), k=1)
    if D[0][0] > 0.85:
        return product_texts[I[0][0]]
    return ""

def gpt_reply(user_id, user_msg):
    save_convo(user_id, 'user', user_msg)
    context = query_vector_db(user_msg)
    messages = [
        {"role": "system", "content": "You are a helpful assistant for a clothing store."},
    ]
    if context:
        messages.append({"role": "system", "content": f"Relevant product info: {context}"})
    messages += user_memory[user_id][-6:] + [{"role": "user", "content": user_msg}]

    try:
        res = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.6,
            max_tokens=200
        )
        reply = res.choices[0].message.content.strip()
        save_convo(user_id, 'assistant', reply)
        return reply
    except Exception as e:
        return "⚠️ GPT failed. Please try again."

@app.route('/refresh_products')
def refresh_products():
    try:
        global index, product_texts
        html = requests.get("https://shop.factory43.com").text
        soup = BeautifulSoup(html, 'html.parser')
        product_divs = soup.find_all('div')
        chunks = []
        for div in product_divs:
            txt = div.get_text(strip=True)
            if 40 < len(txt) < 500:
                chunks.append(txt)
        product_texts = chunks
        embeddings = [embed(chunk) for chunk in chunks]
        dim = len(embeddings[0])
        index = faiss.IndexFlatL2(dim)
        index.add(np.array(embeddings).astype('float32'))
        return "✅ Vector DB refreshed."
    except Exception as e:
        return f"❌ Refresh failed: {e}"
