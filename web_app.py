import os
import json
import uuid
import requests
from datetime import datetime, timedelta
from openai import OpenAI
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn
import trafilatura
from ddgs import DDGS  # 备用，但主搜索使用博查

load_dotenv()
api_key = os.getenv("DEEPSEEK_API_KEY")
bocha_api_key = os.getenv("BOCHA_API_KEY")
client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

# ---------- ChromaDB 多集合 ----------
chroma_client = chromadb.PersistentClient(path="./my_ai_memory")
embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

COLLECTIONS = {
    "chat": "chat_memory",
    "mail": "mail_memory",
    "bill": "bill_memory",
    "test": "test_memory"
}
memory_collections = {}
for key, name in COLLECTIONS.items():
    memory_collections[key] = chroma_client.get_or_create_collection(
        name=name, embedding_function=embed_fn
    )

# ---------- 工具定义 ----------
tools = [
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "将内容写入服务器上的文件。",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {"type": "string"},
                    "content": {"type": "string"}
                },
                "required": ["filename", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "搜索互联网信息（博查 API），返回标题、链接和正文摘要。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "order_item",
            "description": "模拟下单购买商品。",
            "parameters": {
                "type": "object",
                "properties": {
                    "item": {"type": "string"},
                    "price": {"type": "number"}
                },
                "required": ["item", "price"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "获取当前的准确北京时间。",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "switch_memory_mode",
            "description": "切换当前使用的记忆库模式（chat/mail/bill/test）。",
            "parameters": {
                "type": "object",
                "properties": {
                    "mode": {"type": "string", "enum": ["chat", "mail", "bill", "test"]}
                },
                "required": ["mode"]
            }
        }
    }
]

# ---------- 工具实现 ----------
def write_file(filename, content):
    with open(filename, "w", encoding="utf-8") as f:
        f.write(content)
    return f"文件 {filename} 已写入成功"

def fetch_webpage_content(url, max_chars=2000):
    try:
        downloaded = trafilatura.fetch_url(url)
        if downloaded:
            extracted = trafilatura.extract(downloaded, include_comments=False, include_tables=False)
            if extracted:
                return extracted[:max_chars]
        return None
    except Exception as e:
        print(f"抓取失败 {url}: {e}")
        return None

def web_search(query):
    if not bocha_api_key:
        return "搜索功能未配置：缺少 BOCHA_API_KEY"
    try:
        url = "https://api.bochaai.com/v1/web-search"
        payload = json.dumps({"query": query, "count": 3, "summary": True})
        headers = {'Authorization': f'Bearer {bocha_api_key}', 'Content-Type': 'application/json'}
        resp = requests.post(url, headers=headers, data=payload, timeout=15)
        if resp.status_code != 200:
            return f"搜索失败：{resp.status_code}"
        data = resp.json()
        web_pages = data.get('data', {}).get('webPages', {}).get('value', [])
        if not web_pages:
            return f"未找到关于「{query}」的结果"
        results = []
        for page in web_pages[:3]:
            title = page.get('name', '无标题')
            link = page.get('url', '')
            if not link:
                continue
            content = page.get('summary', '')
            if not content or len(content) < 50:
                fetched = fetch_webpage_content(link)
                content = fetched if fetched else "无法获取详细内容"
            results.append(f"### {title}\n🔗 {link}\n📄 {content}\n")
        return f"【搜索词：{query}】\n\n" + "\n".join(results)
    except Exception as e:
        return f"搜索失败：{str(e)}"

def order_item(item, price):
    return f"已模拟下单：{item}，价格{price}元。"

def get_current_time():
    now_utc = datetime.utcnow()
    now_beijing = now_utc + timedelta(hours=8)
    return now_beijing.strftime("%Y-%m-%d %H:%M:%S %A")

def switch_memory_mode(session_id, new_mode):
    session_file = get_session_file(session_id)
    if os.path.exists(session_file):
        with open(session_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = {"messages": [], "memory_mode": "chat"}
    data["memory_mode"] = new_mode
    with open(session_file, "w", encoding="utf-8") as f:
        json.dump(data, f)
    return f"已切换到 {new_mode} 记忆库。"

# ---------- 记忆操作 ----------
def store_memory(content, role, mode):
    if not content:
        return
    coll = memory_collections.get(mode)
    if not coll:
        return
    doc_id = f"{uuid.uuid4().hex}"
    obj = {"role": role, "content": content, "timestamp": datetime.now().isoformat(), "mode": mode}
    coll.add(documents=[json.dumps(obj, ensure_ascii=False)], metadatas=[{"role": role}], ids=[doc_id])

def retrieve_memories(query_text, n_results=5):
    all_mem = []
    seen = set()
    for mode, coll in memory_collections.items():
        if coll.count() == 0:
            continue
        try:
            res = coll.query(query_texts=[query_text], n_results=n_results)
            if res['documents'] and res['documents'][0]:
                for doc in res['documents'][0]:
                    if doc in seen:
                        continue
                    seen.add(doc)
                    try:
                        data = json.loads(doc)
                        all_mem.append(data)
                    except:
                        all_mem.append({"role": "unknown", "content": doc, "mode": mode})
        except:
            pass
    return all_mem

# ---------- 会话管理 ----------
SESSION_DIR = "./sessions"
os.makedirs(SESSION_DIR, exist_ok=True)

def get_session_file(session_id):
    return os.path.join(SESSION_DIR, f"{session_id}.json")

def load_session_data(session_id):
    filepath = get_session_file(session_id)
    if os.path.exists(filepath):
        try:
            with open(filepath, "r") as f:
                data = json.load(f)
                return data.get("messages", []), data.get("memory_mode", "chat")
        except:
            return [], "chat"
    return [], "chat"

def save_session_data(session_id, messages, memory_mode):
    filepath = get_session_file(session_id)
    with open(filepath, "w") as f:
        json.dump({"messages": messages, "memory_mode": memory_mode}, f)

def serialize_message(msg):
    if isinstance(msg, dict):
        return msg
    # 处理 OpenAI 对象
    return {"role": msg.role, "content": msg.content} if hasattr(msg, "role") else msg

# ---------- FastAPI ----------
app = FastAPI()

class ChatRequest(BaseModel):
    message: str
    session_id: str

@app.post("/chat")
async def chat(chat_req: ChatRequest):
    sid = chat_req.session_id
    user_msg = chat_req.message

    # 加载会话历史、当前记忆库模式
    history, mode = load_session_data(sid)

    # 检索记忆（跨所有库）
    memories = retrieve_memories(user_msg)
    memory_ctx = ""
    if memories:
        lines = []
        for m in memories[:10]:
            role_label = "用户" if m.get("role") == "user" else "AI" if m.get("role") == "assistant" else "历史"
            lines.append(f"{role_label}：{m.get('content', '')}")
        memory_ctx = "【相关记忆】\n" + "\n".join(lines)

    # 系统提示
    system = (
        f"当前时间：{get_current_time()}\n"
        f"当前记忆库模式：{mode}\n"
        "你可以调用 switch_memory_mode 切换模式。\n"
        "你可以调用 web_search 搜索、get_current_time 查看时间、write_file 写文件、order_item 模拟下单。\n"
        "回答要自然，可使用颜文字。\n\n"
        f"{memory_ctx}"
    )

    messages = [{"role": "system", "content": system}] + history + [{"role": "user", "content": user_msg}]

    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        tools=tools,
        tool_choice="auto"
    )
    resp_msg = resp.choices[0].message

    new_history = history + [{"role": "user", "content": user_msg}]
    if resp_msg.content:
        new_history.append({"role": "assistant", "content": resp_msg.content})

    # 处理工具调用
    if resp_msg.tool_calls:
        new_history.append(resp_msg)
        for tc in resp_msg.tool_calls:
            func = tc.function.name
            args = json.loads(tc.function.arguments)
            if func == "write_file":
                result = write_file(args["filename"], args["content"])
            elif func == "web_search":
                result = web_search(args["query"])
            elif func == "order_item":
                result = order_item(args["item"], args["price"])
            elif func == "get_current_time":
                result = get_current_time()
            elif func == "switch_memory_mode":
                new_mode = args["mode"]
                result = switch_memory_mode(sid, new_mode)
                mode = new_mode
            else:
                result = "未知工具"
            new_history.append({"role": "tool", "tool_call_id": tc.id, "content": result})
        # 再次调用生成最终回复
        second = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "system", "content": system}] + new_history
        )
        final = second.choices[0].message.content
        new_history.append({"role": "assistant", "content": final})
    else:
        final = resp_msg.content or ""

    # 保存会话（包含 mode）
    save_session_data(sid, new_history, mode)

    # 存储长期记忆到当前模式
    store_memory(user_msg, role="user", mode=mode)
    if final:
        store_memory(final, role="assistant", mode=mode)

    return {"reply": final}

@app.get("/history")
async def get_history(session_id: str):
    history, _ = load_session_data(session_id)
    display = [{"role": m["role"], "content": m["content"]} for m in history if m["role"] in ("user", "assistant")]
    return {"messages": display}

@app.get("/")
async def root():
    html = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>小闻 AI 助理</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { background: #F0F4F8; font-family: system-ui, sans-serif; display: flex; justify-content: center; align-items: center; min-height: 100vh; padding: 1rem; }
        .chat-container { max-width: 800px; width: 100%; background: white; border-radius: 24px; box-shadow: 0 12px 32px rgba(0,0,0,0.1); height: 85vh; display: flex; flex-direction: column; overflow: hidden; }
        .chat-header { background: #404e5b; color: white; padding: 1rem; text-align: center; }
        .chat-messages { flex: 1; overflow-y: auto; padding: 1rem; background: #fafafa; display: flex; flex-direction: column; gap: 0.8rem; }
        .message { display: flex; }
        .user { justify-content: flex-end; }
        .ai { justify-content: flex-start; }
        .bubble { max-width: 80%; padding: 0.4rem 0.8rem; line-height: 1.4; word-wrap: break-word; }
        .user .bubble { color: #8fa0b8; }
        .ai .bubble { color: #333; }
        .ai .bubble a { color: #5a7d9a; }
        .typing-indicator { display: flex; gap: 4px; padding: 0.2rem 0; }
        .typing-indicator span { width: 6px; height: 6px; background: #8fa0b8; border-radius: 50%; animation: bounce 1.4s infinite; }
        @keyframes bounce { 0%,80%,100%{ transform: scale(0); } 40%{ transform: scale(1); } }
        .input-area { display: flex; padding: 1rem; gap: 0.5rem; border-top: 1px solid #ddd; background: white; }
        textarea { flex: 1; border: 1px solid #ccc; border-radius: 0; padding: 0.7rem; font-family: inherit; resize: vertical; }
        button { background: #8fa0b8; border: none; padding: 0 1.5rem; color: white; cursor: pointer; }
        button:hover { background: #404e5b; }
    </style>
</head>
<body>
<div class="chat-container">
    <div class="chat-header"><h1>✦ 小闻 AI 助理 ✦</h1></div>
    <div class="chat-messages" id="messages"></div>
    <div class="input-area">
        <textarea id="input" rows="1" placeholder="Shift+Enter换行，Enter发送"></textarea>
        <button id="sendBtn">发送</button>
    </div>
</div>
<script>
    const sessionId = "shared_user";
    const messagesDiv = document.getElementById('messages');
    const inputEl = document.getElementById('input');
    const sendBtn = document.getElementById('sendBtn');

    function linkify(text) {
        return text.replace(/(https?:\/\/[^\s]+)/g, '<a href="$1" target="_blank">$1</a>').replace(/\n/g, '<br>');
    }

    function addMessage(role, text, isTyping = false) {
        const div = document.createElement('div');
        div.className = `message ${role}`;
        const bubble = document.createElement('div');
        bubble.className = 'bubble';
        if (isTyping) {
            bubble.innerHTML = '<div class="typing-indicator"><span></span><span></span><span></span></div>';
        } else if (role === 'ai') {
            bubble.innerHTML = linkify(text);
        } else {
            bubble.innerText = text;
        }
        div.appendChild(bubble);
        messagesDiv.appendChild(div);
        messagesDiv.scrollTop = messagesDiv.scrollHeight;
        return div;
    }

    async function loadHistory() {
        const res = await fetch(`/history?session_id=${sessionId}`);
        const data = await res.json();
        if (data.messages.length) {
            messagesDiv.innerHTML = '';
            for (const msg of data.messages) addMessage(msg.role, msg.content);
        } else {
            addMessage('ai', '你好，我是小闻。有什么可以帮你的？');
        }
    }

    let typingDiv = null;

    async function send() {
        const msg = inputEl.value.trim();
        if (!msg) return;
        addMessage('user', msg);
        inputEl.value = '';
        inputEl.style.height = 'auto';
        sendBtn.disabled = true;
        sendBtn.textContent = '发送中...';
        typingDiv = addMessage('ai', '', true);
        try {
            const res = await fetch('/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message: msg, session_id: sessionId })
            });
            const data = await res.json();
            if (typingDiv) typingDiv.remove();
            addMessage('ai', data.reply || '抱歉，没有收到回复。');
        } catch (err) {
            if (typingDiv) typingDiv.remove();
            addMessage('ai', '网络错误，请稍后再试。');
        } finally {
            sendBtn.disabled = false;
            sendBtn.textContent = '发送';
        }
    }

    inputEl.addEventListener('keydown', e => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            send();
        }
    });
    sendBtn.addEventListener('click', send);
    inputEl.addEventListener('input', function() { this.style.height = 'auto'; this.style.height = Math.min(150, this.scrollHeight) + 'px'; });
    loadHistory();
</script>
</body>
</html>"""
    return HTMLResponse(content=html)

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)import os
import json
import uuid
import requests
from datetime import datetime, timedelta
from openai import OpenAI
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn
import trafilatura
from ddgs import DDGS  # 备用，但主搜索使用博查

load_dotenv()
api_key = os.getenv("DEEPSEEK_API_KEY")
bocha_api_key = os.getenv("BOCHA_API_KEY")
client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

# ---------- ChromaDB 多集合 ----------
chroma_client = chromadb.PersistentClient(path="./my_ai_memory")
embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

COLLECTIONS = {
    "chat": "chat_memory",
    "mail": "mail_memory",
    "bill": "bill_memory",
    "test": "test_memory"
}
memory_collections = {}
for key, name in COLLECTIONS.items():
    memory_collections[key] = chroma_client.get_or_create_collection(
        name=name, embedding_function=embed_fn
    )

# ---------- 工具定义 ----------
tools = [
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "将内容写入服务器上的文件。",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {"type": "string"},
                    "content": {"type": "string"}
                },
                "required": ["filename", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "搜索互联网信息（博查 API），返回标题、链接和正文摘要。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "order_item",
            "description": "模拟下单购买商品。",
            "parameters": {
                "type": "object",
                "properties": {
                    "item": {"type": "string"},
                    "price": {"type": "number"}
                },
                "required": ["item", "price"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "获取当前的准确北京时间。",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "switch_memory_mode",
            "description": "切换当前使用的记忆库模式（chat/mail/bill/test）。",
            "parameters": {
                "type": "object",
                "properties": {
                    "mode": {"type": "string", "enum": ["chat", "mail", "bill", "test"]}
                },
                "required": ["mode"]
            }
        }
    }
]

# ---------- 工具实现 ----------
def write_file(filename, content):
    with open(filename, "w", encoding="utf-8") as f:
        f.write(content)
    return f"文件 {filename} 已写入成功"

def fetch_webpage_content(url, max_chars=2000):
    try:
        downloaded = trafilatura.fetch_url(url)
        if downloaded:
            extracted = trafilatura.extract(downloaded, include_comments=False, include_tables=False)
            if extracted:
                return extracted[:max_chars]
        return None
    except Exception as e:
        print(f"抓取失败 {url}: {e}")
        return None

def web_search(query):
    if not bocha_api_key:
        return "搜索功能未配置：缺少 BOCHA_API_KEY"
    try:
        url = "https://api.bochaai.com/v1/web-search"
        payload = json.dumps({"query": query, "count": 3, "summary": True})
        headers = {'Authorization': f'Bearer {bocha_api_key}', 'Content-Type': 'application/json'}
        resp = requests.post(url, headers=headers, data=payload, timeout=15)
        if resp.status_code != 200:
            return f"搜索失败：{resp.status_code}"
        data = resp.json()
        web_pages = data.get('data', {}).get('webPages', {}).get('value', [])
        if not web_pages:
            return f"未找到关于「{query}」的结果"
        results = []
        for page in web_pages[:3]:
            title = page.get('name', '无标题')
            link = page.get('url', '')
            if not link:
                continue
            content = page.get('summary', '')
            if not content or len(content) < 50:
                fetched = fetch_webpage_content(link)
                content = fetched if fetched else "无法获取详细内容"
            results.append(f"### {title}\n🔗 {link}\n📄 {content}\n")
        return f"【搜索词：{query}】\n\n" + "\n".join(results)
    except Exception as e:
        return f"搜索失败：{str(e)}"

def order_item(item, price):
    return f"已模拟下单：{item}，价格{price}元。"

def get_current_time():
    now_utc = datetime.utcnow()
    now_beijing = now_utc + timedelta(hours=8)
    return now_beijing.strftime("%Y-%m-%d %H:%M:%S %A")

def switch_memory_mode(session_id, new_mode):
    session_file = get_session_file(session_id)
    if os.path.exists(session_file):
        with open(session_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = {"messages": [], "memory_mode": "chat"}
    data["memory_mode"] = new_mode
    with open(session_file, "w", encoding="utf-8") as f:
        json.dump(data, f)
    return f"已切换到 {new_mode} 记忆库。"

# ---------- 记忆操作 ----------
def store_memory(content, role, mode):
    if not content:
        return
    coll = memory_collections.get(mode)
    if not coll:
        return
    doc_id = f"{uuid.uuid4().hex}"
    obj = {"role": role, "content": content, "timestamp": datetime.now().isoformat(), "mode": mode}
    coll.add(documents=[json.dumps(obj, ensure_ascii=False)], metadatas=[{"role": role}], ids=[doc_id])

def retrieve_memories(query_text, n_results=5):
    all_mem = []
    seen = set()
    for mode, coll in memory_collections.items():
        if coll.count() == 0:
            continue
        try:
            res = coll.query(query_texts=[query_text], n_results=n_results)
            if res['documents'] and res['documents'][0]:
                for doc in res['documents'][0]:
                    if doc in seen:
                        continue
                    seen.add(doc)
                    try:
                        data = json.loads(doc)
                        all_mem.append(data)
                    except:
                        all_mem.append({"role": "unknown", "content": doc, "mode": mode})
        except:
            pass
    return all_mem

# ---------- 会话管理 ----------
SESSION_DIR = "./sessions"
os.makedirs(SESSION_DIR, exist_ok=True)

def get_session_file(session_id):
    return os.path.join(SESSION_DIR, f"{session_id}.json")

def load_session_data(session_id):
    filepath = get_session_file(session_id)
    if os.path.exists(filepath):
        try:
            with open(filepath, "r") as f:
                data = json.load(f)
                return data.get("messages", []), data.get("memory_mode", "chat")
        except:
            return [], "chat"
    return [], "chat"

def save_session_data(session_id, messages, memory_mode):
    filepath = get_session_file(session_id)
    with open(filepath, "w") as f:
        json.dump({"messages": messages, "memory_mode": memory_mode}, f)

def serialize_message(msg):
    if isinstance(msg, dict):
        return msg
    # 处理 OpenAI 对象
    return {"role": msg.role, "content": msg.content} if hasattr(msg, "role") else msg

# ---------- FastAPI ----------
app = FastAPI()

class ChatRequest(BaseModel):
    message: str
    session_id: str

@app.post("/chat")
async def chat(chat_req: ChatRequest):
    sid = chat_req.session_id
    user_msg = chat_req.message

    # 加载会话历史、当前记忆库模式
    history, mode = load_session_data(sid)

    # 检索记忆（跨所有库）
    memories = retrieve_memories(user_msg)
    memory_ctx = ""
    if memories:
        lines = []
        for m in memories[:10]:
            role_label = "用户" if m.get("role") == "user" else "AI" if m.get("role") == "assistant" else "历史"
            lines.append(f"{role_label}：{m.get('content', '')}")
        memory_ctx = "【相关记忆】\n" + "\n".join(lines)

    # 系统提示
    system = (
        f"当前时间：{get_current_time()}\n"
        f"当前记忆库模式：{mode}\n"
        "你可以调用 switch_memory_mode 切换模式。\n"
        "你可以调用 web_search 搜索、get_current_time 查看时间、write_file 写文件、order_item 模拟下单。\n"
        "回答要自然，可使用颜文字。\n\n"
        f"{memory_ctx}"
    )

    messages = [{"role": "system", "content": system}] + history + [{"role": "user", "content": user_msg}]

    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        tools=tools,
        tool_choice="auto"
    )
    resp_msg = resp.choices[0].message

    new_history = history + [{"role": "user", "content": user_msg}]
    if resp_msg.content:
        new_history.append({"role": "assistant", "content": resp_msg.content})

    # 处理工具调用
    if resp_msg.tool_calls:
        new_history.append(resp_msg)
        for tc in resp_msg.tool_calls:
            func = tc.function.name
            args = json.loads(tc.function.arguments)
            if func == "write_file":
                result = write_file(args["filename"], args["content"])
            elif func == "web_search":
                result = web_search(args["query"])
            elif func == "order_item":
                result = order_item(args["item"], args["price"])
            elif func == "get_current_time":
                result = get_current_time()
            elif func == "switch_memory_mode":
                new_mode = args["mode"]
                result = switch_memory_mode(sid, new_mode)
                mode = new_mode
            else:
                result = "未知工具"
            new_history.append({"role": "tool", "tool_call_id": tc.id, "content": result})
        # 再次调用生成最终回复
        second = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "system", "content": system}] + new_history
        )
        final = second.choices[0].message.content
        new_history.append({"role": "assistant", "content": final})
    else:
        final = resp_msg.content or ""

    # 保存会话（包含 mode）
    save_session_data(sid, new_history, mode)

    # 存储长期记忆到当前模式
    store_memory(user_msg, role="user", mode=mode)
    if final:
        store_memory(final, role="assistant", mode=mode)

    return {"reply": final}

@app.get("/history")
async def get_history(session_id: str):
    history, _ = load_session_data(session_id)
    display = [{"role": m["role"], "content": m["content"]} for m in history if m["role"] in ("user", "assistant")]
    return {"messages": display}

@app.get("/")
async def root():
    html = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>小闻 AI 助理</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { background: #F0F4F8; font-family: system-ui, sans-serif; display: flex; justify-content: center; align-items: center; min-height: 100vh; padding: 1rem; }
        .chat-container { max-width: 800px; width: 100%; background: white; border-radius: 24px; box-shadow: 0 12px 32px rgba(0,0,0,0.1); height: 85vh; display: flex; flex-direction: column; overflow: hidden; }
        .chat-header { background: #404e5b; color: white; padding: 1rem; text-align: center; }
        .chat-messages { flex: 1; overflow-y: auto; padding: 1rem; background: #fafafa; display: flex; flex-direction: column; gap: 0.8rem; }
        .message { display: flex; }
        .user { justify-content: flex-end; }
        .ai { justify-content: flex-start; }
        .bubble { max-width: 80%; padding: 0.4rem 0.8rem; line-height: 1.4; word-wrap: break-word; }
        .user .bubble { color: #8fa0b8; }
        .ai .bubble { color: #333; }
        .ai .bubble a { color: #5a7d9a; }
        .typing-indicator { display: flex; gap: 4px; padding: 0.2rem 0; }
        .typing-indicator span { width: 6px; height: 6px; background: #8fa0b8; border-radius: 50%; animation: bounce 1.4s infinite; }
        @keyframes bounce { 0%,80%,100%{ transform: scale(0); } 40%{ transform: scale(1); } }
        .input-area { display: flex; padding: 1rem; gap: 0.5rem; border-top: 1px solid #ddd; background: white; }
        textarea { flex: 1; border: 1px solid #ccc; border-radius: 0; padding: 0.7rem; font-family: inherit; resize: vertical; }
        button { background: #8fa0b8; border: none; padding: 0 1.5rem; color: white; cursor: pointer; }
        button:hover { background: #404e5b; }
    </style>
</head>
<body>
<div class="chat-container">
    <div class="chat-header"><h1>✦ 小闻 AI 助理 ✦</h1></div>
    <div class="chat-messages" id="messages"></div>
    <div class="input-area">
        <textarea id="input" rows="1" placeholder="Shift+Enter换行，Enter发送"></textarea>
        <button id="sendBtn">发送</button>
    </div>
</div>
<script>
    const sessionId = "shared_user";
    const messagesDiv = document.getElementById('messages');
    const inputEl = document.getElementById('input');
    const sendBtn = document.getElementById('sendBtn');

    function linkify(text) {
        return text.replace(/(https?:\/\/[^\s]+)/g, '<a href="$1" target="_blank">$1</a>').replace(/\n/g, '<br>');
    }

    function addMessage(role, text, isTyping = false) {
        const div = document.createElement('div');
        div.className = `message ${role}`;
        const bubble = document.createElement('div');
        bubble.className = 'bubble';
        if (isTyping) {
            bubble.innerHTML = '<div class="typing-indicator"><span></span><span></span><span></span></div>';
        } else if (role === 'ai') {
            bubble.innerHTML = linkify(text);
        } else {
            bubble.innerText = text;
        }
        div.appendChild(bubble);
        messagesDiv.appendChild(div);
        messagesDiv.scrollTop = messagesDiv.scrollHeight;
        return div;
    }

    async function loadHistory() {
        const res = await fetch(`/history?session_id=${sessionId}`);
        const data = await res.json();
        if (data.messages.length) {
            messagesDiv.innerHTML = '';
            for (const msg of data.messages) addMessage(msg.role, msg.content);
        } else {
            addMessage('ai', '你好，我是小闻。有什么可以帮你的？');
        }
    }

    let typingDiv = null;

    async function send() {
        const msg = inputEl.value.trim();
        if (!msg) return;
        addMessage('user', msg);
        inputEl.value = '';
        inputEl.style.height = 'auto';
        sendBtn.disabled = true;
        sendBtn.textContent = '发送中...';
        typingDiv = addMessage('ai', '', true);
        try {
            const res = await fetch('/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message: msg, session_id: sessionId })
            });
            const data = await res.json();
            if (typingDiv) typingDiv.remove();
            addMessage('ai', data.reply || '抱歉，没有收到回复。');
        } catch (err) {
            if (typingDiv) typingDiv.remove();
            addMessage('ai', '网络错误，请稍后再试。');
        } finally {
            sendBtn.disabled = false;
            sendBtn.textContent = '发送';
        }
    }

    inputEl.addEventListener('keydown', e => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            send();
        }
    });
    sendBtn.addEventListener('click', send);
    inputEl.addEventListener('input', function() { this.style.height = 'auto'; this.style.height = Math.min(150, this.scrollHeight) + 'px'; });
    loadHistory();
</script>
</body>
</html>"""
    return HTMLResponse(content=html)

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)