"""酒蔵AI広報アシスタント：API接続モード用の薄いバックエンド

    pip install -r requirements.txt
    export ANTHROPIC_API_KEY=...
    uvicorn server:app --port 8000

http://localhost:8000/?mode=api を開くと、画面の「API」モードで Claude が文面を作成します。
発表本番は index.html をそのまま開き、ダミーモードで動かしてください。
"""

import json
from pathlib import Path

import anthropic
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

MODEL = "claude-opus-5"

app = FastAPI()
# index.html を file:// で開いた場合にも呼べるようにする
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["POST"], allow_headers=["*"])
client = anthropic.Anthropic()


class Profile(BaseModel):
    name: str
    pref: str = ""
    brand: str = ""
    feature: str = ""


class GenerateRequest(BaseModel):
    profile: Profile
    memo: str
    audience: str
    language: str = "ja"  # "ja" | "en"
    format: str = "ig"  # "ig" | "x"


OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "caption": {"type": "string"},
        "hashtags": {"type": "array", "items": {"type": "string"}},
        "banner_text": {"type": "string"},
    },
    "required": ["caption", "hashtags", "banner_text"],
    "additionalProperties": False,
}

SYSTEM = """あなたは小規模な日本酒蔵の広報担当者です。
蔵人の短いメモから、指定された相手に届くSNS投稿を1つ作ります。

- メモにない事実（日付・価格・受賞歴など）は足さないでください。
- 絵文字は使わないでください。
- banner_text は投稿画像に重ねるキャッチコピーです。「1行目／2行目」の形で、1行目は15字前後の印象的な一言、2行目は要件（例：ひやおろし 出荷開始）にしてください。
- hashtags は "#" 付きで3〜6個。"""

FORMAT_RULES = {
    "ig": "Instagram投稿。改行と余白を活かし、3〜8行程度のキャプションにしてください。",
    "x": "X投稿。caption とハッシュタグを半角スペースでつないだ全体が140字以内（英語なら280字以内）に収まるようにしてください。",
}


@app.get("/")
def index():
    return FileResponse(Path(__file__).with_name("index.html"))


@app.post("/generate")
def generate(req: GenerateRequest):
    p = req.profile
    lang = "英語" if req.language == "en" else "日本語"
    prompt = f"""# 蔵の情報
蔵名：{p.name}
所在地：{p.pref}
代表銘柄：{p.brand}
特徴：{p.feature}

# 蔵人のメモ
{req.memo}

# 届けたい相手
{req.audience}

# 出力形式
{FORMAT_RULES.get(req.format, FORMAT_RULES["ig"])}
投稿は{lang}で書いてください（banner_text も{lang}）。相手に合わせて語り口と伝える要素を選んでください。"""

    try:
        response = client.beta.messages.create(
            model=MODEL,
            max_tokens=4000,
            system=SYSTEM,
            messages=[{"role": "user", "content": prompt}],
            output_config={"effort": "low", "format": {"type": "json_schema", "schema": OUTPUT_SCHEMA}},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
    except anthropic.APIError as e:
        raise HTTPException(status_code=502, detail=str(e))

    if response.stop_reason in ("refusal", "max_tokens"):
        raise HTTPException(status_code=502, detail=f"generation stopped: {response.stop_reason}")

    text = next((b.text for b in response.content if b.type == "text"), "")
    return json.loads(text)
