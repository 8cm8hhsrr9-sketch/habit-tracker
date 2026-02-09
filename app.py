# app.py
import json
from datetime import date, timedelta
from typing import Optional, Dict, Any, Tuple

import requests
import pandas as pd
import streamlit as st


# =========================
# 기본 설정
# =========================
st.set_page_config(page_title="AI 습관 트래커", page_icon="📊", layout="wide")

st.title("📊 AI 습관 트래커")
st.caption("오늘의 습관 체크인 → 달성률/차트 → 날씨/강아지 → AI 코치 리포트")


# =========================
# 사이드바: API 키 입력
# =========================
with st.sidebar:
    st.header("🔑 API Keys")
    openai_api_key = st.text_input("OpenAI API Key", type="password", placeholder="sk-...")
    weather_api_key = st.text_input("OpenWeatherMap API Key", type="password", placeholder="키를 입력하세요")
    st.divider()
    st.markdown("✅ 키는 브라우저 세션에만 사용되며 저장하지 않습니다.")


# =========================
# 유틸 / API 함수
# =========================
def safe_get_json(url: str, params: Optional[dict] = None, timeout: int = 10) -> Optional[dict]:
    try:
        r = requests.get(url, params=params, timeout=timeout)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:
        return None


def get_weather(city: str, api_key: str) -> Optional[Dict[str, Any]]:
    """
    OpenWeatherMap에서 현재 날씨를 가져옵니다.
    - 한국어(lang=kr), 섭씨(units=metric)
    - 실패 시 None
    """
    if not api_key:
        return None

    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {
        "q": city,
        "appid": api_key,
        "lang": "kr",
        "units": "metric",
    }
    data = safe_get_json(url, params=params, timeout=10)
    if not data or "weather" not in data or "main" not in data:
        return None

    try:
        desc = data["weather"][0].get("description")
        main = data["main"]
        return {
            "city": city,
            "description": desc,
            "temp_c": float(main.get("temp")),
            "feels_like_c": float(main.get("feels_like")),
            "humidity": int(main.get("humidity")),
        }
    except Exception:
        return None


def get_dog_image() -> Optional[Dict[str, str]]:
    """
    Dog CEO에서 랜덤 강아지 사진 URL과 품종을 가져옵니다.
    - 실패 시 None
    """
    data = safe_get_json("https://dog.ceo/api/breeds/image/random", timeout=10)
    if not data or data.get("status") != "success":
        return None

    try:
        url = data["message"]
        # URL 예: https://images.dog.ceo/breeds/hound-afghan/n02088094_1003.jpg
        breed_part = url.split("/breeds/")[1].split("/")[0]  # hound-afghan
        # 하이픈/슬래시 처리
        breed = breed_part.replace("-", " ").replace("_", " ").title()
        return {"url": url, "breed": breed}
    except Exception:
        return None


def _build_system_prompt(style: str) -> str:
    base = (
        "당신은 'AI 습관 트래커'의 코치입니다. 사용자의 오늘 습관/기분/날씨/강아지 품종 정보를 바탕으로 "
        "짧지만 임팩트 있게 한국어로 코칭하세요. 과장된 의학/정신건강 진단은 금지하고, 실행 가능한 행동을 제안하세요.\n\n"
        "출력 형식은 반드시 아래 5개 섹션을 지키고, 각 섹션을 굵은 제목으로 시작하세요:\n"
        "1) **컨디션 등급(S~D)**: 한 줄\n"
        "2) **습관 분석**: 3~6줄\n"
        "3) **날씨 코멘트**: 1~3줄\n"
        "4) **내일 미션**: 체크리스트 3개\n"
        "5) **오늘의 한마디**: 한 줄\n"
    )

    if style == "스파르타 코치":
        return base + "\n스타일: 엄격하고 단호하게. 핑계 차단, 숫자/팩트 중심. 짧고 강하게."
    if style == "따뜻한 멘토":
        return base + "\n스타일: 따뜻하고 공감적으로. 작은 성취를 인정하고 부드럽게 다음 행동을 이끈다."
    # 게임 마스터
    return base + "\n스타일: RPG 게임 마스터 톤. 퀘스트/경험치/레벨업 같은 표현을 활용하되 과도한 설정은 금지."


def generate_report(
    openai_api_key: str,
    coach_style: str,
    payload: Dict[str, Any],
) -> Optional[str]:
    """
    습관+기분+날씨+강아지 품종을 모아서 OpenAI에 전달해 리포트를 생성합니다.
    - 모델: gpt-5-mini
    - 실패 시 None
    """
    if not openai_api_key:
        return None

    system_prompt = _build_system_prompt(coach_style)

    # 모델 입력(요약 JSON + 자연어)
    user_input = (
        "아래는 사용자 오늘 데이터입니다. 이를 바탕으로 코칭 리포트를 작성하세요.\n\n"
        f"데이터(JSON):\n{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )

    # OpenAI Python SDK (v1) 우선, 실패 시 REST로 간단 폴백
    try:
        from openai import OpenAI  # type: ignore

        client = OpenAI(api_key=openai_api_key)

        # Responses API 우선 시도
        try:
            resp = client.responses.create(
                model="gpt-5-mini",
                instructions=system_prompt,
                input=user_input,
            )
            text = getattr(resp, "output_text", None)
            if text:
                return text.strip()
        except Exception:
            pass

        # Chat Completions 폴백(환경에 따라 가능)
        try:
            resp = client.chat.completions.create(
                model="gpt-5-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_input},
                ],
            )
            return resp.choices[0].message.content.strip()
        except Exception:
            return None

    except Exception:
        # SDK가 없거나 깨졌다면: 안전하게 None
        return None


# =========================
# 세션 상태: 6일 샘플 + 오늘
# =========================
HABITS = [
    ("wake", "🌅", "기상 미션"),
    ("water", "💧", "물 마시기"),
    ("study", "📚", "공부/독서"),
    ("workout", "🏃", "운동하기"),
    ("sleep", "😴", "수면"),
]

CITIES = ["Seoul", "Busan", "Incheon", "Daegu", "Daejeon", "Gwangju", "Ulsan", "Suwon", "Jeju", "Gimhae"]
COACH_STYLES = ["스파르타 코치", "따뜻한 멘토", "게임 마스터"]


def _init_demo_history() -> list:
    # 6일 샘플: (오늘-6) ~ (오늘-1)
    # 습관/기분은 데모 랜덤 느낌으로 고정 패턴
    demo = []
    today = date.today()
    patterns = [
        ([1, 1, 0, 0, 1], 6),
        ([1, 0, 1, 0, 1], 7),
        ([1, 1, 1, 0, 0], 5),
        ([0, 1, 1, 1, 0], 8),
        ([1, 1, 1, 1, 0], 9),
        ([0, 0, 1, 1, 1], 7),
    ]
    for i in range(6, 0, -1):
        d = today - timedelta(days=i)
        checks, mood = patterns[6 - i]
        entry = {"date": d.isoformat(), "mood": mood}
        for (key, _, _), c in zip(HABITS, checks):
            entry[key] = bool(c)
        demo.append(entry)

    # 오늘(초기값)
    entry_today = {"date": today.isoformat(), "mood": 6}
    for key, _, _ in HABITS:
        entry_today[key] = False
    demo.append(entry_today)
    return demo


if "history" not in st.session_state:
    st.session_state.history = _init_demo_history()

if "last_report" not in st.session_state:
    st.session_state.last_report = None

if "last_weather" not in st.session_state:
    st.session_state.last_weather = None

if "last_dog" not in st.session_state:
    st.session_state.last_dog = None

if "coach_style" not in st.session_state:
    st.session_state.coach_style = COACH_STYLES[0]


def _get_today_entry() -> dict:
    today_str = date.today().isoformat()
    for e in st.session_state.history:
        if e.get("date") == today_str:
            return e
    # 없으면 추가
    entry_today = {"date": today_str, "mood": 6}
    for key, _, _ in HABITS:
        entry_today[key] = False
    st.session_state.history.append(entry_today)
    # 최근 7개 유지
    st.session_state.history = st.session_state.history[-7:]
    return entry_today


today_entry = _get_today_entry()


# =========================
# 메인 레이아웃
# =========================
left, right = st.columns([1.05, 0.95], gap="large")

with left:
    st.subheader("✅ 오늘의 습관 체크인")

    # 체크박스 5개를 2열 배치 (2열에 2개씩, 마지막은 왼쪽에)
    col1, col2 = st.columns(2)
    # 1,2 -> col1/col2
    (k1, e1, t1), (k2, e2, t2) = HABITS[0], HABITS[1]
    today_entry[k1] = col1.checkbox(f"{e1} {t1}", value=bool(today_entry.get(k1, False)))
    today_entry[k2] = col2.checkbox(f"{e2} {t2}", value=bool(today_entry.get(k2, False)))

    # 3,4 -> col1/col2
    col1, col2 = st.columns(2)
    (k3, e3, t3), (k4, e4, t4) = HABITS[2], HABITS[3]
    today_entry[k3] = col1.checkbox(f"{e3} {t3}", value=bool(today_entry.get(k3, False)))
    today_entry[k4] = col2.checkbox(f"{e4} {t4}", value=bool(today_entry.get(k4, False)))

    # 5 -> col1
    col1, col2 = st.columns(2)
    (k5, e5, t5) = HABITS[4]
    today_entry[k5] = col1.checkbox(f"{e5} {t5}", value=bool(today_entry.get(k5, False)))
    col2.write("")

    mood = st.slider("🙂 기분 (1~10)", min_value=1, max_value=10, value=int(today_entry.get("mood", 6)))
    today_entry["mood"] = mood

    city = st.selectbox("🏙️ 도시 선택", CITIES, index=CITIES.index("Seoul") if "Seoul" in CITIES else 0)
    coach_style = st.radio("🧑‍🏫 코치 스타일", COACH_STYLES, horizontal=True, index=COACH_STYLES.index(st.session_state.coach_style))
    st.session_state.coach_style = coach_style

    # session_state 반영(오늘 엔트리 업데이트 + 최근 7개 유지)
    def _save_today(entry: dict):
        today_str = date.today().isoformat()
        replaced = False
        for i, e in enumerate(st.session_state.history):
            if e.get("date") == today_str:
                st.session_state.history[i] = entry
                replaced = True
                break
        if not replaced:
            st.session_state.history.append(entry)
        st.session_state.history = st.session_state.history[-7:]


    _save_today(today_entry)

    # 달성률 계산
    checked_count = sum(1 for key, _, _ in HABITS if today_entry.get(key))
    total = len(HABITS)
    achievement = round((checked_count / total) * 100)

    st.divider()
    st.subheader("📈 오늘 요약")

    m1, m2, m3 = st.columns(3)
    m1.metric("달성률", f"{achievement}%")
    m2.metric("달성 습관", f"{checked_count}/{total}")
    m3.metric("기분", f"{mood}/10")

    # 7일 바 차트
    df = pd.DataFrame(st.session_state.history).copy()
    # 안전: 없을 수 있는 컬럼 채우기
    for key, _, _ in HABITS:
        if key not in df.columns:
            df[key] = False
    if "mood" not in df.columns:
        df["mood"] = 6

    df["achieved"] = df[[k for k, _, _ in HABITS]].sum(axis=1)
    df["achievement_pct"] = (df["achieved"] / total * 100).round(0).astype(int)
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%m/%d")

    st.caption("🧪 데모용 6일 샘플 + 오늘 데이터 = 7일 차트 (세션에 저장)")
    st.bar_chart(df.set_index("date")["achievement_pct"], height=220)


with right:
    st.subheader("🧠 AI 코치 리포트")

    # API 데이터 가져오기(버튼 누를 때)
    generate = st.button("🚀 컨디션 리포트 생성", use_container_width=True)

    if generate:
        # 날씨 / 강아지
        weather = get_weather(city, weather_api_key)
        dog = get_dog_image()

        st.session_state.last_weather = weather
        st.session_state.last_dog = dog

        payload = {
            "date": date.today().isoformat(),
            "city": city,
            "coach_style": coach_style,
            "mood": mood,
            "habits": {label: bool(today_entry[key]) for key, _, label in HABITS},
            "achievement_pct": achievement,
            "weather": weather,
            "dog": dog,
        }

        report = generate_report(openai_api_key, coach_style, payload)
        st.session_state.last_report = report

    # 표시(최근 결과)
    weather = st.session_state.last_weather
    dog = st.session_state.last_dog
    report = st.session_state.last_report

    # 날씨 + 강아지 카드 (2열)
    c1, c2 = st.columns(2, gap="medium")

    with c1:
        st.markdown("#### 🌦️ 오늘 날씨")
        if weather:
            st.write(f"**도시:** {weather['city']}")
            st.write(f"**상태:** {weather['description']}")
            st.write(f"**기온:** {weather['temp_c']:.1f}°C (체감 {weather['feels_like_c']:.1f}°C)")
            st.write(f"**습도:** {weather['humidity']}%")
        else:
            st.info("날씨 정보를 가져오지 못했어요. (API 키/도시/네트워크 확인)")

    with c2:
        st.markdown("#### 🐶 오늘의 강아지")
        if dog:
            st.image(dog["url"], use_container_width=True)
            st.caption(f"품종: **{dog['breed']}**")
        else:
            st.info("강아지 이미지를 가져오지 못했어요. (네트워크 확인)")

    st.divider()

    # AI 리포트
    st.markdown("#### 📝 리포트")
    if report:
        st.markdown(report)
    else:
        st.warning("아직 리포트가 없어요. 위 버튼을 눌러 생성해보세요. (OpenAI API Key 필요)")

    # 공유용 텍스트
    st.divider()
    st.markdown("#### 📣 공유용 텍스트")
    share_text = (
        f"📊 AI 습관 트래커 ({date.today().isoformat()})\n"
        f"- 도시: {city}\n"
        f"- 코치: {coach_style}\n"
        f"- 달성률: {achievement}% ({checked_count}/{total})\n"
        f"- 기분: {mood}/10\n"
        f"- 체크: " + ", ".join([f"{emoji}{label}" for (key, emoji, label) in HABITS if today_entry.get(key)]) +
        ("\n\n📝 리포트\n" + report.strip() if report else "\n\n📝 리포트\n(아직 생성 전)")
    )
    st.code(share_text, language="text")


# =========================
# 하단 API 안내
# =========================
with st.expander("ℹ️ API 안내 / 설정 방법"):
    st.markdown(
        """
**1) OpenAI API Key**
- OpenAI 대시보드에서 발급한 키를 사이드바에 입력하세요.
- 모델은 `gpt-5-mini`를 사용합니다.

**2) OpenWeatherMap API Key**
- OpenWeatherMap에서 API Key를 발급받아 사이드바에 입력하세요.
- 현재 날씨를 `한국어(lang=kr)`, `섭씨(units=metric)`로 조회합니다.

**3) Dog CEO API**
- 키 없이 무료로 랜덤 강아지 이미지를 가져옵니다.

**문제 해결**
- 날씨가 안 나오면: OpenWeatherMap 키, 도시 이름(영문), 네트워크를 확인하세요.
- 리포트가 안 나오면: OpenAI 키가 올바른지, 계정/모델 접근 권한을 확인하세요.
        """.strip()
    )
