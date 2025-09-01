## Ai Summarizer - Deep Dive...
## Any kind of youtube video, or webpage, url, pdf uploaded in this model will be text-summarized using llms
'''
Project: OmniBrief (AI Summarizer)
Goal = Summarize content from a URL (youtube, websitem pdf)
What this can teach:
1- How to build streamlit advace quick UI
2- Loading real world content(youtube, websitem pdf)
3- Chunking long text and running a map-reduced summarization chain
4- using groq llms using langchain in a safe way
'''

#Imports
import os, re, json, tempfile
from urllib.parse import urlparse

#Network and validation
import requests # to fetch web/pdf/caption files
import validators #for validating urls inputs

# UI framework
import streamlit as st
from validators import url as is_url

#langcahin core pieces
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain #summarization chain loader (Chain and then summarize)

#Loaders
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader, YoutubeLoader, UnstructuredURLLoader #More loaders are also available, check later

#LLM
from langchain_groq import ChatGroq

#Youtube caption edge case and fallback
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound, VideoUnavailable
from yt_dlp import YoutubeDL  ## If transcription of video is not available, then it will summarize the video based on human captions.. means it will take the voice of the narrator of video and make captions and them the summary

#Streamlit
st.set_page_config(page_title="OmniBrief (AI Summarizer)", page_icon="🎯")

# Styling and designing
# --- Custom CSS for Professional Look ---

st.markdown(
    """
    <style>
    /* Text Input */
    div[data-testid="stTextInput"] input {
        border: 2px solid #555; /* Neutral dark border */
        border-radius: 4px;
        padding: 8px;
    }

    /* File Uploader */
    div[data-testid="stFileUploader"] {
        border: 2px solid #555; /* Neutral dark border */
        border-radius: 4px;
        padding: 8px;
    }

    /* Dropdown (Selectbox) */
        border: 2px solid #555; /* Neutral dark border */
        border: 2px solid red;
        border-radius: 4px;
    }

    /* Sliders */
    div[data-testid="stSlider"] {
        border: 2px solid #555; /* Neutral dark border */
        border-radius: 4px;
        padding: 6px;
    }

    /* Buttons */
    button[kind="primary"], button[kind="secondary"] {
        border: 2px solid #555; /* Neutral dark border */
        border-radius: 4px;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #D2B48C, #C19A6B); /* Lighter chocolate gradient */
        color: #3E2C23; /* Dark brown text for readability */
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #D3D3D3; /* Pure white sidebar */
        border-right: 1px solid #e0e0e0;
    }

    /* Sidebar Hover Effects */
    [data-testid="stSidebar"] .stSelectbox:hover,
    [data-testid="stSidebar"] .stSlider:hover {
        transform: scale(1.02);
        transition: transform 0.2s ease-in-out;
        background-color: #f1f1f1;
        border-radius: 6px;
    }

    /* Buttons */
    div.stButton > button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px 20px;
        font-size: 16px;
        transition: background-color 0.3s ease, transform 0.2s;
    }
    div.stButton > button:hover {
        background-color: #45a049;
        transform: scale(1.05);
    }
    </style>
    """,
    unsafe_allow_html=True
)



#transitions in title

st.markdown(
    """
    <style>
    .custom-title {
        padding: 14px 24px;
        border-radius: 14px;
        text-align: center;
        background: linear-gradient(90deg, #5A4FCF, #A78BFA);
        color: #FFFFFF;
        font-size: 28px;
        font-weight: bold;
        box-shadow: 0px 4px 20px rgba(90, 79, 207, 0.4);
        transition: transform 0.2s ease-in-out;
    }
    .custom-caption {
        font-size: 14px;
        color: #F0EFFF;
        margin-top: 6px;
        display: block;
    }
    .custom-title:hover {
        transform: scale(1.05);
    }
    </style>
    <div class="custom-title">
        ⏱ OmniBrief - Summarize URLs, Youtube, PDFs
        <span class="custom-caption">Built with Streamlit + Langchain + Groq</span>
    </div>
    """,
    unsafe_allow_html=True
)


#Sidebar
# LLM model, temperature, target, target length, etc

with st.sidebar:
    st.subheader("**🗝 API and model**")
    groq_api_key = st.text_input("**GROQ_API_KEY**",type="password", value=os.getenv("GROQ_API_KEY",""), placeholder="Please enter your API Key here")

    model = st.selectbox(
    "**Groq Model**",
    ["gemma2-9b-it","deepseek-r1-distill-llama-70b","llama-3.1-8b-instant"],
    index=0,
    help="If you get 'Model not found', update this id to a valid groq model"
    )
    custom_model = st.text_input("**Custom model (optional)**", help="Override selection above if filled")

    st.subheader("🔥 **Generation**")
    temperature=st.slider("**Temperature(Creativity)**", 0.0,1.0,0.2,0.05)
    out_len = st.slider("**Target summary length (words)**", 90,800, 300, 20)


    # Customizing for the client, making it easy and interactive
    st.subheader("✍🏻 Style")
    out_style = st.selectbox("Output style",["Bullets","Paragraph","Both"]) #How the client wants his answer
    tone = st.selectbox("Tone", ["Neutral","Formal","Casual","Executive Brief"])
    out_lang = st.selectbox("Language",["English","Urdu","Roman Urdu","Auto"])

    st.subheader("⚙ Processing")
    chain_mode=st.radio("Chain type", ["Auto","Stuff","Map-reduce"],index=0)
    chunk_size = st.slider("Chunk Size (characters)", 500,4000,1600,20)
    chunk_overlap= st.slider("Chunk Overlap (characters)",0,800,150,10)
    max_map_chunks=st.slider("Max Chunks(for combine steps)",9,64,28,1)

    st.subheader("📃 Extras ")
    show_preview = st.checkbox('Show source preview', value=True)
    want_outline = st.checkbox("Also produce an outline",value=True)
    want_keywords = st.checkbox("Also extracts keywords & hashtags",value=True)

#Main Input
st.markdown("<div style='margin-top: 25px;'></div>", unsafe_allow_html=True)  ## For proper spacing b/w title and insertions below

left,right = st.columns([2,1])

with left:
    url = st.text_input("**Paste URL (Website, Youtube or direct PDF link)**")
with right:
    uploaded = st.file_uploader("**... or upload a pdf**",type=["pdf"])

#Tiny helper

def is_youtube(u:str) -> bool:
    try:
        netloc = urlparse(u).netloc.lower()
        return any(host in netloc for host in ["youtube.com","youtu.be"])
    except Exception:
        return False

def head_content_type(u: str, timeout=12) -> str | None:    #To guess the type of file before downloading
    try:
        r=requests.head(u,allow_redirects=True,timeout=timeout, headers={"User Agent":"Mozilla/5.0"})
        return( r.headers.get("Content-Type")or "").lower()
    except Exception:
        return None
        
def clean_caption_text(text:str)->str:
    text = re.sub(r"\[(?:music|applause|laughter| .*?)]"," ",text,flags=re.I)
    text = re.sub(r"\s+"," ",text)
    return text.strip()
    
def json3_to_text(s: str) -> str:
    try:
        data=json.loads(s)
        lines=[]
        for ev in data.get("events",[]):
            for seg in ev.get("segs",[]) or []:
                t = seg.get("utf8","")
                if t:
                    lines.append(t.replace("\n"," "))
        return clean_caption_text(" ".join(lines))
    except Exception:
        return clean_caption_text(s)
        
def fetch_caption_text(cap_url:str) -> str:
    resp = requests.get(cap_url, timeout=20, headers={"User-Agent":"Mozilla/5.0"})
    ctype = ( resp.headers.get("Content-type") or "").lower()
    body = resp.text

    if "text/vtt" in ctype or cap_url.endswith(".vtt"):
        ## Strip timestamps and headers
        out=[]
        for line in body.splitlines():
            s = line.strip()
            if not s in "-->" in s or s.isdigit() or s.startswith("WEBVTT"):
                continue
            out.append(s)
        return clean_caption_text(" ".join(out))
        
    if "application/json" in ctype or cap_url.endswith(".json3") or body.strip().startswith("{"):
        return json3_to_text(body)
        
    #Plain text fallback
    return clean_caption_text(body)

def build_llm(groq_api_key: str, model:str, temperature:float):
    chosen = (custom_model.strip() if custom_model else model)  #If the client want to use the custome model(Other than optional)
    return ChatGroq(model=chosen, groq_api_key=groq_api_key, temperature=temperature)

## Now build two Prompts
# 1- map prompt: Short summary for each chunk (kept tiny to protect context window)
# 2- Combine prompt: Final, summary with your style/tone/length

def build_prompts(out_len: int, out_style:str, tone:str, want_outline:bool, want_keywords:bool, out_lang:str):
    #Map prompt - Summarize a chunk into 3-6 bullets(<= 80 words total)
    map_template=''' 

    Summarize the following text into 3-6 crisp bullet points, max 80 words total.
    keep only the core facts/claims

    TEXT:
    {text}
    '''

    map_prompt = PromptTemplate(template=map_template,input_variables=["text"])

    style_map = {

        "Bullets":"Return crisp bullet points only",
        "Paragraph":"Return one cohensive paragraph",
        "Both":"Start with 6-10 concisebullet points, then a cohensive paragraph"
    }

    tone_map = {
        "Neutral":"neutral, information-dense",
        "Formal":"Formal and precise",
        "Casual":"casual and friendly",
        "Executive":"Executive, top-dpwn, action-oriented"
    }
    tone_text = tone_map.get(tone, "neutral, information-dense")
    lang = "Match the user's language" if out_lang=="Auto" else f"Write in {out_lang}"
    
    extras = []

    if want_outline:
        extras.append("Provide a short outline with top 3-6 sections")
    if want_keywords:
        extras.append("Extract 8-12 keywords and 5-8 suggested hashtags")
    extras_text = ("\n -" + "\n -".join(extras)) if extras else ""
    
    combine_template = f"""
    You will receive multiple mini-summaries of different parts of the same source. Combine them into a single,
    faithful summary.and

    Constraints and style:
    - Target length={out_len} words.
    - Output style:{style_map[out_style]}
    - Tone:{tone_map[tone]},
    - {lang}
    - Be faithful to the source; donot invent facts.
    - If the content is optionated, label  options as options.
    - Avoid repetitions. {extras_text}

    Return only the summary (and requested sections); no preambles

    INPUT_SUMMARIZES:
    {{text}}
    """

    combine_prompt = PromptTemplate(template=combine_template, input_variables=["text"])
    return map_prompt, combine_prompt

def choose_chain_type(chain_mode:str, docs:list) ->str:
    if chain_mode !="Auto":
        return chain_mode.lower().replace("-","_")
    total_chars = sum(len(d.page_content or "")for d in docs)

    return "map_reduce" if total_chars> 15000 else "stuff"

def even_sample(docs, k: int):
    n = len(docs)
    if k>= n:
        return docs
    idxs = [round(i *(n-1)/ (k-1)) for i in range(k)]
    return [docs[i] for i in idxs]

def load_youtube_docs(url:str):

    try:
        loader = YoutubeLoader.from_youtube_url(
            url, add_video_info=True,
            language = ["en","en-US","en-GB","ur","hi"],
            translation = None,
        )
        docs = loader.docs()
        if docs and any ((d.page_content or "").strip() for d in docs):
            return docs, {"type": "youtubes"}
    except Exception:
        pass

    ## Fallback yt-dlp(auto human caption; uses the voice in video and make captions, then summarize)
    ydl_opts= {"skip_download":True, "quiet":True, "socket_timeout": 30, "retries": 10, "format": "bestaudio/best","writesubtitles": True, "writeautomaticsub": True,}

    with YoutubeDL(ydl_opts) as ydl:
        try:
            info = ydl.extract_info(url, download=False)
            # Get subtitles (manual or automatic)
            captions = info.get("subtitles") or info.get("automatic_captions") or {}
        except Exception as e:
            print("Error fetching video info:", e)
            captions = {}

        # Function to get first available track URL
    def first_track_url(track_dict, langs=("en","en-US","en-GB")):
        for lg in langs:
            if lg in track_dict and track_dict[lg]:
                return track_dict[lg][0].get("url")
        return None

    # Use only captions variable for fallback
    cap_url = first_track_url(captions)

    if not cap_url:
        raise RuntimeError("This video exposes no captions (Human or auto)")

    text = fetch_caption_text(cap_url)
    from langchain.schema import Document
    return [Document(page_content=text, metadata={"source": url})], {"type": "youtube_fallback"}


@st.cache_data(show_spinner=False)
def fetch_and_load(url:str, chunk_size:int, chunk_overlap:int):
    meta = {"source": url, "type":"html", "title":None}

    if is_youtube(url):
        docs, yt_meta = load_youtube_docs(url)
        meta.update(yt_meta)
        
        try:
            if docs and docs[0].metadata.get("title"):
                meta["title"] = docs[0].metadata["title"]
        except Exception:
            pass
        return docs, meta
    
    ctype = head_content_type(url) or ""
    if "pdf" in ctype or url.lower().endswith(".pdf"):
        meta["type"]="pdf"
        with requests.get(url, stream=True, timeout=20, headers={"User-agent":"Mozilla/5.0"}) as r:
            r.raise_for_status()
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        tmp.write(chunk)
                tmp_path = tmp.name
        loader = PyPDFLoader(tmp_path)
        docs = loader.docs()
        return docs, meta
    #Webpage
    try:
        loader = WebBaseLoader([url])
        docs = loader.load()
        if docs and docs[0].metadata.get("title"):
            meta["title"] = docs[0].metadata["title"]
    except Exception:
        html = requests.get(url, timeout=20, headers={"User-Agent":"Mozilla/5.0"}).text

        from langchain.schema import Document
        text = re.sub(r"<[^>]+>"," ",html)
        docs = [Document(page_content=text, metadata={"source":url})]

    if docs and sum(len(d.page_content or "") for d in docs)> chunk_size * 1.5:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size = chunk_size, chunk_overlap = chunk_overlap,
            separators=["\n\n","\n",".","?","!"," "],)
        out = []
        for d in docs:
            out.extend(splitter.split_documents([d]))
        return out, meta
    return docs, meta

def load_pdf_from_upload(uploaded_file, chunk_size:int, chunk_overlap: int):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = tmp.name
    loader = PyPDFLoader(tmp_path)
    docs = loader.load()

    if docs and sum(len(d.page_content or "") for d in docs)> chunk_size * 1.5:
        splitter = RecursiveCharacterTextSplitter(chunk_size = chunk_size, chunk_overlap = chunk_overlap)
        parts = []
        for d in docs:
            parts.extend(splitter.split_documents([d]))
        return parts
    return docs 

## Chain Runner
def run_chain(llm, docs, map_prompt:PromptTemplate, combine_prompt:PromptTemplate, mode:str, max_map_chunk:int) -> str:
    mode =  mode.lower().replace("-","_")

    if mode=="stuff":
        chain =  load_summarize_chain(llm, chain_type= "stuff", prompt = combine_prompt)
    else:
        if len(docs)> max_map_chunk:
            docs = even_sample(docs, max_map_chunks)
            st.info(f"Long Source: Sampled{max_map_chunk} chunks evenly to fit the context.")

        chain = load_summarize_chain(
            llm,
            chain_type="map_reduce",
            map_prompt = map_prompt,
            combine_prompt = combine_prompt
        )

    try:
        res = chain.invoke({"input_documents":docs})
        return res["output_text"] if isinstance(res, dict) and "output_text" in res else str(res)
    except TypeError:
        return chain.run(input_documents = docs)
    

st.markdown("### 🚀 RUN")
go = st.button("Summarize")

if go:
    if not groq_api_key.strip():
        st.error("Please provide Groq API Key in the sidebar")

    docs, meta = [], {"type": None, "source": None, "title":None}

    try:
        stage = "loading_source"
        with st.spinner("Loading Source..."):
            if uploaded is not None:
                docs = load_pdf_from_upload(uploaded, chunk_size,  chunk_overlap)
                meta.update({"type":"pdf", "source":uploaded.name})
            elif url.strip():
                if not validators.url(url):
                    st.error("Please enter a valid URL.")
                    st.stop()
                docs, meta  = fetch_and_load(url, chunk_size, chunk_overlap)
            else:
                st.error("Provide a URL or upload a PDF")
                st.stop()
            if not docs or not any ((d.page_content or "").strip() for d in docs):
                st.error("Could not extract text. See Notes Below.")
                st.stop()
        # Quick preview for sanity
        if show_preview:
            with st.expander(" 🔍 Source preview"):
                preview = "".join(d.page_content or "" for d in docs[:3])[:1200].strip()
                st.write(f"**Detected Type:** `{meta.get('type')}`")
                if meta.get("title"): st.write(f"**Title:** {meta['title']}")
                st.text_area("First ~1200 characters", preview,  height=150)
        #Build LLm+prompt
        stage = "Initializing LLM"
        llm = build_llm(groq_api_key, model, temperature)
        stage = "Building prompts"
        map_prompt, combine_prompt = build_prompts(out_len,out_style,tone,want_outline, want_keywords, out_lang)

        #Pick chain type(auto/stuff/map_reduce)
        stage = "Selecting chain"
        mode = choose_chain_type(chain_mode, docs)

        #Run the chain and display
        stage = f"running chain({mode})"
        with st.spinner(f"Summarizing via {(custom_model or model)} ({model})..."):
            summary = run_chain(llm, docs, map_prompt, combine_prompt, mode, max_map_chunks)

        st.success("Done")
        st.subheader(" ✅ Summary")
        st.markdown(f"""
    <div style="
        padding: 20px;
        border-radius: 12px;
        background: linear-gradient(145deg, #1e3a8a, #3b82f6);
        color: white;
        font-size: 16px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    ">
    {summary}
    </div>
        """, unsafe_allow_html=True)
        
        # Export
        st.download_button(" ⬇ Download .txt", data=summary, file_name = "summary.txt",
                           mime = "text/plain")
        st.download_button("⬇ Download .md", data = f"#Summary\n\n{summary}\n",
                           file_name = "summary.md", mime = "text/markdown")
    except Exception as e:
        st.error(f"Failed during **{stage}** -> {type(e).__name__}:{e}")
        import traceback; st.code(traceback.format_exc())

with st.expander(" ⚠ Notes: what works vs. what to avoid"):
    st.markdown(
        """
- **Best:** Public webpages, Youtube videos with captions (or auto-captions), direct PDF links, and uploaded PDFs.
- **Might Fail:** Login-only pages, heavy js Pages, scanned PDFs with OCR, or sites that blocks scrapes (CORS Blockage)
- **Too Long?:** Lower chunk size / Max chunks, or keep Map-reduce ON. (This avoids content-lenght errors.)

"""
    )
