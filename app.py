import streamlit as st
import pandas as pd
import fitz  # PyMuPDF - 한국어 PDF에 강한 PDF 리더

from supabase import create_client, Client

# --- LlamaIndex 핵심 도구 ---
from llama_index.core import (
    VectorStoreIndex,
    Document,
    StorageContext,
    Settings,
)
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from google.genai.types import EmbedContentConfig
from llama_index.vector_stores.supabase import SupabaseVectorStore

# --------------------------------------------------------------------------
# 페이지 설정
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="사업보고서 RAG 챗봇",
    page_icon="📊",
    layout="wide",
)

# --------------------------------------------------------------------------
# Secrets 불러오기
# --------------------------------------------------------------------------
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
SUPABASE_DB_CONNECTION = st.secrets["SUPABASE_DB_CONNECTION"]

# --------------------------------------------------------------------------
# Supabase 및 LlamaIndex 초기화
# --------------------------------------------------------------------------
@st.cache_resource
def init_supabase() -> Client:
    return create_client(SUPABASE_URL, SUPABASE_KEY)

@st.cache_resource
def init_llama_index():
    # LLM: 답변 생성용
    Settings.llm = GoogleGenAI(
        model="gemini-2.5-flash",
        api_key=GEMINI_API_KEY,
        temperature=0.1,
    )
    # 임베딩: 텍스트 → 768차원 벡터
    Settings.embed_model = GoogleGenAIEmbedding(
        model_name="gemini-embedding-001",
        api_key=GEMINI_API_KEY,
        embedding_config=EmbedContentConfig(
            output_dimensionality=768
        ),
    )
    Settings.chunk_size = 500
    Settings.chunk_overlap = 50

def get_vector_store(company_name: str):
    """회사명으로 컬렉션 분리. 영문 변환은 공백→_, 소문자만."""
    return SupabaseVectorStore(
        postgres_connection_string=SUPABASE_DB_CONNECTION,
        collection_name=company_name.replace(" ", "_").lower(),
        dimension=768,
    )

supabase = init_supabase()
init_llama_index()

# --------------------------------------------------------------------------
# PDF 텍스트 추출 (PyMuPDF 이용 - 한국어 폰트 처리에 강함)
# --------------------------------------------------------------------------
def extract_pdf_with_pymupdf(pdf_bytes: bytes, company_name: str):
    """
    PyMuPDF로 PDF를 페이지 단위로 텍스트 추출.
    pypdf보다 한국어 CID 폰트 처리가 훨씬 안정적임.
    
    반환:
        documents: LlamaIndex Document 리스트 (페이지별)
        page_previews: (페이지 번호, 텍스트) 튜플 리스트 (UI 미리보기용)
    """
    documents = []
    page_previews = []
    total_chars = 0
    
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        for page_num, page in enumerate(doc, start=1):
            text = page.get_text("text")
            if text.strip():
                total_chars += len(text)
                page_previews.append((page_num, text))
                documents.append(
                    Document(
                        text=text,
                        metadata={
                            "page_label": str(page_num),
                            "company": company_name,
                        }
                    )
                )
    return documents, page_previews, total_chars

# --------------------------------------------------------------------------
# 화면 UI
# --------------------------------------------------------------------------
st.title("📊 사업보고서 RAG 챗봇")
st.info(
    "💡 안내: PDF 사업보고서를 업로드하면, AI가 내용을 학습하고 자연어로 질문에 답해드립니다.\n\n"
    "📌 권장: 같은 회사명으로 여러 번 업로드하면 데이터가 중복되니, "
    "다시 시도할 때는 회사명에 숫자를 붙이세요 (예: 삼성전자2)."
)

# --- 세션 상태 초기화 ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_company" not in st.session_state:
    st.session_state.current_company = None
if "index" not in st.session_state:
    st.session_state.index = None
if "pending_documents" not in st.session_state:
    st.session_state.pending_documents = None
if "pending_company" not in st.session_state:
    st.session_state.pending_company = None

tab1, tab2, tab3 = st.tabs(["📤 업로드", "💬 챗봇", "📋 채팅 기록"])

# ====================================================================
# [탭 1] 사업보고서 업로드 (2단계: 추출 → 미리보기 → 임베딩)
# ====================================================================
with tab1:
    st.subheader("사업보고서 인덱싱")
    
    company_name = st.text_input(
        "📁 회사명 입력",
        placeholder="예: 삼성전자, 카카오, 네이버",
        help="이 보고서가 어느 회사의 것인지 표시하기 위한 라벨입니다.",
    )
    
    uploaded_file = st.file_uploader(
        "📄 PDF 파일 선택",
        type=["pdf"],
        help="전자공시시스템(DART)에서 다운로드한 사업보고서 PDF를 업로드하세요.",
    )
    
    if uploaded_file is not None and company_name:
        pdf_bytes = uploaded_file.getvalue()
        
        # === 1단계: 텍스트 추출 + 미리보기 ===
        st.markdown("### 1단계: 텍스트 추출 및 검증")
        if st.button("🔍 PDF 텍스트 추출하기", type="secondary"):
            with st.spinner("PyMuPDF로 텍스트 추출 중..."):
                try:
                    documents, page_previews, total_chars = extract_pdf_with_pymupdf(
                        pdf_bytes, company_name
                    )
                    st.session_state.pending_documents = documents
                    st.session_state.pending_company = company_name
                    
                    if not documents or total_chars < 100:
                        st.error(
                            "⚠️ 텍스트를 거의 추출하지 못했습니다. "
                            "이미지로만 구성된 PDF일 가능성이 높습니다. "
                            "DART에서 Ctrl+P → 'PDF로 저장' 방식으로 다시 만들어주세요."
                        )
                        st.session_state.pending_documents = None
                    else:
                        st.success(
                            f"✅ {len(documents)}개 페이지에서 총 {total_chars:,}자 추출 완료"
                        )
                        
                        # 미리보기로 한글 정상 여부 확인
                        st.markdown("### 📝 추출된 텍스트 미리보기")
                        st.warning(
                            "👇 아래 미리보기에서 **한글이 정상적으로 보이는지 반드시 확인**하세요. "
                            "깨진 문자(`zEoy�`, `\\x00` 등)가 보이면 PDF를 다시 만들어야 합니다."
                        )
                        for page_num, text in page_previews[:3]:
                            with st.expander(f"페이지 {page_num} (처음 500자)", expanded=True):
                                st.text(text[:500])
                except Exception as e:
                    st.error(f"PDF 읽기 오류: {e}")
                    st.session_state.pending_documents = None
        
        # === 2단계: 임베딩 (미리보기 확인 후) ===
        if st.session_state.pending_documents:
            st.markdown("---")
            st.markdown("### 2단계: 임베딩 및 Supabase 저장")
            st.caption(
                "위 미리보기에서 한글이 정상이면 아래 버튼을 누르세요. "
                "Gemini가 페이지마다 임베딩을 만들어 Supabase에 저장합니다."
            )
            if st.button("🚀 임베딩 시작 (약 1~3분 소요)", type="primary"):
                with st.spinner("Gemini 임베딩 → Supabase 저장 중..."):
                    try:
                        documents = st.session_state.pending_documents
                        cn = st.session_state.pending_company
                        
                        vector_store = get_vector_store(cn)
                        storage_context = StorageContext.from_defaults(
                            vector_store=vector_store
                        )
                        index = VectorStoreIndex.from_documents(
                            documents,
                            storage_context=storage_context,
                            show_progress=True,
                        )
                        st.session_state.index = index
                        st.session_state.current_company = cn
                        st.session_state.pending_documents = None
                        
                        st.success(
                            f"✅ '{cn}' 인덱싱 완료! ({len(documents)} 페이지 처리)"
                        )
                        st.info("💬 챗봇 탭으로 이동해서 질문해보세요.")
                    except Exception as e:
                        st.error(f"임베딩 오류: {e}")

# ====================================================================
# [탭 2] 챗봇 (RAG 질의응답)
# ====================================================================
with tab2:
    st.subheader("💬 사업보고서에 질문하기")
    
    if st.session_state.current_company:
        st.caption(f"📂 분석 대상: **{st.session_state.current_company}**")
    else:
        st.warning("⚠️ 먼저 '업로드' 탭에서 사업보고서를 인덱싱해주세요.")
    
    # 과거 메시지 표시
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("📄 참고 출처"):
                    for src in msg["sources"]:
                        st.caption(src)
    
    # 새 질문 처리
    if prompt := st.chat_input("질문을 입력하세요 (예: 작년 매출은?)"):
        if not st.session_state.index:
            st.error("먼저 사업보고서를 업로드해주세요.")
        else:
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            with st.chat_message("assistant"):
                with st.spinner("답변 생성 중..."):
                    try:
                        query_engine = st.session_state.index.as_query_engine(
                            similarity_top_k=5,
                        )
                        response = query_engine.query(prompt)
                        answer = str(response)
                        st.markdown(answer)
                        
                        # 출처 페이지 정리
                        sources = []
                        for node in response.source_nodes:
                            page = node.metadata.get("page_label", "?")
                            sources.append(f"페이지 {page}: {node.text[:100]}...")
                        
                        if sources:
                            with st.expander("📄 참고 출처"):
                                for src in sources:
                                    st.caption(src)
                        
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer,
                            "sources": sources
                        })
                        
                        # Supabase chat_history 테이블에 저장
                        try:
                            supabase.table("chat_history").insert({
                                "question": prompt,
                                "answer": answer,
                                "sources": sources,
                                "company_name": st.session_state.current_company,
                            }).execute()
                        except Exception as db_e:
                            st.toast(f"DB 저장 실패: {db_e}")
                    except Exception as e:
                        st.error(f"답변 생성 오류: {e}")

# ====================================================================
# [탭 3] 채팅 기록 (Supabase에서 조회)
# ====================================================================
with tab3:
    st.subheader("📋 채팅 기록")
    try:
        response = (
            supabase.table("chat_history")
            .select("*")
            .order("created_at", desc=True)
            .limit(50)
            .execute()
        )
        if response.data:
            df = pd.DataFrame(response.data)
            df = df[["created_at", "company_name", "question", "answer"]]
            df.columns = ["시간", "회사", "질문", "답변"]
            st.dataframe(df, use_container_width=True, hide_index=True)
            
            csv = df.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "📥 CSV로 다운로드",
                csv,
                "chat_history.csv",
                "text/csv",
            )
        else:
            st.info("아직 저장된 대화가 없습니다. 챗봇 탭에서 질문해보세요!")
    except Exception as e:
        st.error(f"채팅 기록 불러오기 실패: {e}")
