"""
このファイルは、画面表示以外の様々な関数定義のファイルです。
"""

############################################################
# ライブラリの読み込み
############################################################
import os
import csv
from dotenv import load_dotenv
import streamlit as st
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import HumanMessage
from langchain_openai import ChatOpenAI
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.schema import Document as LangChainDocument
import constants as ct


############################################################
# 設定関連
############################################################
# 「.env」ファイルで定義した環境変数の読み込み
load_dotenv()


############################################################
# 関数定義
############################################################

def get_source_icon(source):
    """
    メッセージと一緒に表示するアイコンの種類を取得

    Args:
        source: 参照元のありか

    Returns:
        メッセージと一緒に表示するアイコンの種類
    """
    # 参照元がWebページの場合とファイルの場合で、取得するアイコンの種類を変える
    if source.startswith("http"):
        icon = ct.LINK_SOURCE_ICON
    else:
        icon = ct.DOC_SOURCE_ICON
    
    return icon


def build_error_message(message):
    """
    エラーメッセージと管理者問い合わせテンプレートの連結

    Args:
        message: 画面上に表示するエラーメッセージ

    Returns:
        エラーメッセージと管理者問い合わせテンプレートの連結テキスト
    """
    return "\n".join([message, ct.COMMON_ERROR_MESSAGE])


def is_employee_search_query(chat_message):
    """
    ユーザーの問い合わせが従業員情報の検索かどうかを判定
    
    Args:
        chat_message: ユーザー入力値
    
    Returns:
        従業員情報の検索である場合True、それ以外False
    """
    # 従業員情報検索に関連するキーワード
    employee_keywords = [
        "従業員", "社員", "人事部", "営業部", "IT部", "マーケティング部", 
        "経理部", "総務部", "部署", "スタッフ", "メンバー", "一覧",
        "リスト", "名簿", "所属"
    ]
    
    # いずれかのキーワードが含まれている場合、従業員検索と判定
    return any(keyword in chat_message for keyword in employee_keywords)


def load_employee_csv_all():
    """
    従業員CSVファイル全件を読み込む
    
    Returns:
        全従業員情報を含むDocumentのリスト
    """
    csv_path = "./data/社員について/社員名簿.csv"
    
    if not os.path.exists(csv_path):
        return []
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        
        # 全レコードをテキストとして結合
        content_parts = []
        for i, row in enumerate(rows, 1):
            record_text = f"\n【従業員{i}】\n"
            for key, value in row.items():
                record_text += f"{key}: {value}\n"
            content_parts.append(record_text)
        
        combined_content = "\n".join(content_parts)
        
        return [LangChainDocument(
            page_content=combined_content,
            metadata={"source": csv_path, "total_records": len(rows)}
        )]


def get_llm_response(chat_message):
    """
    LLMからの回答取得

    Args:
        chat_message: ユーザー入力値

    Returns:
        LLMからの回答
    """
    # LLMのオブジェクトを用意
    llm = ChatOpenAI(model_name=ct.MODEL, temperature=ct.TEMPERATURE)

    # 従業員情報の検索かどうかを判定
    is_employee_query = is_employee_search_query(chat_message)
    
    # 従業員情報の検索の場合、CSV全件を直接取得
    if is_employee_query:
        # CSV全件を取得
        employee_docs = load_employee_csv_all()
        
        if employee_docs:
            # モードによってLLMから回答を取得する用のプロンプトを変更
            if st.session_state.mode == ct.ANSWER_MODE_1:
                question_answer_template = ct.SYSTEM_PROMPT_DOC_SEARCH
            else:
                question_answer_template = ct.SYSTEM_PROMPT_INQUIRY
            
            # プロンプトテンプレートを作成
            question_answer_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", question_answer_template),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}")
                ]
            )
            
            # LLMから回答を取得する用のChainを作成
            question_answer_chain = create_stuff_documents_chain(llm, question_answer_prompt)
            
            # CSV全件をコンテキストとして直接渡す
            llm_response = question_answer_chain.invoke({
                "input": chat_message,
                "chat_history": st.session_state.chat_history,
                "context": employee_docs
            })
            
            # LLMレスポンスを会話履歴に追加
            st.session_state.chat_history.extend([HumanMessage(content=chat_message), llm_response])
            
            # 返り値を UI が期待する形に揃える
            response = {
                "answer": llm_response,
                "context": employee_docs
            }
            
            return response
    
    # 通常のベクトル検索処理
    # 会話履歴なしでもLLMに理解してもらえる、独立した入力テキストを取得するためのプロンプトテンプレートを作成
    question_generator_template = ct.SYSTEM_PROMPT_CREATE_INDEPENDENT_TEXT
    question_generator_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", question_generator_template),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ]
    )

    # モードによってLLMから回答を取得する用のプロンプトを変更
    if st.session_state.mode == ct.ANSWER_MODE_1:
        # モードが「社内文書検索」の場合のプロンプト
        question_answer_template = ct.SYSTEM_PROMPT_DOC_SEARCH
    else:
        # モードが「社内問い合わせ」の場合のプロンプト
        question_answer_template = ct.SYSTEM_PROMPT_INQUIRY
    # LLMから回答を取得する用のプロンプトテンプレートを作成
    question_answer_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", question_answer_template),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ]
    )

    # 会話履歴なしでもLLMに理解してもらえる、独立した入力テキストを取得するためのRetrieverを作成
    history_aware_retriever = create_history_aware_retriever(
        llm, st.session_state.retriever, question_generator_prompt
    )

    # LLMから回答を取得する用のChainを作成
    question_answer_chain = create_stuff_documents_chain(llm, question_answer_prompt)
    # 「RAG x 会話履歴の記憶機能」を実現するためのChainを作成
    chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    # LLMへのリクエストとレスポンス取得
    llm_response = chain.invoke({"input": chat_message, "chat_history": st.session_state.chat_history})

    # LLMレスポンスを会話履歴に追加
    st.session_state.chat_history.extend([HumanMessage(content=chat_message), llm_response["answer"]])

    # 返り値を UI が期待する形に揃える（answer と context を含む辞書）
    response = {
        "answer": llm_response.get("answer", ""),
        "context": llm_response.get("context", [])
    }

    return response