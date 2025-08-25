from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic_models import QueryInput, QueryResponse, DocumentInfo, DeleteFileRequest
from langchain_utils import get_rag_chain
from db_utils import insert_application_logs, get_chat_history, get_all_documents, insert_document_record, delete_document_record
from chroma_utils import index_document_to_chroma, delete_doc_from_chroma
import os
import uuid
import logging
import shutil
from datetime import datetime

# Logging
logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = FastAPI()

def save_conversation_to_txt(session_id: str, user_query: str, ai_response: str):
    directory = "user-conversations"
    os.makedirs(directory, exist_ok=True)
    filename = os.path.join(directory, f"{session_id}.txt")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(filename, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] User: {user_query}\n")
        f.write(f"[{timestamp}] Assistant: {ai_response}\n")
        f.write("-" * 50 + "\n")

@app.post("/chat", response_model=QueryResponse)
def chat(query_input: QueryInput):
    session_id = query_input.session_id or str(uuid.uuid4())
    chat_history = get_chat_history(session_id)
    rag_chain = get_rag_chain(query_input.model.value)

    answer = rag_chain.invoke({
        "input": query_input.question,
        "chat_history": chat_history
    })['answer']

    insert_application_logs(session_id, query_input.question, answer, query_input.model.value)
    save_conversation_to_txt(session_id, query_input.question, answer)

    return QueryResponse(answer=answer, session_id=session_id, model=query_input.model)

@app.post("/upload-doc")
async def upload_and_index_document(files: list[UploadFile] = File(...)):
    results = []
    for file in files:
        allowed_extensions = ['.pdf', '.docx', '.html', '.csv', '.xlsx', '.txt']
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in allowed_extensions:
            results.append({"filename": file.filename, "error": f"Unsupported type"})
            continue

        temp_file_path = f"temp_{uuid.uuid4().hex}{ext}"
        try:
            with open(temp_file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            file_id = insert_document_record(file.filename, file.size, file.content_type, temp_file_path)
            if file_id:
                success = index_document_to_chroma(temp_file_path, file_id)
                if success:
                    results.append({"filename": file.filename, "message": "Uploaded & indexed", "file_id": file_id})
                else:
                    delete_document_record(file_id)
                    results.append({"filename": file.filename, "error": "Index failed"})
            else:
                results.append({"filename": file.filename, "error": "DB insert failed"})
        except Exception as e:
            results.append({"filename": file.filename, "error": str(e)})
        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
    return results

@app.get("/list-docs", response_model=list[DocumentInfo])
def list_documents():
    try:
        return get_all_documents()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/delete-doc")
def delete_document(request: DeleteFileRequest):
    chroma_ok = delete_doc_from_chroma(request.file_id)
    db_ok = delete_document_record(request.file_id) if chroma_ok else False
    if chroma_ok and db_ok:
        return {"message": f"Deleted document {request.file_id}"}
    raise HTTPException(status_code=500, detail=f"Failed to delete file {request.file_id}")
