from pathlib import Path

import gradio as gr
import httpx

API_BASE = "http://127.0.0.1:8000"


def _call_api(method: str, path: str, **kwargs):
    """Central place for every API call — one spot to translate backend
    errors into something Gradio can show the user instead of crashing."""
    try:
        resp = httpx.request(method, f"{API_BASE}{path}", timeout=60, **kwargs)
        resp.raise_for_status()
        return resp.json()
    except httpx.ConnectError:
        raise gr.Error("Can't reach the backend — is `uvicorn app.main:app` running on port 8000?")
    except httpx.HTTPStatusError as e:
        detail = e.response.json().get("detail", e.response.text)
        raise gr.Error(f"{e.response.status_code}: {detail}")


def _conversation_choices(user_id: str):
    conversations = _call_api("GET", "/conversations", params={"user_id": user_id})
    return [(c["title"] or c["thread_id"][:8], c["thread_id"]) for c in conversations]


def load_conversations(user_id: str):
    if not user_id:
        raise gr.Error("Enter a user ID first.")
    return gr.Dropdown(choices=_conversation_choices(user_id), value=None)


def start_new_conversation(user_id: str, title: str):
    if not user_id:
        raise gr.Error("Enter a user ID first.")
    new_conversation = _call_api(
        "POST", "/conversations", json={"user_id": user_id, "title": title or None}
    )
    choices = _conversation_choices(user_id)
    return gr.Dropdown(choices=choices, value=new_conversation["thread_id"]), []


def load_conversation_history(user_id: str, thread_id: str):
    if not user_id or not thread_id:
        return []
    return _call_api("GET", f"/conversations/{thread_id}/messages", params={"user_id": user_id})


def delete_selected_conversation(user_id: str, thread_id: str):
    if not user_id:
        raise gr.Error("Enter a user ID first.")
    if not thread_id:
        raise gr.Error("Pick a conversation to delete first.")

    _call_api("DELETE", f"/conversations/{thread_id}", params={"user_id": user_id})
    choices = _conversation_choices(user_id)
    return gr.Dropdown(choices=choices, value=None), []


def send_message(user_id: str, thread_id: str, message: str, use_retrieval: bool, history: list):
    if not user_id or not thread_id:
        raise gr.Error("Pick (or start) a conversation first.")
    if not message.strip():
        return history, ""

    result = _call_api(
        "POST",
        "/chat",
        json={
            "user_id": user_id,
            "thread_id": thread_id,
            "message": message,
            "use_retrieval": use_retrieval,
        },
    )
    history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": result["response"]},
    ]
    return history, ""


def upload_document_text(user_id: str, title: str, text: str):
    if not user_id:
        raise gr.Error("Enter a user ID first.")
    if not title.strip() or not text.strip():
        raise gr.Error("Both a title and some text are required.")

    result = _call_api("POST", "/documents", json={"user_id": user_id, "title": title, "text": text})
    return (
        f"✅ Stored {result['chunks_stored']} new chunk(s), "
        f"skipped {result['chunks_skipped_duplicate']} duplicate(s) already in the knowledge base."
    )


def upload_document_file(user_id: str, source: str, filepath: str):
    if not user_id:
        raise gr.Error("Enter a user ID first.")
    if not filepath:
        raise gr.Error("Choose a .pdf, .docx, or .txt file first.")
    if not source.strip():
        raise gr.Error("Give it a title/source name.")

    with open(filepath, "rb") as f:
        result = _call_api(
            "POST",
            "/documents/upload",
            data={"user_id": user_id, "source": source},
            files={"file": (Path(filepath).name, f)},
        )
    status = (
        f"✅ Extracted and stored {result['chunks_stored']} new text chunk(s), "
        f"skipped {result['chunks_skipped_duplicate']} duplicate(s)."
    )
    if result.get("images_found"):
        status += (
            f"\n\nAlso found {result['images_found']} embedded image(s) — "
            f"stored {result['images_stored']}, skipped {result['images_skipped_duplicate']} duplicate(s)."
        )
    return status


def upload_image(user_id: str, source: str, filepath: str):
    if not user_id:
        raise gr.Error("Enter a user ID first.")
    if not filepath:
        raise gr.Error("Choose an image first.")
    if not source.strip():
        raise gr.Error("Give it a title/source name.")

    with open(filepath, "rb") as f:
        result = _call_api(
            "POST",
            "/images",
            data={"user_id": user_id, "source": source},
            files={"file": (Path(filepath).name, f)},
        )
    return "✅ Image stored in the knowledge base." if result.get("stored") else "Already stored (duplicate image skipped)."


with gr.Blocks(title="BrainFood ChatBot") as demo:
    gr.Markdown("# BrainFood ChatBot")

    user_id_box = gr.Textbox(label="Your user ID", placeholder="e.g. srujan")

    with gr.Tabs():
        with gr.Tab("Chat"):
            with gr.Row():
                conversation_dropdown = gr.Dropdown(label="Conversation", choices=[], scale=2)
                refresh_btn = gr.Button("Load my conversations", scale=1)
                delete_btn = gr.Button("Delete selected", scale=1, variant="stop")
            with gr.Row():
                new_title_box = gr.Textbox(label="New conversation title (optional)", scale=2)
                new_btn = gr.Button("Start new conversation", scale=1)

            chatbot = gr.Chatbot(label="Chat", height=450)
            use_retrieval_box = gr.Checkbox(
                label="Use knowledge base (RAG)",
                value=False,
                info="Retrieves relevant uploaded/shared documents before answering.",
            )
            message_box = gr.Textbox(label="Message", placeholder="Type a message and press Enter")

        with gr.Tab("Knowledge base — text & documents"):
            gr.Markdown(
                "Content here is retrievable when **Use knowledge base** is checked. Private to your "
                "user ID unless it matches a shared/global document. Identical content uploaded twice "
                "is automatically skipped, not duplicated."
            )
            with gr.Group():
                gr.Markdown("**Paste text directly**")
                doc_title_box = gr.Textbox(label="Title / source name", placeholder="e.g. meeting-notes-2026-08")
                doc_text_box = gr.Textbox(label="Document text", lines=8, placeholder="Paste the content to store...")
                upload_text_btn = gr.Button("Upload text")
                upload_text_status = gr.Markdown()

            with gr.Group():
                gr.Markdown("**Or upload a file** — .pdf, .docx, or .txt")
                file_source_box = gr.Textbox(label="Title / source name", placeholder="e.g. Q1-trend-report.pdf")
                file_upload_box = gr.File(label="File", file_types=[".pdf", ".docx", ".txt"])
                upload_file_btn = gr.Button("Upload file")
                upload_file_status = gr.Markdown()

        with gr.Tab("Knowledge base — images"):
            gr.Markdown(
                "Images are searched separately from text, using CLIP — ask a question with "
                "**Use knowledge base** checked and a matching image can be found by meaning, "
                "not just filename."
            )
            image_source_box = gr.Textbox(label="Title / source name", placeholder="e.g. booth-photo-2026")
            image_upload_box = gr.Image(label="Image", type="filepath")
            upload_image_btn = gr.Button("Upload image")
            upload_image_status = gr.Markdown()

    refresh_btn.click(load_conversations, inputs=user_id_box, outputs=conversation_dropdown)
    conversation_dropdown.change(
        load_conversation_history,
        inputs=[user_id_box, conversation_dropdown],
        outputs=chatbot,
    )
    delete_btn.click(
        delete_selected_conversation,
        inputs=[user_id_box, conversation_dropdown],
        outputs=[conversation_dropdown, chatbot],
    )
    new_btn.click(
        start_new_conversation,
        inputs=[user_id_box, new_title_box],
        outputs=[conversation_dropdown, chatbot],
    )
    message_box.submit(
        send_message,
        inputs=[user_id_box, conversation_dropdown, message_box, use_retrieval_box, chatbot],
        outputs=[chatbot, message_box],
    )
    upload_text_btn.click(
        upload_document_text,
        inputs=[user_id_box, doc_title_box, doc_text_box],
        outputs=upload_text_status,
    )
    upload_file_btn.click(
        upload_document_file,
        inputs=[user_id_box, file_source_box, file_upload_box],
        outputs=upload_file_status,
    )
    upload_image_btn.click(
        upload_image,
        inputs=[user_id_box, image_source_box, image_upload_box],
        outputs=upload_image_status,
    )

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())
