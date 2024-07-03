import requests

API_BASE_URL = "https://api.txyz.ai/v1"
API_KEY = "your_api_key_here"  # Your API key generated from platform console


def create_session():
    url = f"{API_BASE_URL}/sessions"
    headers = {"Authorization": f"Bearer {API_KEY}"}

    response = requests.post(url, headers=headers)
    return response.json()


def add_document(session_id, file_path):
    url = f"{API_BASE_URL}/sessions/{session_id}/documents"
    headers = {"Authorization": f"Bearer {API_KEY}"}
    files = {"file": open(file_path, "rb")}

    response = requests.post(url, headers=headers, files=files)
    return response.json()


def chat_with_ai(session_id, query):
    url = f"{API_BASE_URL}/chat"
    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "session_id": f"{session_id}",
        "query": f"{query}",
        "response_mode": "COMPLETE"
    }

    response = requests.post(url, headers=headers, json=payload, stream=True)
    print(response)

    if response.status_code == 200:
        for event in response.iter_lines():
            print(event.decode("utf-8"))


if __name__ == "__main__":
    pdf_file_path = "/Users/pyq/Documents/google_file_system.pdf"

    try:
        session_info = create_session()

        if session_info:
            print("Session created:", session_info)
            new_session_id = session_info["session"]["session_id"]

            document_info = add_document(new_session_id, pdf_file_path)
            if document_info:
                print("Document added:", document_info)

                # Start chat with the AI
                user_query = "What is the document about?"
                chat_with_ai(new_session_id, user_query)
            else:
                print("Failed to add document")
        else:
            print("Failed to create session")
    except Exception as e:
        print("Get exception:", e)
