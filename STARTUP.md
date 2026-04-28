# 🚀 Startup Guide: The Final 5 Steps to Launch

It looks like you are currently downloading **Llama 3.1**, which is about 4.9 GB. While that's a powerful model, remember that your laptop has a **GTX 1650 Ti (4GB VRAM)**. Llama 3.1 might run a bit slow or "lag" because it's right at the limit of your hardware.

> [!TIP]
> **Recommendation:** If Llama 3.1 feels too slow, you can stop it (`Ctrl + C`) and run `ollama run phi3` instead. Phi-3 is much smaller (around 2.3 GB), very fast, and perfect for this specific PDF task.

Since your `venv` is already active, follow these steps in order to get the project running:

## 1. Install the Libraries
Inside your active (`venv`) terminal, run this command to install the stack we discussed:

```powershell
pip install fastapi uvicorn pydantic python-multipart langchain langchain-community langchain-ollama langchain-huggingface sentence-transformers chromadb pypdf
```

## 2. Prepare the Models (Ollama)
Open a second terminal window and keep it running in the background with your model:

```powershell
ollama run phi3
```
*(Wait until you see the `>>>` prompt, which means the model is ready to answer.)*

## 3. Start the Backend
Go back to your first terminal (where the `venv` is active) and start your FastAPI server:

```powershell
uvicorn main:app --reload
```
> [!NOTE]
> If your code is inside a folder named `backend`, you might need to run `cd backend` first, or run `uvicorn backend.main:app --reload`.

## 4. Open the Frontend
Go to your `frontend` folder.
Right-click `index.html` and select **"Open with Live Server"** (if using VS Code) or just double-click the file to open it in Chrome.

## 5. The Demo Test (Crucial for Webseeder)
1. **Upload a PDF** (not a `.txt`).
2. Wait for the **"Success"** message.
3. Ask a question like: *"What is the main summary of this document?"*
4. **Capture this:** This is when you should record your screen (`Win + Alt + R`) to show the hiring team that it works!

---

### Quick Troubleshooting

*   **ModuleNotFoundError:** If it says "No module named 'langchain'", it means the `pip install` in Step 1 didn't finish or you aren't in the `venv`.
*   **Ollama Connection Error:** Ensure you can see the Ollama icon in your Windows System Tray (bottom right of your taskbar).
