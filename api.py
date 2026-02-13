from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from agent.agent import Agent
import shutil
from pathlib import Path

app = FastAPI(title="Agentic AI API")
agent = Agent()

@app.post("/query")
async def query_agent(query: str = Form(...)):
    """Execute agent query with text input"""
    try:
        result = agent.run(query)
        return {"result": result}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/analyze")
async def analyze_image(
    image: UploadFile = File(...),
    query: str = Form(...)
):
    """Analyze screenshot with query"""
    try:
        temp_path = f"temp_{image.filename}"
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
        
        result = agent.analyze_screenshot(temp_path, query)
        Path(temp_path).unlink()
        
        return {"result": result}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/health")
async def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
