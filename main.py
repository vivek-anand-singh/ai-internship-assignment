import os
import logging
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_openai import ChatOpenAI
import PyPDF2
import io

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.DEBUG)

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize OpenAI model
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

# Initialize tools
search_tool = DuckDuckGoSearchRun()

# Define agents
analyst = Agent(
    role='Blood Test Analyst',
    goal='Analyze blood test reports and provide summaries',
    backstory='You are an expert in interpreting blood test results',
    verbose=True,
    allow_delegation=False,
    tools=[search_tool],
    llm=llm
)

researcher = Agent(
    role='Health Researcher',
    goal='Find relevant health articles based on blood test results',
    backstory='You are a skilled researcher specializing in health-related topics',
    verbose=True,
    allow_delegation=False,
    tools=[search_tool],
    llm=llm
)

advisor = Agent(
    role='Health Advisor',
    goal='Provide health recommendations based on blood test results and research',
    backstory='You are an experienced health advisor who can provide personalized recommendations',
    verbose=True,
    allow_delegation=False,
    tools=[search_tool],
    llm=llm
)

def read_pdf(file_content):
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_content))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() or ""  # Ensure we handle None values
        return text
    except Exception as e:
        logging.error(f"Error reading PDF: {e}")
        return ""

@app.post("/analyze")
async def analyze_blood_test(file: UploadFile = File(...)):
    try:
        contents = await file.read()

        if file.filename.lower().endswith('.pdf'):
            blood_test_report = read_pdf(contents)
        else:
            blood_test_report = contents.decode("utf-8")

        # Debugging
        logging.debug(f"Extracted report: {blood_test_report[:500]}")  # Print first 500 chars

        # Define tasks
        task1 = Task(
            description=f"Analyze the following blood test report and provide a summary: {blood_test_report}",
            agent=analyst,
            expected_output="Summary of the blood test report"
        )

        task2 = Task(
            description="Search for relevant health articles based on the blood test analysis",
            agent=researcher,
            expected_output="List of relevant health articles"
        )

        task3 = Task(
            description="Provide health recommendations based on the blood test analysis and research",
            agent=advisor,
            expected_output="List of health recommendations"
        )

        # Create crew
        crew = Crew(
            agents=[analyst, researcher, advisor],
            tasks=[task1, task2, task3],
            verbose=True,
            process=Process.sequential
        )

        logging.debug("Starting crew execution")
        result = crew.kickoff()

        # Adjust based on result structure
        structured_result = {
            "summary": getattr(result, "summary", "No summary available"),
            "articles": getattr(result, "articles", "No articles found"),
            "recommendations": getattr(result, "recommendations", "No recommendations available")
        }

        return {"result": structured_result}

    except Exception as e:
        logging.error(f"Exception during analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
