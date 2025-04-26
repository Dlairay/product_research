# AI-Powered Product Research Tool

This is an AI-powered product research tool built for a school project, exploring the use of emerging technologies like Retrieval-Augmented Generation (RAG) and AI Agents.

The core idea is simple:  
You input an image of a product, and the tool runs a pipeline that collects reviews and feedback for that product and its competitors. The collected data is then processed into meaningful, actionable insights.  
Agents are employed to validate and extract relevant information from official product manuals, which are embedded into a vector database. These manuals, combined with user feedback, are used in a RAG system to generate targeted design improvement suggestions — for example, specific changes like "Increase maximum table height from 70 cm to 76 cm."

The code was designed to execute end-to-end from `app.py`. While it currently functions as a proof-of-concept prototype, some bugs remain, and full flawless execution may require additional patches.

---

## Setup Instructions

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Mac/Linux
   .\venv\Scripts\activate    # On Windows
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the project root directory and add your API keys:
   ```env
   OPENAI_API_KEY=your_openai_api_key
   GOOGLE_API_KEY=your_google_api_key
   TAVILY_API_KEY=your_tavily_api_key
   ```

