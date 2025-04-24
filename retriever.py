from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from dotenv import load_dotenv

load_dotenv()
# Custom RAG prompt template
# rag_prompt = PromptTemplate.from_template("""
# You are an expert product design researcher.

# Using the following manual content, suggest specific improvements to the product
# that would address common user complaints.

# If you cannot find relevant context in the manual, respond with "No suggestions found".

# Context:
# {context}

# Question:
# {question}
# """)

# rag_prompt = PromptTemplate.from_template("""
# You are an expert product design researcher. Following manual content, suggest specific improvements to the product and provide suggestions in this EXACT format:

# Name: "Secretlab Titan Evo Lite"
# Dimensions: "[length x width x height]" (only if available in context)
# Parts: "[main components mentioned]"
# Sources: "[document names used for suggestions]"
# Improvements: "[numbered list of 3-5 improvements based on user complaints, if possible, add comparison to other competitor product, also add in numbers or percentage like how much bigger or smaller]"
                                          


# If information isn't available for a section,s dont show the part(if dimension not available, dont show or mention dimension )
# Never add extra text or explanations beyond this format.

# Context:
# {context}

# Question:
# {question}
# """)
rag_prompt = PromptTemplate.from_template("""
You are an expert product design analyst. Analyze the product manual and provide suggestions in this EXACT format:

Name: "[Exact product name from context]"
Dimensions: "[Convert all dimensions to metric and imperial]"
Parts: "[List 3-5 key components from technical specs]"
Sources: "[Exact document filenames used]"
Improvements: 
1. "[Specific change] - Increase/Decrease [component] by [X%] ([current] → [proposed]) to match [Competitor Brand]'s [feature]"
2. "[Measurable adjustment] - Expand [dimension] by [X cm/mm] ([X%]) based on [study/report] from [source]"
3. "[Ergonomic improvement] - Modify [element] angle by [X°] following [standard] used by [Competitor]"
4. "[Material change] - Use [material] with [X%] greater [property] than current [material]"

Rules:
1. ALWAYS include competitor comparisons and exact numbers/percentages
2. Use measurements from context first, then industry standards
3. If no numeric data exists, state "No quantitative data found for [aspect]"
4. Never use placeholders - omit sections if no data exists

Context:
{context}

Question:
{question}
""")

def get_retrieval_qa(product: str):
    db = Chroma(
    persist_directory="./chromadb",
    collection_name="product_manuals",
    embedding_function=OpenAIEmbeddings()
)

    retriever = db.as_retriever(search_kwargs={"k": 3, "filter": {"product": product}})
    llm = ChatOpenAI(temperature=0)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": rag_prompt},
        return_source_documents=True
    )
    return qa_chain

def run_suggestions(product: str):
    print(f"💡 Generating suggestions for: {product}")
    qa_chain = get_retrieval_qa(product)

    # question = (
    #     "How can this product be improved based on the top user complaints?"
    # )
    question = (
        f"Provide product analysis and improvement suggestions about the {product}  in the exact format: "
        "name, dimensions (if available), parts, improvements, sources."
    )
    print(question)
    response = qa_chain.invoke({"query": question})
    print(f"\n🧠 Suggestions:\n{response['result']}\n")
    return response["result"]


run_suggestions("Secretlab Titan Evo Lite")
