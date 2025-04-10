from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="gpt-3.5-turbo",  # or "gpt-4" if you need more power
    temperature=0,
)

def filter_comments_batch(comments: list[str], batch_size: int = 10) -> list[bool]:
    """
    Batch process comments using a single LLM call to reduce token usage and latency.
    Returns a list of True/False values indicating relevance.
    """
    results = []
    for i in range(0, len(comments), batch_size):
        batch = comments[i:i + batch_size]
        formatted_comments = "\n".join([f"{idx+1}. {comment}" for idx, comment in enumerate(batch)])
        
        prompt = f"""
            You are helping me filter YouTube comments to make them more relevant to a product review.

            Strictly return a list of 1s and 0s indicating relevance, in the same order:
            - Return '1' if the comment is relevant to the product
            - Return '0' if the comment is not

            Examples:
            Relevant:
            - This product is so sleek
            - The battery life is amazing

            Irrelevant:
            - I love your videos
            - What camera do you use?

            Now classify the following comments:
            {formatted_comments}

            Return format: [1, 0, 1, ...]
        """
        response = llm.invoke(prompt).content.strip()
        
        # Basic cleanup and evaluation
        try:
            binary = eval(response)
            results.extend([str(val).strip() == '1' for val in binary])
        except:
            print("Failed to parse response:", response)
            results.extend([False] * len(batch))  # fallback

    return results
