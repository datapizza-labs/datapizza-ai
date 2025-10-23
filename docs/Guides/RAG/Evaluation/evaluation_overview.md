# Metrics

This document outlines the evaluation metrics available for assessing the performance of Retrieval Augmented Generation (RAG) systems, particularly focusing on the retrieval and generation components. The implementations can be found in `datapizza/evaluation/metrics.py`.

## Retrieval Evaluation Metrics

These metrics assess the quality of the retrieved context. They typically compare a list of retrieved chunks (or their embeddings) against a "golden" set of ground truth relevant chunks.

### Defining Relevance: Exact Match vs. Cosine Similarity

At the core of retrieval evaluation is determining whether a retrieved chunk is "relevant" when compared to a ground truth chunk. Two primary methods are employed:

*   **Exact Match**:
    *   **Concept**: This method considers a retrieved chunk relevant if it is an exact string match to one of the ground truth chunks.
    *   **When to use**: Exact match is suitable when the retrieved information must be precisely identical to the source. This is often the case for retrieving identifiers, codes, specific terminologies, or when the phrasing itself is critical and variations are not acceptable. It's a stringent measure that ensures verbatim retrieval.

*   **Cosine Similarity (Similarity-based)**:
    *   **Concept**: This method determines relevance based on the semantic similarity between chunks. First, text chunks are converted into numerical vector representations (embeddings). Then, the cosine similarity between the embedding of a retrieved chunk and the embeddings of ground truth chunks is calculated. If this similarity score exceeds a predefined `similarity_threshold` (typically between 0 and 1), the retrieved chunk is considered relevant. A higher threshold demands greater similarity.
    *   **When to use**: Cosine similarity is preferred when the meaning or semantic content is more important than the exact wording. This is useful for tasks where paraphrasing is common, synonyms are expected, or the goal is to retrieve conceptually similar information even if the surface form differs. It allows for more flexibility in matching relevant content that might be expressed in various ways.

The choice between exact match and cosine similarity depends on the specific requirements of your RAG application and the nature of the data you are working with.

### Precision@k

Precision at k (P@k) measures the proportion of the top `k` retrieved items that are relevant. A higher P@k indicates that the items ranked highly by the retriever are indeed useful.

It is calculated as:
\[ P@k = \frac{\text{Number of relevant items in top k}}{\text{k}} \]

Relevance can be determined using either exact match or cosine similarity. The corresponding functions are:

```python
def precision_at_k_exact(
    retrieved_chunks: list[str],
    ground_truth_chunks: list[str],
    k: int
) -> float:

def precision_at_k_similarity(
    retrieved_embeddings: list[np.ndarray],
    ground_truth_embeddings: list[np.ndarray],
    k: int,
    similarity_threshold: float = 0.8,
) -> float:
```

### Recall@k

Recall at k (R@k) measures the proportion of all truly relevant items that are found within the top `k` retrieved items. A higher R@k indicates that the retriever is successful in finding most of the relevant information.

It is calculated as:
\[ R@k = \frac{\text{Number of relevant items in top k}}{\text{Total number of relevant items}} \]

Relevance can be determined using either exact match or cosine similarity. The corresponding functions are:

```python
def recall_at_k_exact(
    retrieved_chunks: list[str],
    ground_truth_chunks: list[str],
    k: int
) -> float:

def recall_at_k_similarity(
    retrieved_embeddings: list[np.ndarray],
    ground_truth_embeddings: list[np.ndarray],
    k: int,
    similarity_threshold: float = 0.8,
) -> float:
```

### F1-score@k

The F1-score at k (F1@k) is the harmonic mean of Precision@k and Recall@k. It provides a single score that balances both precision and recall, useful when you want a combined measure of retrieval performance.

It is calculated as:
\[ F1@k = 2 \times \frac{P@k \times R@k}{P@k + R@k} \]

Relevance can be determined using either exact match or cosine similarity. The corresponding functions are:

```python
def f1_at_k_exact(
    retrieved_chunks: list[str],
    ground_truth_chunks: list[str],
    k: int
) -> float:

def f1_at_k_similarity(
    retrieved_embeddings: list[np.ndarray],
    ground_truth_embeddings: list[np.ndarray],
    k: int,
    similarity_threshold: float = 0.8,
) -> float:
```

### Hybrid Log-Rank Score

This metric provides a more nuanced evaluation by combining recall with a rank-aware scoring component. It rewards systems that not only retrieve relevant items but also place them at higher ranks. The rank-based component uses a logarithmic scoring function (`log_rank_score`) which gives higher scores to items ranked closer to the top.

The score is a weighted average of recall and rank quality:
\[ \text{Hybrid Score} = \alpha \times \text{Recall} + (1 - \alpha) \times \text{Rank Quality} \]
where `alpha` controls the trade-off, and `gamma` in the `log_rank_score` function controls the steepness of the logarithmic curve for rank scoring.

Relevance for the recall component can be determined using either exact match or cosine similarity. The corresponding functions are:

```python
def hybrid_log_rank_score_exact(
    retrieved_chunks: list[str],
    ground_truth_chunks: list[str],
    gamma: float = 1.0,
    alpha: float = 0.5,
) -> float:

def hybrid_log_rank_score_similarity(
    retrieved_embeddings: list[np.ndarray],
    ground_truth_embeddings: list[np.ndarray],
    similarity_threshold: float = 0.8,
    gamma: float = 1.0,
    alpha: float = 0.5,
) -> float:
```

### BLEU Score (Bilingual Evaluation Understudy)

BLEU measures the similarity between a machine-generated text (hypothesis) and one or more high-quality reference texts. It counts matching n-grams (contiguous sequences of n words) between the hypothesis and references, penalizing for brevity. While originally for machine translation, it's adapted for other text generation tasks. Scores are typically between 0 and 1 (or 0-100, here scaled to 0-1).

```python
def bleu_score(
    retrieved_chunk: str,  # Intended to be the generated text
    ground_truth_chunks: list[str] # List of reference texts
) -> float:
```
*Note: The parameter `retrieved_chunk` here refers to the generated text to be evaluated, and `ground_truth_chunks` are the reference texts.*

### ROUGE Score (Recall-Oriented Understudy for Gisting Evaluation)

ROUGE is a set of metrics used for evaluating automatic summarization and machine translation. It works by comparing an automatically produced summary or translation against a set of reference summaries (typically human-produced).

Key ROUGE variants include:

-   **ROUGE-N**: Measures overlap of n-grams. `ROUGE-1` refers to unigram overlap, `ROUGE-2` to bigram overlap.
-   **ROUGE-L**: Measures longest common subsequence (LCS) based statistics.

The implementation returns F1-scores for ROUGE-1, ROUGE-2, and ROUGE-L.

```python
def rouge_score(
    retrieved_chunk: str, # Intended to be the generated text
    ground_truth_chunk: str # A single reference text
) -> dict: # Returns {"rouge1": ..., "rouge2": ..., "rougeL": ...}
```
*Note: Similar to BLEU, `retrieved_chunk` refers to the generated text, and `ground_truth_chunk` is the reference text.*

## Generation Evaluation Metrics

These metrics assess the quality of the text generated by the language model, typically an answer or a summary, based on the retrieved context.

### LLM-as-judge

**Concept**:
LLM-as-judge is an evaluation technique where a separate, often more powerful, Language Model (the "judge" LLM) is used to assess the quality of the output generated by the system under evaluation. The judge LLM is given a prompt that includes the input query, the generated response, and sometimes reference answers or specific criteria. It then outputs a score or a qualitative assessment based on these inputs.

**Why and When to Use**:
This technique is particularly useful for evaluating aspects of generation that are subjective and difficult to capture with traditional automated metrics. These aspects can include:

*   **Coherence and Readability**: How well-written and easy to understand is the generated text?
*   **Relevance**: How relevant is the answer to the input query?
*   **Helpfulness and Instructiveness**: How well does the answer satisfy the user's intent?
*   **Harmlessness**: Does the output avoid generating biased, offensive, or unsafe content?
*   **Factual Accuracy**: While LLMs can hallucinate, a judge LLM can sometimes cross-check facts if provided with context or if it has strong internal knowledge.

LLM-as-judge offers a scalable alternative to human evaluation, which can be time-consuming and costly. It's beneficial when you need nuanced feedback on generation quality, especially for open-ended tasks where defining precise ground truth is challenging. However, it's important to be aware that judge LLMs can also have biases and their judgments might not always align perfectly with human evaluators. Careful prompt engineering for the judge LLM is crucial for obtaining reliable results.

**Example Implementation**:

The following example demonstrates how to use a Google Gemini model as a judge to determine if a predicted answer matches a given answer for a specific query.

```python
import os
from pydantic import BaseModel
from datapizza.clients.google_client import GoogleClient

# Initialize the GoogleClient
try:
    client_judge = GoogleClient(
        api_key=os.getenv("GOOGLE_API_KEY"),
        model="gemini-2.5-flash-preview-04-17",
    )
except ValueError as e:
    print(f"Error initializing GoogleClient: {e}")
    print("Please ensure GOOGLE_API_KEY is set or Vertex AI parameters are correctly configured.")
    exit()


class MatchingResult(BaseModel):
    is_matching: bool
    # Optionally, add a field for the judge's reasoning
    # reasoning: str | None = None

# Sample data
data = [
    {"query": "What is the capital of France?", "answer": "Paris", "prediction": "Paris is the capital of France."},
    {"query": "What is 2+2?", "answer": "4", "prediction": "The result is three."},
    {"query": "Explain black holes.", "answer": "A black hole is a region of spacetime where gravity is so strong that nothing, including light and other electromagnetic waves, has enough energy to escape its event horizon.", "prediction": "Black holes are mysterious space objects."},
]

# System prompt for the judge LLM to guide its evaluation
# This is crucial for getting consistent and accurate judgments.
judge_system_prompt = """You are an AI assistant acting as an impartial judge.
Your task is to determine if the 'PREDICTION' accurately and satisfactorily answers the 'ORIGINAL QUERY', considering the provided 'ANSWER' as a reference.
Respond with a JSON object containing two keys:
1.  `is_matching`: a boolean value (true if the prediction matches, false otherwise).
2.  `reasoning`: a brief explanation for your decision, especially if it's not a match.

Focus on semantic similarity and factual correctness. Minor phrasing differences are acceptable if the meaning is preserved.
If the prediction is too vague, incomplete, or factually incorrect compared to the answer and query, it is not matching.
"""

for item in data:
    # Construct the input prompt for the judge LLM
    judge_input_prompt = f"""ORIGINAL QUERY:
{item['query']}

ANSWER:
{item['answer']}

PREDICTION:
{item['prediction']}"""

    try:
        # The `structured_response` method in GoogleClient is designed to return a Pydantic model.
        # It internally handles prompting the LLM to output JSON matching the Pydantic model's schema.
        client_response = client_judge.structured_response(
            input=judge_input_prompt,
            output_cls=MatchingResult,
            system_prompt=judge_system_prompt, # Pass the detailed system prompt
            # temperature=0.2 # Lower temperature for more deterministic judging
        )

        # The structured_data will be a list of Pydantic model instances.
        matching_data = client_response.structure_data[0] # Accessing the Pydantic model instance
        item["is_matching"] = matching_data.is_matching
        # item["reasoning"] = matching_data.reasoning # If you added reasoning to MatchingResult
        print(f"Query: {item['query']}")
        print(f"  Prediction: {item['prediction']}")
        print(f"  Is matching: {item['is_matching']}")
        # print(f"  Reasoning: {item.get('reasoning', 'N/A')}")

    except Exception as e:
        item["is_matching"] = False # Default on error
        # item["reasoning"] = f"Error during LLM call: {e!s}"
        print(f"Error processing item for query '{item['query']}': {e}")
        print(f"  Is matching: {item['is_matching']}")
```

# Building a Golden Dataset

Creating a high-quality "golden" dataset is fundamental for robust RAG evaluation. This dataset consists of queries paired with ideal outcomes for both retrieval and generation stages.

## Retrieval Ground Truth

For retrieval, the goal is to define, for each sample query, which document chunks are considered relevant.

**Ideal Format**:
The most effective ground truth for retrieval typically involves:

*   A list of **precise text spans** that are relevant to the query.
*   These spans can be represented as:
    *   The **exact string content** of the relevant chunk.
    *   Alternatively, a pointer to the chunk's location, such as `filename`, `start_char_offset`, and `end_char_offset`. This is particularly useful for large documents or when context around the chunk is important or when the same string may appear on multiple documents.

**Key Insights & Considerations**:

*   **Chunk Granularity**:
    *   A critical decision is the size of the ground truth chunks. Should they be individual sentences, paragraphs, or fixed-size blocks of text?
    *   **Alignment with System**: Ideally, the granularity of your ground truth chunks should align with how your RAG system chunks and retrieves documents. This makes comparisons more direct.
    *   **Atomic vs. Contextual**: Smaller, more atomic chunks can be good for evaluating precision (i.e., did the system find the exact piece of information?). Larger chunks might provide more context but could dilute the signal if only a small part of the chunk is truly relevant.
    *   **Practical Approach**: Often, sentence-to-paragraph level granularity strikes a good balance.
*   **Number of Relevant Chunks**: Some queries may have a single, definitive relevant chunk, while others might have multiple relevant pieces of information scattered across different chunks. Your golden dataset should reflect this variability.
*   **Metadata of Queries**: Annotate queries features (e.g. the source of the query). This way you can subset dataset, in this way it will be representative of what your RAG system will encounter. This could include real user logs, expert-crafted questions, or even synthetically generated queries designed to test specific retrieval challenges.

## Generation Ground Truth

For generation, the goal is to define, for each sample query (and ideally, its corresponding golden retrieved context), what constitutes a high-quality generated response.

**Ideal Format**:

*   **Human-Generated Responses**: The gold standard for generation ground truth is responses written or reviewed by humans. These capture nuance, correctness, and desired style better than any automated method.
*   **Multiple References**: For a single query, having multiple valid human-generated reference answers can be highly beneficial. This acknowledges that there can be several good ways to answer a question.
*   **Negative References**: Including examples of incorrect or inadequate responses is equally important. These negative references help train evaluation models to recognize what constitutes a poor response, whether due to factual inaccuracies, incompleteness, or inappropriate style. They provide contrast to positive examples and strengthen the evaluation framework's ability to discriminate between high and low-quality outputs.

**Key Insights & Considerations**:

*   **Quality over Quantity**: A smaller dataset of high-quality, carefully curated human responses is often more valuable than a large dataset of noisy or inconsistent ones.
*   **Specificity and Completeness**: Ground truth answers should be as specific and complete as required by the user's intent and the context.
*   **Factual Accuracy**: All reference answers must be factually correct and up-to-date.
*   **Style and Tone**: The ground truth should reflect the desired style, tone, and persona of your RAG system's output.
*   **Consistency with Retrieved Context (for RAG)**: When evaluating the generation component of a RAG system in isolation, the golden answer should ideally be based *only* on the information present in the "golden" retrieved context provided for that query. This helps differentiate between failures in retrieval versus failures in generation.
*   **Annotation Effort**: Creating high-quality generation ground truth is labor-intensive. Plan for significant time and resources, especially if relying on human annotators. Clear instructions and calibration exercises for annotators are essential.
*   **Diversity**: Ensure your dataset covers a wide range of query types, expected answer lengths, and complexities to thoroughly test the generation capabilities.

Building a comprehensive and well-annotated golden dataset is an iterative process, but it's an invaluable investment for understanding and improving your RAG system's performance.

## Building (Semi-)Synthetic Golden Datasets

While manually curated "golden" datasets are the ideal, they can be time-consuming and expensive to create. (Semi)synthetic dataset generation offers a way to augment or bootstrap this process using LLMs. This involves using an LLM to generate queries, answers, or both, potentially based on your existing document corpus.

**Pros**:

*   **Scalability & Speed**: LLMs can generate large volumes of query-answer pairs much faster than human annotators.
*   **Cost-Effectiveness**: Can be cheaper than extensive human annotation, especially for initial dataset creation.
*   **Coverage**: Can help generate queries for a wider range of documents or topics in your corpus, potentially uncovering blind spots.
*   **Bootstrapping**: Useful for creating an initial dataset when no or very little human-annotated data is available.

**Cons**:

*   **Quality Variation**: The quality of synthetic data can vary significantly. LLMs might generate:
    *   **Unrealistic Queries**: Queries that don't reflect real user search intent or language.
    *   **Hallucinated Answers**: Factually incorrect or nonsensical answers.
    *   **Lack of Domain Knowledge**: LLMs might not have sufficient domain-specific knowledge or understanding of the nuances within your documents to create high-quality, relevant query-answer pairs.
    *   **Limited Context Understanding**: LLMs might struggle to generate queries/answers that require deep contextual understanding spanning multiple documents or complex reasoning.
*   **Bias Amplification**: If the LLM used for generation has biases, these can be reflected and amplified in the synthetic dataset.
*   **Evaluation of the Generator**: The process itself requires careful evaluation – how do you ensure the synthetic data generator is good?

**Tips for Creating (Semi-)Synthetic Datasets**:

*   **Human-in-the-Loop / Supervision**:
    *   **Highly Recommended**: Human review and refinement of synthetically generated data are crucial to ensure quality, relevance, and factual accuracy. Don't rely solely on fully synthetic generation without oversight.
    *   **Few-Shot Prompting**: Provide the LLM with a few high-quality, human-curated examples (few-shot learning) to guide its generation process. This can significantly improve the relevance and style of the generated queries and answers, making them more similar to final user queries.
*   **Leverage Powerful Models for Ground Truth Generation**:
    *   Consider using a more powerful (and potentially more expensive) LLM or a more complex pipeline to generate ground truth data. This higher-quality synthetic data can then be used to evaluate your actual production RAG pipeline, which might use a more cost-effective model.
*   **Contextual Grounding**:
    *   When generating query-answer pairs, ensure the LLM is grounded in your actual documents. This can involve providing specific document chunks as context and instructing the LLM to generate questions and answers based *only* on that context.
*   **Iterative Refinement**: Start with a small batch, review, refine your prompts or generation strategy, and then scale up.
*   **Vary Generation Techniques**:
    *   **Question Generation from Text**: Provide a document chunk and ask the LLM to generate relevant questions.
    *   **Answer Generation from Question+Context**: Provide a (human or synthetic) question and a relevant document chunk, and ask the LLM to generate an answer.
*   **Negative Examples**: Instruct the LLM to also generate plausible but incorrect answers or irrelevant questions to help build a more robust evaluation set.
*   **Control for Complexity and Style**: Use prompts to guide the LLM on the desired complexity, type (e.g., factual, comparison, summary), and style of the generated queries and answers.
*   **Filter and Validate**: Implement post-processing steps to filter out low-quality or irrelevant synthetic data. This could involve heuristics, similarity checks against existing data, or even using another LLM as a preliminary judge.

Building a (semi)synthetic dataset is a powerful technique, but it requires careful planning, execution, and ongoing human oversight to ensure it genuinely contributes to improving your RAG system.
