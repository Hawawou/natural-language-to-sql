# natural-language-to-sql
This project aims to develop a system that converts natural language questions into SQL queries, enabling non-technical users to query databases easily. The task is fundamental in applications such as chatbots, voice assistants, and user-friendly database interfaces. The system is built using the WikiSQL dataset, which includes tables, natural language questions, and their corresponding SQL queries.

Objectives
- Create a pipeline that accurately transforms natural language questions into SQL queries.
- Utilize pre-trained transformer models (e.g., BART) and fine-tune them for the Text-to-SQL task.
- Evaluate performance using standard metrics like exact match accuracy and execution accuracy.
## Measure system performance using key metrics:
- Exact Match (EM): Achieved 21%, reflecting the proportion of queries perfectly matching references.
- BLEU Score: Scored 0.598, showing moderate similarity to reference queries.
- ROUGE-L: Scored 0.86, indicating high overlap in sequence structure
- Analyze the system's strengths and weaknesses to suggest improvements.
