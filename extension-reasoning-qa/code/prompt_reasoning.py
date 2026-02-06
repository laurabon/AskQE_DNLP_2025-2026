"""
Reasoning-augmented QA prompt for ASKQE extension.

This prompt instructs the model to:
1. Ground its answer strictly in the provided context
2. Perform intermediate reasoning before producing the final answer
3. Output only the final answer (reasoning is internal)

The goal is to study whether reasoning-augmented QA:
- Reduces noise introduced by back-translation
- Or over-stabilizes answer pairs, masking translation errors
"""

qa_prompt_reasoning = """Task: Answer questions based ONLY on the given sentence.

CRITICAL INSTRUCTIONS:
- Ground your answer STRICTLY in the provided context
- Do NOT use any external knowledge
- If the information is not in the sentence, answer based only on what IS present

For each question:
1. First, identify the exact part of the sentence that relates to the question
2. Reason about what the sentence EXPLICITLY states (not what you infer)
3. Formulate your answer using ONLY information from the sentence

Output ONLY the final answers as a Python list format. Do not include reasoning in output.
Do not output as code format (```python```).

*** Example Starts ***
Sentence: The patient reported chest pain that radiates to the left arm.
Questions: ["What did the patient report?", "Where does the pain radiate to?"]

Internal reasoning (not in output):
- Q1: Sentence says "patient reported chest pain" → Answer: "Chest pain"
- Q2: Sentence says "radiates to the left arm" → Answer: "The left arm"

Answers: ["Chest pain", "The left arm"]

Sentence: Diabetes mellitus (784, 10.9%), chronic lung disease (656, 9.2%), and cardiovascular disease (647, 9.0%) were the most frequently reported conditions among all cases.
Questions: ["What were the most frequently reported conditions?", "What percentage of cases reported diabetes mellitus?"]

Internal reasoning (not in output):
- Q1: Sentence lists "Diabetes mellitus, chronic lung disease, and cardiovascular disease" as most frequent
- Q2: Sentence explicitly states "(784, 10.9%)" for diabetes mellitus → 10.9%

Answers: ["Diabetes mellitus, chronic lung disease, and cardiovascular disease", "10.9%"]
*** Example Ends ***

Sentence: {{sentence}}
Questions: {{questions}}
Answers: """
