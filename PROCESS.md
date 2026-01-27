1. Project Overview
Goal: Create an AI-powered documentation generator that reads codebases, understands cross-file dependencies, and outputs formatted Markdown. Tools Used: Streamlit, Pandas, Scikit-Learn (TF-IDF), LiteLLM (OpenAI), Python AST, Google Gemini as my AI agent.

2. Key Prompts (The "Vibe Coding" Loop)
a. Below are the critical prompts used to evolve the script from a broken prototype to a robust tool.
b. The API Connection Fix: "Error on Self_Assessment_Vibe_Coding.xlsx: litellm.APIConnectionError: argument of type 'NoneType' is not iterable"
c. The Encoding Fix: "Error on Self_Assessment_Vibe_Coding.xlsx: 'gbk' codec can't encode character '\u2011' in position 157"
d. The Data Sanity Check: "The sanity check returns: Baseline Accuracy (Majority Class) 4.55%"
e. The BANA 275 Lab Alignment: "I need our script have the exact same TF-IDF functionality as this one [uploaded Week 2 Lab notebook]"
f. The Small-Corpus Crash: "ValueError: max_df corresponds to < documents than min_df"
g. The Logic Upgrade (Cross-File): "I'd like to refine this script so that it can also understand the code files' logic cross-file, breaks the single-tern limit once and for all"
h. The Final Polish (Jupyter & Inputs): "My input file should be ./src and seemingly it can not read in the ipynb file"

3. Challenges & Solutions
Challenge A: The "NoneType" API Error
Issue: The script was crashing immediately when processing Excel files. Root Cause: The litellm library was failing to read the API key from the environment variables during the first call. Solution: We switched from implicit environment variables (os.environ) to explicit key passing. We added api_key=api_key directly into the completion() function call.

Challenge B: The "Lab Spec" vs. "Real World" Conflict
Issue: The BANA 275 Lab required min_df=2 (ignore words appearing in only 1 document). However, when testing on small folders (1-2 files), this caused a crash because no word appeared in 2 files. Root Cause: Scikit-learn throws a ValueError when the vocabulary becomes empty after pruning. Solution: Added Dynamic Constraints. safe_min_df = 2 if len(file_data) > 1 else 1 This respects the Lab requirement when possible but falls back to "Survival Mode" for small tests.

Challenge C: Single-Turn Blindness (The "Limit")
Issue: When the AI documented main.py, it didn't know data_loader.py existed. It couldn't explain how the functions interacted. Solution: Implemented a Two-Pass "X-Ray" System:

Pass 1: Scan all files and extract "Skeletons" (function definitions) using RegEx/AST.

Pass 2: Before generating docs for File A, use TF-IDF to find its top 3 neighbors and inject their "Skeletons" into the system prompt.

4. What Worked vs. What Didn't
What Didn't Work:
Default pd.read_excel: It is too fragile for real-world business files with title rows or merged cells. It necessitates a custom "scanner" logic. Processing Files Sequentially (Naive Loop): This hit OpenAI Rate Limits immediately when processing larger folders. We had to add a try...except retry loop with backoff. Ignoring .ipynb: Treating Notebooks as text files resulted in the AI trying to document raw JSON metadata instead of Python code.

What Worked:
Streamlit Session State: Essential for keeping the analysis results visible while the user toggles between "Documentation" and "Sanity Checks." Without it, every click wiped the screen. Skeleton Injection: This was the breakthrough for "Cross-File Logic." Sending just the def function_name() lines (instead of the whole file) saved 90% of tokens while still giving the AI enough context to link files together. Explicit UTF-8 Encoding: Forcing encoding='utf-8' in both reading and writing solved the Windows gbk codec errors permanently.

5. AI-generated % vs. manually written %: Over 90% of the code is written by AI. 
6. Time Saved: At least 2-3 days.