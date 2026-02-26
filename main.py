import contextlib
import requests
import json
import time
import queue
import os
import re
import multiprocessing
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from colorama import init, Fore, Style, Back
from ddgs import DDGS
import io

init(autoreset=True)

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

OLLAMA_API = "http://localhost:11434/api"
MODEL_SOLO = "qwen2.5:14b"
EXCEL_FILE = os.path.join(DATA_DIR, "groupchat_research_results.xlsx")
LOCAL_DB_FILE = os.path.join(DATA_DIR, "local_database.txt")
MAX_TURNS = 15
MAX_SOLO_TURNS = 5
GLOBAL_SEED = 2026

# Initialize local database with English content
if not os.path.exists(LOCAL_DB_FILE):
    with open(LOCAL_DB_FILE, "w", encoding="utf-8") as f:
        f.write("--- LOCAL KNOWLEDGE BASE ---\n")
        f.write("Project_Gamma status: Critical. Access code: 778899\n")
        f.write("Q3 Financial Report: Revenue 1.2M PLN, Costs 0.9M PLN.\n")
        f.write("Server emergency procedure: restart sshd service, then flush iptables.\n")
        f.write("Database admin password is: adm1n_p@ssw0rd_local\n")

ROLE_COLORS = {
    "orchestrator": Fore.MAGENTA + Style.BRIGHT,
    "analyst": Fore.BLUE + Style.BRIGHT,
    "archivist": Fore.CYAN + Style.DIM,
    "coder": Fore.CYAN + Style.BRIGHT,
    "qa_engineer": Fore.LIGHTRED_EX + Style.BRIGHT,
    "solver": Fore.GREEN + Style.BRIGHT,
    "critic": Fore.YELLOW + Style.BRIGHT,
    "devil": Fore.RED + Style.BRIGHT,
    "secretary": Fore.WHITE + Style.BRIGHT,
    "judge": Fore.LIGHTBLACK_EX + Style.BRIGHT,
    "system": Fore.WHITE + Style.DIM,
    "tool": Back.WHITE + Fore.BLACK,
    "solo": Fore.LIGHTYELLOW_EX + Style.BRIGHT
}

PRICE_PER_1M_TOKENS = {
    "qwen2.5:14b": 2.50,
    "gemma2:9b": 0.50,
    "gemma2:2b": 0.15,
    "llama3.1:8b": 0.50,
    "deepseek-r1:14b": 3.00,
    "phi4": 0.20
}


def calculate_cost(model_name, tokens):
    price = PRICE_PER_1M_TOKENS.get(model_name, 1.00)
    return (tokens / 1000000) * price


def run_code_with_timeout(code, queue_obj):
    buffer = io.StringIO()
    try:
        with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
            exec(code, {"__name__": "__main__", "math": __import__("math"), "random": __import__("random")})

        output = buffer.getvalue()
        queue_obj.put(output if output else "[Code executed successfully]")

    except BaseException as e:
        output = buffer.getvalue()
        queue_obj.put(output + f"\n[Terminated/Error: {type(e).__name__}]: {e}")


def execute_python_code(code, timeout=10):
    start_t = time.time()
    print(ROLE_COLORS["tool"] + f" [TOOL] Running Python code (timeout {timeout}s)... ", end="")

    queue_obj = multiprocessing.Queue()
    p = multiprocessing.Process(target=run_code_with_timeout, args=(code, queue_obj))
    p.start()
    p.join(timeout)

    if p.is_alive():
        p.terminate()
        p.join()
        result = f"ERROR: Time limit exceeded ({timeout}s)!"
    else:
        try:
            result = queue_obj.get(timeout=1)
        except queue.Empty:
            result = "CRITICAL ERROR: Process terminated unexpectedly without returning a result (e.g., critical memory error)."

    execution_time = time.time() - start_t
    safe_result_preview = result[:50].replace('\n', ' ')
    print(f"Result: {safe_result_preview}... ({execution_time:.2f}s)" + Style.RESET_ALL)
    return result, execution_time


def search_web(query):
    start_t = time.time()
    print(ROLE_COLORS["tool"] + f" [TOOL] Searching the web: '{query}'... ", end="")
    try:
        results = DDGS().text(query, max_results=3)
        if results:
            summary = "\n".join([f"- {r['title']}: {r['body']}" for r in results])
        else:
            summary = "No results found."
    except Exception as e:
        summary = f"Search error: {e}"

    execution_time = time.time() - start_t
    print(f"Done. ({execution_time:.2f}s)" + Style.RESET_ALL)
    return summary, execution_time


def search_local_docs(query):
    start_t = time.time()
    print(ROLE_COLORS["tool"] + f" [TOOL] Searching local database: '{query}'... ", end="")
    try:
        with open(LOCAL_DB_FILE, "r", encoding="utf-8") as f:
            lines = f.readlines()
        query_terms = query.lower().split()
        results = [line.strip() for line in lines if all(term in line.lower() for term in query_terms)]
        summary = "\n".join(results) if results else "No results in the local database for this query."
    except Exception as e:
        summary = f"Database read error: {e}"

    execution_time = time.time() - start_t
    print(f"Done. ({execution_time:.2f}s)" + Style.RESET_ALL)
    return summary[:2000], execution_time


def print_header(text, color=Fore.WHITE):
    print(color + "\n" + "=" * 60)
    print(color + f" {text}")
    print(color + "=" * 60 + Style.RESET_ALL)


def manage_model(model_name, action="load"):
    if action == "unload":
        try:
            requests.post(f"{OLLAMA_API}/generate", json={"model": model_name, "keep_alive": 0})
        except:
            pass


def call_ollama(prompt, system_prompt, model, agent_role="system"):
    text_color = ROLE_COLORS.get(agent_role, Fore.WHITE)
    print(text_color + f"   [{model}] Generating...", end=" ", flush=True)

    payload = {
        "model": model,
        "prompt": prompt,
        "system": system_prompt,
        "stream": True,
        "options": {
            "temperature": 0.4,
            "num_ctx": 8192,
            "repeat_penalty": 1.1,
            "stop": ["<|eot_id|>", "---", "user:", "assistant:"],
            "seed": GLOBAL_SEED
        }
    }

    full_response = ""
    start_time = time.time()
    print(f"\n{text_color}", end="")
    MAX_CHARS = 10000

    try:
        with requests.post(f"{OLLAMA_API}/generate", json=payload, stream=True) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if line:
                    body = json.loads(line)
                    token = body.get('response', '')

                    print(token, end="", flush=True)
                    full_response += token

                    if len(full_response) > MAX_CHARS:
                        print(Fore.RED + "\n[SYSTEM: INTERRUPTED - RESPONSE TOO LONG]" + Style.RESET_ALL)
                        manage_model(model, action="unload")
                        return {"text": full_response, "generation_time": time.time() - start_time, "tokens": 0,
                                "cost": 0}

                    if body.get('done'):
                        generation_time = time.time() - start_time
                        tokens = body.get('prompt_eval_count', 0) + body.get('eval_count', 0)
                        cost = calculate_cost(model, tokens)
                        print(Style.RESET_ALL)
                        manage_model(model, action="unload")
                        return {"text": full_response, "generation_time": generation_time, "tokens": tokens,
                                "cost": cost}

            print(Fore.RED + "\n[SYSTEM ERROR]: Ollama interrupted generation (missing 'done' flag)." + Style.RESET_ALL)
            manage_model(model, action="unload")
            return {
                "text": full_response if full_response else "ERROR: Empty response",
                "generation_time": time.time() - start_time,
                "tokens": 0,
                "cost": 0
            }

    except Exception as e:
        print(Fore.RED + f"\n!!! API ERROR: {e}")
        return {"text": "ERROR", "generation_time": 0, "tokens": 0, "cost": 0}


def process_tool_calls(response_text):
    tool_output = ""
    tool_time_spent = 0

    if "```python" in response_text:
        try:
            code_match = re.search(r"```python(.*?)```", response_text, re.DOTALL)
            if code_match:
                code = code_match.group(1)
                result, tool_t = execute_python_code(code)
                tool_time_spent += tool_t
                tool_output += f"\n[SYSTEM: Python code execution result]:\n{result}\n"
        except Exception as e:
            tool_output += f"\n[SYSTEM ERROR]: {e}\n"

    if "SEARCH:" in response_text:
        try:
            search_match = re.search(r"SEARCH:(.*)", response_text)
            if search_match:
                query = search_match.group(1).strip()
                result, tool_t = search_web(query)
                tool_time_spent += tool_t
                tool_output += f"\n[SYSTEM: Search results for '{query}']:\n{result}\n"
        except Exception as e:
            tool_output += f"\n[SYSTEM ERROR]: {e}\n"

    if "LOCAL_DOC:" in response_text:
        try:
            search_match = re.search(r"LOCAL_DOC:(.*)", response_text)
            if search_match:
                query = search_match.group(1).strip()
                result, tool_t = search_local_docs(query)
                tool_time_spent += tool_t
                tool_output += f"\n[SYSTEM: Local database results for '{query}']:\n{result}\n"
        except Exception as e:
            tool_output += f"\n[SYSTEM ERROR]: {e}\n"

    return tool_output, tool_time_spent


def run_solo_with_tools(task_query, model_name):
    chat_history = f"TASK: {task_query}"
    total_generation_time = 0
    total_tool_time = 0
    total_tokens = 0
    total_cost = 0.0
    turn_count = 0
    final_answer = ""

    system_prompt = """You are an advanced AI assistant solving complex problems.
You have access to the following tools. Use them by writing EXACTLY on a new line:
1. SEARCH: query (Searches the internet for information)
2. LOCAL_DOC: query (Searches the secure company local database)
3. ```python
print(2+2)
``` (Executes Python code. ALWAYS use print() to see the result).

After using a tool, the system will return the result to you. Analyze it and continue. If you have the final answer and are sure of it, write 'I APPROVE THE RESULT' and provide the solution."""

    print_header(f"START SOLO CHAT WITH TOOLS (Max {MAX_SOLO_TURNS} turns)", Fore.LIGHTYELLOW_EX)

    while turn_count < MAX_SOLO_TURNS:
        turn_count += 1
        print_header(f"SOLO TURN {turn_count}", ROLE_COLORS["solo"])

        agent_input = f"{chat_history}\n\n[Awaiting your analysis and tool usage, or the words I APPROVE THE RESULT if you know the answer]"
        res = call_ollama(agent_input, system_prompt, model_name, "solo")

        total_generation_time += res['generation_time']
        total_tokens += res['tokens']
        total_cost += res['cost']
        response_text = res['text']

        tool_output, tool_t = process_tool_calls(response_text)
        total_tool_time += tool_t

        chat_history += f"\n\n--- Turn {turn_count} ---\n{response_text}"

        if tool_output:
            print(ROLE_COLORS['tool'] + tool_output + Style.RESET_ALL)
            chat_history += tool_output

        final_answer = response_text

        if "I APPROVE THE RESULT" in response_text.upper():
            print(Fore.LIGHTYELLOW_EX + "\n[SOLO] Finished the reasoning process.")
            break

    return {
        "final_answer": final_answer,
        "generation_time": total_generation_time,
        "tool_time": total_tool_time,
        "total_time": total_generation_time + total_tool_time,
        "tokens": total_tokens,
        "cost": total_cost,
        "chat_history": chat_history,
        "turns": turn_count
    }


def select_next_speaker(history, task, agents_config, last_speaker_id):
    if last_speaker_id == "secretary":
        print(Fore.MAGENTA + "\n[SYSTEM] Secretary summarized. END.")
        return "finish", {"generation_time": 0, "tokens": 0, "cost": 0}

    if last_speaker_id == "critic":
        try:
            last_msg = history.split("---")[-1].strip().upper()
            if "APPROVE" in last_msg:
                print(Fore.MAGENTA + "\n[SYSTEM] Critic approved. Calling Secretary.")
                return "secretary", {"generation_time": 0, "tokens": 0, "cost": 0}
        except:
            pass

    devil_count = history.count("Devil's Advocate")
    if last_speaker_id == "solver" and devil_count >= 2:
        print(Fore.MAGENTA + "\n[SYSTEM] Devil has already spoken twice. Forcing Critic.")
        return "critic", {"generation_time": 0, "tokens": 0, "cost": 0}

    if last_speaker_id == "coder":
        print(Fore.MAGENTA + "\n[SYSTEM] Coder finished. Forcing QA Engineer.")
        available_for_prompt = ["qa_engineer"]
    elif last_speaker_id == "solver":
        print(Fore.MAGENTA + "\n[SYSTEM] Solver finished. Forcing Critic or Devil's Advocate.")
        available_for_prompt = ["critic", "devil"]
    else:
        agent_names = [a['id'] for a in agents_config if a['id'] != 'orchestrator']
        available_for_prompt = [name for name in agent_names if name != last_speaker_id]

    orchestrator_cfg = agents_config[0]

    prompt = f"""
    TASK: {task}
    Last spoken: {last_speaker_id}.

    Choose the next step from the list: {available_for_prompt}.
    RULES:
    - After 'analyst' -> 'solver' or 'coder'.
    - After 'archivist' -> 'solver' or 'coder'.
    - After 'coder' -> 'qa_engineer'.
    - After 'qa_engineer' (if code worked correctly) -> 'solver'.
    - After 'qa_engineer' (if code has an error) -> 'coder'.
    - After 'solver' -> 'critic' or 'devil'.
    - After 'critic' (if logical error) -> 'solver'.
    - After 'devil' -> 'solver' or 'coder'.
    - GUARDRAIL: Do not loop the conversation! If you are stuck, call 'secretary'.

    Reply ONLY WITH ONE WORD (Agent ID).
    """

    print(ROLE_COLORS['orchestrator'] + f"\n[ORCHESTRATOR] Deciding (Last: {last_speaker_id})...", end="")
    res = call_ollama(prompt, orchestrator_cfg['system_prompt'], orchestrator_cfg['model'], "orchestrator")
    decision = res['text'].strip().lower()

    chosen_agent = None
    for name in available_for_prompt:
        if name in decision:
            chosen_agent = name
            break

    if "finish" in decision:
        chosen_agent = "finish"

    if not chosen_agent:
        print(Fore.RED + " [SYSTEM] Orchestrator unclear. Using default path.")
        if last_speaker_id == "none":
            chosen_agent = "analyst"
        elif last_speaker_id in ["analyst", "archivist"]:
            chosen_agent = "solver"
        elif last_speaker_id == "coder":
            chosen_agent = "qa_engineer"
        elif last_speaker_id == "qa_engineer":
            chosen_agent = "solver"
        elif last_speaker_id == "solver":
            chosen_agent = "critic"
        elif last_speaker_id == "devil":
            chosen_agent = "solver"
        elif last_speaker_id == "critic":
            chosen_agent = "solver"
        else:
            chosen_agent = "secretary"

    return chosen_agent, res


def run_group_chat_loop(task_query, agents_config, task_id):
    chat_history = ""
    total_generation_time = 0
    total_tool_time = 0
    total_tokens = 0
    total_cost = 0.0
    turn_count = 0
    final_answer = ""
    last_speaker = "none"

    agent_path = []
    snapshot_5 = ""
    snapshot_10 = ""

    print_header(f"START GROUP CHAT (Max {MAX_TURNS})", Fore.MAGENTA)

    while turn_count < MAX_TURNS:
        turn_count += 1
        if turn_count == MAX_TURNS - 1 and last_speaker != "secretary":
            print(Fore.MAGENTA + "\n[SYSTEM] Turn limit reached! Forcing Secretary to summarize.")
            next_agent_id = "secretary"
            orch_res = {"generation_time": 0, "tokens": 0, "cost": 0.0}
        else:
            next_agent_id, orch_res = select_next_speaker(chat_history, task_query, agents_config, last_speaker)

        total_generation_time += orch_res['generation_time']
        total_tokens += orch_res['tokens']
        total_cost += orch_res['cost']

        if next_agent_id == "finish":
            print(Fore.MAGENTA + "\n[ORCHESTRATOR] -> END.")
            final_answer = chat_history.split("---")[-1]
            break

        agent_path.append(next_agent_id)

        selected_agent = next((a for a in agents_config if a['id'] == next_agent_id), None)
        print_header(f"TURN {turn_count}: {selected_agent['role']}", ROLE_COLORS.get(next_agent_id, Fore.WHITE))
        history_parts = chat_history.split("---")
        recent_history = "---".join(history_parts[-5:]) if len(history_parts) > 5 else chat_history
        agent_input = f"TASK: {task_query}\n\nRECENT EVENTS IN HISTORY:\n{recent_history}\n\nYou are {selected_agent['id']}. If you need a tool, use the format:\nSEARCH: query\nLOCAL_DOC: query\n```python\ncode\n```"

        agent_res = call_ollama(agent_input, selected_agent['system_prompt'], selected_agent['model'], next_agent_id)

        total_generation_time += agent_res['generation_time']
        total_tokens += agent_res['tokens']
        total_cost += agent_res['cost']
        response_text = agent_res['text']

        tool_output, tool_t = process_tool_calls(response_text)
        total_tool_time += tool_t

        if tool_output:
            print(ROLE_COLORS['tool'] + tool_output + Style.RESET_ALL)
            response_text += tool_output

        chat_history += f"\n--- {selected_agent['role']} ---\n{response_text}\n"
        final_answer = response_text
        last_speaker = next_agent_id

        if turn_count == 5:
            snapshot_5 = final_answer
        if turn_count == 10:
            snapshot_10 = final_answer

    if not snapshot_5:
        snapshot_5 = final_answer
    if not snapshot_10:
        snapshot_10 = final_answer

    return {
        "final_answer": final_answer,
        "snapshot_5": snapshot_5,
        "snapshot_10": snapshot_10,
        "turns": turn_count,
        "generation_time": total_generation_time,
        "tool_time": total_tool_time,
        "total_time": total_generation_time + total_tool_time,
        "tokens": total_tokens,
        "cost": total_cost,
        "agent_path": " -> ".join(agent_path),
        "chat_history": chat_history
    }


def get_judge_score(task_query, expected, answer_solo, ans_5, ans_10, ans_final):
    print_header("JURY IS EVALUATING", ROLE_COLORS['judge'])
    JURY_MODELS = ["phi4", "qwen2.5:14b", "llama3.1:8b"]
    JUDGE_CTX_SIZE = 4096
    MAX_RETRIES = 2

    solo_scores = []
    scores_5 = []
    scores_10 = []
    scores_final = []
    jury_reasons = []
    hallucination_solo_flags = []
    hallucination_multi_flags = []
    judge_total_cost = 0.0

    prompt_template = f"""
        You are a professional judge in a logic competition.

        TASK: {task_query}
        CORRECT REFERENCE (TRUTH): {expected}

        ANSWER A (Solo Model - ReAct): {answer_solo}
        ANSWER B (Multi - Turn 5): {ans_5}
        ANSWER C (Multi - Turn 10): {ans_10}
        ANSWER D (Multi - Final): {ans_final}

        --- SCORING CRITERIA (0-100 PTS) ---
        100 pts: Perfect answer, matches the REFERENCE, correct reasoning.
        75 pts: Correct result, but the reasoning is imprecise or contains minor inaccuracies.
        50 pts: Partially correct result (e.g., correct number, wrong units) OR correct result but flawed logic (by chance).
        25 pts: Incorrect result, but shows an attempt at correct reasoning (good formula, calculation error).
        0 pts: Incorrect result, flawed logic, hallucinations, or contradicts the REFERENCE.

        --- IMPORTANT: RETURN FORMAT ---
        You MUST return ONLY a pure JSON object. No introductions, no markdown code blocks, no explanations before or after.
        Here is the schema you must use (start with "reason" to log your thoughts):
        {{ 
            "reason": "Enter your detailed step-by-step analysis here",
            "score_solo": X, 
            "score_multi_5": Y,
            "score_multi_10": Z,
            "score_multi_final": W,
            "hallucination_solo": true, 
            "hallucination_multi_final": false
        }}
        """

    for judge_model in JURY_MODELS:
        print(ROLE_COLORS['judge'] + f"   [JURY: {judge_model}] Evaluating...", end="", flush=True)

        current_prompt = prompt_template
        success = False

        for attempt in range(MAX_RETRIES):
            try:
                payload = {
                    "model": judge_model,
                    "prompt": current_prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.0 if attempt == 0 else 0.2,
                        "num_ctx": JUDGE_CTX_SIZE,
                        "seed": GLOBAL_SEED
                    },
                    "format": "json"
                }
                resp_data = requests.post(f"{OLLAMA_API}/generate", json=payload).json()
                response_text = resp_data.get('response', '').strip()

                j_tokens = resp_data.get('prompt_eval_count', 0) + resp_data.get('eval_count', 0)
                judge_total_cost += calculate_cost(judge_model, j_tokens)

                if not response_text:
                    raise ValueError("Empty response from the judge model.")

                # ROBUST JSON EXTRACTION: Strip markdown blocks and apply greedy match
                clean_text = response_text.strip()
                if clean_text.startswith("```json"):
                    clean_text = clean_text[7:]
                elif clean_text.startswith("```"):
                    clean_text = clean_text[3:]
                if clean_text.endswith("```"):
                    clean_text = clean_text[:-3]

                json_match = re.search(r'\{.*\}', clean_text, re.DOTALL)

                if not json_match:
                    raise ValueError(f"No JSON format in response: {clean_text[:50]}...")

                try:
                    result = json.loads(json_match.group(0))
                except json.JSONDecodeError as e:
                    raise ValueError(f"JSON parsing error: {e}. Returned text: {json_match.group(0)[:50]}...")

                s_solo = int(result.get("score_solo") or 0)
                s_5 = int(result.get("score_multi_5") or 0)
                s_10 = int(result.get("score_multi_10") or 0)
                s_fin = int(result.get("score_multi_final") or 0)

                h_solo = bool(result.get("hallucination_solo", False))
                h_multi = bool(result.get("hallucination_multi_final", False))
                reason = result.get("reason", "None")

                solo_scores.append(s_solo)
                scores_5.append(s_5)
                scores_10.append(s_10)
                scores_final.append(s_fin)

                hallucination_solo_flags.append(h_solo)
                hallucination_multi_flags.append(h_multi)
                jury_reasons.append(f"[{judge_model}]: {reason}")

                if attempt > 0:
                    print(Fore.GREEN + f" [FIXED IN ATTEMPT {attempt + 1}]", end="")

                print(f" -> Solo: {s_solo} | Multi(5): {s_5} | Multi(10): {s_10} | Multi(Fin): {s_fin}")
                success = True
                break

            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    print(Fore.YELLOW + f" -> Error: {e}. Reporting to model and retrying... " + ROLE_COLORS[
                        'judge'], end="", flush=True)
                    current_prompt = prompt_template + f"\n\n[SYSTEM ERROR]: Your previous response caused a Python error: {e}\nYOU MUST RETURN ONLY A CLEAN JSON OBJECT. No introductory text!"
                else:
                    print(Fore.RED + f" -> Critical judge error {judge_model}: {e}")

        if not success:
            solo_scores.append(0)
            scores_5.append(0)
            scores_10.append(0)
            scores_final.append(0)
            hallucination_solo_flags.append(False)
            hallucination_multi_flags.append(False)

        manage_model(judge_model, action="unload")

    avg_solo = round(sum(solo_scores) / len(solo_scores), 1) if solo_scores else 0
    avg_5 = round(sum(scores_5) / len(scores_5), 1) if scores_5 else 0
    avg_10 = round(sum(scores_10) / len(scores_10), 1) if scores_10 else 0
    avg_fin = round(sum(scores_final) / len(scores_final), 1) if scores_final else 0

    final_h_solo = sum(hallucination_solo_flags) > (len(JURY_MODELS) / 2)
    final_h_multi = sum(hallucination_multi_flags) > (len(JURY_MODELS) / 2)
    final_reason = " | ".join(jury_reasons)

    print(ROLE_COLORS['judge'] + "-" * 30)
    print(f"   [FINAL VERDICT] Solo: {avg_solo} | Multi Final: {avg_fin} | Jury Cost: ${judge_total_cost:.4f}")
    print(ROLE_COLORS['judge'] + "-" * 30 + Style.RESET_ALL)

    return {
        "score_solo": avg_solo,
        "score_multi_5": avg_5,
        "score_multi_10": avg_10,
        "score_multi_final": avg_fin,
        "hallucination_solo": final_h_solo,
        "hallucination_multi": final_h_multi,
        "reason": final_reason,
        "judge_cost": judge_total_cost
    }


def visualize_results(df, run_timestamp):
    if df.empty: return
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Single Model (ReAct) vs Multi-Agent Swarm', fontsize=16)

    avg_scores = [df['score_solo'].mean(), df['score_multi_final'].mean()]
    axes[0, 0].bar(['Solo (ReAct)', 'Multi-Agent'], avg_scores, color=['gray', 'green'])
    axes[0, 0].set_title('Average Score')

    solo_gen = df['solo_generation_time'].mean()
    multi_gen = df['multi_generation_time'].mean()
    solo_tool = df['solo_tool_time'].mean()
    multi_tool = df['multi_tool_time'].mean()

    axes[0, 1].bar(['Solo', 'Group'], [solo_gen, multi_gen], color=['gray', 'orange'], label='LLM Generation Time')
    axes[0, 1].bar(['Solo', 'Group'], [solo_tool, multi_tool], bottom=[solo_gen, multi_gen], color='red', alpha=0.7,
                   label='Tool Time (I/O)')
    axes[0, 1].set_title('Average Time: Generation vs Tools')
    axes[0, 1].legend()

    avg_efficiency = [
        (df['score_solo'].mean() / (df['solo_tokens'].mean() / 1000)) if df['solo_tokens'].mean() > 0 else 0,
        (df['score_multi_final'].mean() / (df['multi_tokens'].mean() / 1000)) if df['multi_tokens'].mean() > 0 else 0
    ]
    axes[0, 2].bar(['Solo', 'Group'], avg_efficiency, color=['gray', 'purple'])
    axes[0, 2].set_title('Token Efficiency (Pts / 1k Tokens)')

    avg_costs = [df['solo_cost_usd'].mean(), df['multi_cost_usd'].mean()]
    axes[1, 0].bar(['Solo', 'Group Total'], avg_costs, color=['gray', 'gold'])
    axes[1, 0].set_title('Average Cost per Task (USD)')
    axes[1, 0].set_ylabel('USD')

    stages = ['Turn 5', 'Turn 10', 'Final']
    scores_evolution = [df['score_multi_5'].mean(), df['score_multi_10'].mean(), df['score_multi_final'].mean()]
    axes[1, 1].plot(stages, scores_evolution, marker='o', color='b', linestyle='-', linewidth=2, markersize=8)
    axes[1, 1].set_title('Multi-Agent Score Evolution')
    axes[1, 1].set_ylim(0, 100)
    axes[1, 1].grid(True, linestyle='--', alpha=0.7)

    axes[1, 2].axis('off')

    plt.tight_layout()

    chart_path = os.path.join(DATA_DIR, f"charts_{run_timestamp}.png")
    plt.savefig(chart_path, dpi=300)
    print(Fore.GREEN + f"\n[SYSTEM] Charts saved to {chart_path}")

    plt.show()


def generate_html_log(task_id, run_timestamp, chat_history_solo, chat_history_multi, solo_cost, multi_cost):
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Task log: {task_id}</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f4f4f9; color: #333; margin: 20px; }}
        h2 {{ color: #4a4a4a; border-bottom: 2px solid #ddd; padding-bottom: 10px; }}
        .stats {{ background: #e2e8f0; padding: 10px; border-radius: 5px; margin-bottom: 20px; font-weight: bold; }}
        .section-title {{ background: #0056b3; color: white; padding: 10px; border-radius: 5px; margin-top: 30px; }}
        .message {{ background: #fff; border: 1px solid #e1e4e8; border-radius: 6px; padding: 15px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }}
        .role {{ font-weight: bold; color: #0056b3; margin-bottom: 10px; font-size: 1.1em; }}
        .content {{ white-space: pre-wrap; line-height: 1.6; font-size: 0.95em; }}
    </style>
</head>
<body>
    <h2>Task: {task_id} ({run_timestamp})</h2>
    <div class="stats">Solo Cost: ${solo_cost:.4f} | Multi Cost (with judges): ${multi_cost:.4f}</div>

    <div class="section-title">Solo Model Log (ReAct)</div>
"""
    html_content += f"<div class='message'><div class='content'>{chat_history_solo}</div></div>"

    html_content += "<div class='section-title'>Agent Group Log</div>"

    parts = chat_history_multi.split('---')
    for part in parts:
        if part.strip():
            lines = part.strip().split('\n', 1)
            role = lines[0].strip()
            content = lines[1].strip() if len(lines) > 1 else ""
            html_content += f"""
    <div class="message">
        <div class="role">{role}</div>
        <div class="content">{content}</div>
    </div>
"""
    html_content += "</body></html>"

    filepath = os.path.join(DATA_DIR, f"log_{run_timestamp}_{task_id}.html")
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(html_content)


def run_research():
    tasks = json.load(open('tasks.json', 'r', encoding='utf-8'))
    agents = json.load(open('agents.json', 'r', encoding='utf-8'))

    run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    sheet_name = f"Run_{run_timestamp}"
    results_data = []

    for task in tasks:
        print_header(f"TASK: {task['id']}", Fore.WHITE)

        solo_res = run_solo_with_tools(task['query'], MODEL_SOLO)

        group_res = run_group_chat_loop(task['query'], agents, task['id'])

        scores = get_judge_score(
            task['query'],
            task['expected_answer'],
            solo_res['final_answer'],
            group_res['snapshot_5'],
            group_res['snapshot_10'],
            group_res['final_answer']
        )

        solo_eff = round(scores.get('score_solo', 0) / (solo_res['tokens'] / 1000), 2) if solo_res['tokens'] > 0 else 0
        multi_eff = round(scores.get('score_multi_final', 0) / (group_res['tokens'] / 1000), 2) if group_res[
                                                                                                       'tokens'] > 0 else 0

        multi_total_cost = group_res['cost'] + scores['judge_cost']

        row = {
            "task_id": task['id'],
            "score_solo": scores.get('score_solo', 0),
            "score_multi_5": scores.get('score_multi_5', 0),
            "score_multi_10": scores.get('score_multi_10', 0),
            "score_multi_final": scores.get('score_multi_final', 0),
            "hallucination_solo": scores.get('hallucination_solo', False),
            "hallucination_multi": scores.get('hallucination_multi', False),
            "solo_time": round(solo_res['total_time'], 2),
            "solo_generation_time": round(solo_res['generation_time'], 2),
            "solo_tool_time": round(solo_res['tool_time'], 2),
            "multi_time": round(group_res['total_time'], 2),
            "multi_generation_time": round(group_res['generation_time'], 2),
            "multi_tool_time": round(group_res['tool_time'], 2),
            "solo_tokens": solo_res['tokens'],
            "multi_tokens": group_res['tokens'],
            "solo_cost_usd": round(solo_res['cost'], 4),
            "multi_cost_usd": round(multi_total_cost, 4),
            "solo_token_efficiency": solo_eff,
            "multi_token_efficiency": multi_eff,
            "solo_turns": solo_res['turns'],
            "multi_turns": group_res['turns'],
            "agent_path": group_res.get('agent_path', ''),
            "judge_reason": scores.get('reason', '')
        }

        results_data.append(row)

        generate_html_log(task['id'], run_timestamp, solo_res['chat_history'], group_res.get('chat_history', ''),
                          solo_res['cost'], multi_total_cost)

        df = pd.DataFrame(results_data)
        if os.path.exists(EXCEL_FILE):
            try:
                with pd.ExcelWriter(EXCEL_FILE, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
            except ValueError:
                with pd.ExcelWriter(EXCEL_FILE, engine='openpyxl', mode='a') as writer:
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
        else:
            with pd.ExcelWriter(EXCEL_FILE, engine='openpyxl', mode='w') as writer:
                df.to_excel(writer, sheet_name=sheet_name, index=False)

        print(Fore.GREEN + f"\n[SYSTEM] State saved in file {EXCEL_FILE} (Sheet: {sheet_name})")

    visualize_results(pd.DataFrame(results_data), run_timestamp)


if __name__ == "__main__":
    run_research()