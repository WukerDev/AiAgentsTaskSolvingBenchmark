import contextlib
import requests
import json
import time
import os
import re
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from colorama import init, Fore, Style, Back
from duckduckgo_search import DDGS
import io

init(autoreset=True)

OLLAMA_API = "http://localhost:11434/api"
MODEL_SOLO = "llama3.1:8b"
MODEL_JUDGE = "qwen2.5:14b"
EXCEL_FILE = "badanie_groupchat_wyniki.xlsx"
MAX_TURNS = 15

ROLE_COLORS = {
    "orchestrator": Fore.MAGENTA + Style.BRIGHT,
    "analyst": Fore.BLUE + Style.BRIGHT,
    "coder": Fore.CYAN + Style.BRIGHT,
    "solver": Fore.GREEN + Style.BRIGHT,
    "critic": Fore.YELLOW + Style.BRIGHT,
    "devil": Fore.RED + Style.BRIGHT,
    "secretary": Fore.WHITE + Style.BRIGHT,
    "judge": Fore.LIGHTBLACK_EX + Style.BRIGHT,
    "system": Fore.WHITE + Style.DIM,
    "tool": Back.WHITE + Fore.BLACK
}


def execute_python_code(code):
    print(ROLE_COLORS["tool"] + " [TOOL] Uruchamiam kod Python... ", end="")
    buffer = io.StringIO()
    try:
        with contextlib.redirect_stdout(buffer):
            exec(code, {"__name__": "__main__", "math": __import__("math"), "random": __import__("random")})
        result = buffer.getvalue()
        if not result:
            result = "[Kod wykonany poprawnie, brak outputu (użyj print)]"
    except Exception as e:
        result = f"Błąd wykonania kodu: {e}"

    print(f"Wynik: {result[:50]}..." + Style.RESET_ALL)
    return result


def search_web(query):
    print(ROLE_COLORS["tool"] + f" [TOOL] Szukam w sieci: '{query}'... ", end="")
    try:
        results = DDGS().text(query, max_results=3)
        if results:
            summary = "\n".join([f"- {r['title']}: {r['body']}" for r in results])
        else:
            summary = "Brak wyników."
    except Exception as e:
        summary = f"Błąd wyszukiwania: {e}"

    print("Gotowe." + Style.RESET_ALL)
    return summary


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
    print(text_color + f"   [{model}] Generuje...", end=" ", flush=True)

    payload = {
        "model": model,
        "prompt": prompt,
        "system": system_prompt,
        "stream": True,
        "options": {
            "temperature": 0.4,
            "num_ctx": 8192,
            "repeat_penalty": 1.1,
            "stop": ["<|eot_id|>", "---", "user:", "assistant:"]
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
                        print(Fore.RED + "\n[SYSTEM: PRZERWANO - ZA DŁUGA WYPOWIEDŹ]" + Style.RESET_ALL)
                        manage_model(model, action="unload")
                        return {"text": full_response, "total_time": time.time() - start_time, "tokens": 0}

                    if body.get('done'):
                        total_time = time.time() - start_time
                        tokens = body.get('prompt_eval_count', 0) + body.get('eval_count', 0)
                        print(Style.RESET_ALL)
                        manage_model(model, action="unload")
                        return {"text": full_response, "total_time": total_time, "tokens": tokens}
    except Exception as e:
        print(Fore.RED + f"\n!!! BŁĄD API: {e}")
        return {"text": "ERROR", "total_time": 0, "tokens": 0}


def select_next_speaker(history, task, agents_config, last_speaker_id):
    if last_speaker_id == "solver":
        print(Fore.MAGENTA + "\n[SYSTEM] Solver skończył. Wymuszam Krytyka lub Adwokata Diabła.")
        pass
    if last_speaker_id == "secretary":
        print(Fore.MAGENTA + "\n[SYSTEM] Sekretarz podsumował. KONIEC.")
        return "finish", {"total_time": 0, "tokens": 0}
    if last_speaker_id == "critic":
        try:
            last_msg = history.split("---")[-1].strip().upper()
            if "ZATWIERDZAM" in last_msg:
                print(Fore.MAGENTA + "\n[SYSTEM] Krytyk zatwierdził. Wołam Sekretarza.")
                return "secretary", {"total_time": 0, "tokens": 0}
        except:
            pass

    orchestrator_cfg = agents_config[0]
    agent_names = [a['id'] for a in agents_config if a['id'] != 'orchestrator']
    available_for_prompt = [name for name in agent_names if name != last_speaker_id]

    prompt = f"""
    ZADANIE: {task}
    Ostatnio mówił: {last_speaker_id}.

    Wybierz następnego krok z listy: {available_for_prompt}.
    ZASADY:
    - Po 'analyst' -> 'solver' lub 'coder'.
    - Po 'solver' -> 'critic' lub 'devil'.
    - Po 'coder' -> 'solver'.
    - Po 'critic' (jeśli błąd) -> 'solver' lub 'coder'.
    - Po 'devil' -> 'solver' lub 'coder'.
    - STRAŻNIK: Nie zapętlaj rozmowy! Jeśli utknęliście, wołaj 'secretary'.

    Odpowiedz TYLKO JEDNYM SŁOWEM (ID agenta).
    """

    print(ROLE_COLORS['orchestrator'] + f"\n[ORCHESTRATOR] Decyduje (Ostatni: {last_speaker_id})...", end="")
    res = call_ollama(prompt, orchestrator_cfg['system_prompt'], orchestrator_cfg['model'], "orchestrator")
    decision = res['text'].strip().lower()
    chosen_agent = None
    for name in agent_names:
        if name in decision and name != last_speaker_id:
            chosen_agent = name
            break

    if "finish" in decision:
        chosen_agent = "finish"

    if not chosen_agent:
        print(Fore.RED + " [SYSTEM] Orchestrator niejasny. Używam ścieżki domyślnej.")
        if last_speaker_id == "none":
            chosen_agent = "analyst"
        elif last_speaker_id == "analyst":
            chosen_agent = "solver"
        elif last_speaker_id == "coder":
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
    total_time = 0
    total_tokens = 0
    turn_count = 0
    final_answer = ""
    last_speaker = "none"

    agent_path = []
    snapshot_5 = ""
    snapshot_10 = ""

    print_header(f"START GROUP CHAT (Max {MAX_TURNS})", Fore.MAGENTA)

    while turn_count < MAX_TURNS:
        turn_count += 1
        next_agent_id, orch_res = select_next_speaker(chat_history, task_query, agents_config, last_speaker)
        total_time += orch_res['total_time']
        total_tokens += orch_res['tokens']

        if next_agent_id == "finish":
            print(Fore.MAGENTA + "\n[ORCHESTRATOR] -> KONIEC.")
            final_answer = chat_history.split("---")[-1]
            break

        agent_path.append(next_agent_id)

        selected_agent = next((a for a in agents_config if a['id'] == next_agent_id), None)
        print_header(f"TURA {turn_count}: {selected_agent['role']}", ROLE_COLORS.get(next_agent_id, Fore.WHITE))
        agent_input = f"ZADANIE: {task_query}\n\nHISTORIA:\n{chat_history}\n\nJesteś {selected_agent['id']}. Jeśli potrzebujesz narzędzia, użyj formatu:\nSEARCH: zapytanie\n```python\nkod\n```"
        agent_res = call_ollama(agent_input, selected_agent['system_prompt'], selected_agent['model'], next_agent_id)

        total_time += agent_res['total_time']
        total_tokens += agent_res['tokens']
        response_text = agent_res['text']
        tool_output = ""

        if (next_agent_id == "coder" or next_agent_id == "solver") and "```python" in response_text:
            try:
                code_match = re.search(r"```python(.*?)```", response_text, re.DOTALL)
                if code_match:
                    code = code_match.group(1)
                    result = execute_python_code(code)
                    tool_output = f"\n[SYSTEM: Wynik uruchomienia kodu Python]:\n{result}\n"
            except Exception as e:
                tool_output = f"\n[SYSTEM ERROR]: {e}\n"

        if next_agent_id == "analyst" and "SEARCH:" in response_text:
            try:
                search_match = re.search(r"SEARCH:(.*)", response_text)
                if search_match:
                    query = search_match.group(1).strip()
                    result = search_web(query)
                    tool_output = f"\n[SYSTEM: Wyniki wyszukiwania dla '{query}']:\n{result}\n"
            except Exception as e:
                tool_output = f"\n[SYSTEM ERROR]: {e}\n"

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
        "total_time": total_time,
        "tokens": total_tokens,
        "agent_path": " -> ".join(agent_path),
        "chat_history": chat_history
    }


def get_judge_score(task_query, expected, answer_solo, ans_5, ans_10, ans_final):
    print_header("KOMISJA SĘDZIOWSKA OCENIA", ROLE_COLORS['judge'])
    JURY_MODELS = ["phi4", "qwen2.5:14b", "gemma2:9b"]
    JUDGE_CTX_SIZE = 4096

    solo_scores = []
    scores_5 = []
    scores_10 = []
    scores_final = []
    jury_reasons = []
    hallucination_solo_flags = []
    hallucination_multi_flags = []

    prompt_template = f"""
    Jesteś profesjonalnym sędzią w konkursie logicznym.

    ZADANIE: {task_query}
    POPRAWNY WZORZEC (PRAWDA): {expected}

    ODPOWIEDŹ A (Solo Model): {answer_solo}
    ODPOWIEDŹ B (Multi - Tura 5): {ans_5}
    ODPOWIEDŹ C (Multi - Tura 10): {ans_10}
    ODPOWIEDŹ D (Multi - Finał): {ans_final}

    --- KRYTERIA OCENY (0-100 PKT) ---
    100 pkt: Odpowiedź idealna, zgodna z WZORCEM, poprawne uzasadnienie.
    75 pkt: Wynik poprawny, ale uzasadnienie mało precyzyjne lub zawiera drobne nieścisłości.
    50 pkt: Wynik częściowo poprawny (np. dobra liczba, złe jednostki) LUB wynik dobry, ale błędna logika (przypadek).
    25 pkt: Wynik błędny, ale widać próbę poprawnego rozumowania (dobry wzór, błąd rachunkowy).
    0 pkt: Wynik błędny, logika błędna, halucynacje lub zaprzeczenie WZORCOWI.

    --- ZADANIE DODATKOWE ---
    Oceń, czy modele zaczęły "halucynować". Zaznacz to jako `true` lub `false`.

    Twoim zadaniem jest ocenić odpowiedzi zgodnie z WZORCEM.
    Zwróć TYLKO czysty JSON:
    {{ 
        "score_solo": X, 
        "score_multi_5": Y,
        "score_multi_10": Z,
        "score_multi_final": W,
        "hallucination_solo": true/false, 
        "hallucination_multi_final": true/false, 
        "reason": "Krótkie uzasadnienie postępu" 
    }}
    """

    for judge_model in JURY_MODELS:
        print(ROLE_COLORS['judge'] + f"   [JURY: {judge_model}] Ocenia...", end="", flush=True)
        try:
            payload = {
                "model": judge_model,
                "prompt": prompt_template,
                "stream": False,
                "options": {
                    "temperature": 0.0,
                    "num_ctx": JUDGE_CTX_SIZE
                },
                "format": "json"
            }
            resp = requests.post(f"{OLLAMA_API}/generate", json=payload).json()
            response_text = resp.get('response', '')
            result = json.loads(response_text)

            s_solo = int(result.get("score_solo", 0))
            s_5 = int(result.get("score_multi_5", 0))
            s_10 = int(result.get("score_multi_10", 0))
            s_fin = int(result.get("score_multi_final", 0))

            h_solo = bool(result.get("hallucination_solo", False))
            h_multi = bool(result.get("hallucination_multi_final", False))
            reason = result.get("reason", "Brak")

            solo_scores.append(s_solo)
            scores_5.append(s_5)
            scores_10.append(s_10)
            scores_final.append(s_fin)

            hallucination_solo_flags.append(h_solo)
            hallucination_multi_flags.append(h_multi)
            jury_reasons.append(f"[{judge_model}]: {reason}")

            print(f" -> Solo: {s_solo} | Multi(5): {s_5} | Multi(10): {s_10} | Multi(Fin): {s_fin}")
            manage_model(judge_model, action="unload")

        except Exception as e:
            print(Fore.RED + f" -> Błąd sędziego {judge_model}: {e}")
            solo_scores.append(0)
            scores_5.append(0);
            scores_10.append(0);
            scores_final.append(0)
            hallucination_solo_flags.append(False);
            hallucination_multi_flags.append(False)

    avg_solo = round(sum(solo_scores) / len(solo_scores), 1) if solo_scores else 0
    avg_5 = round(sum(scores_5) / len(scores_5), 1) if scores_5 else 0
    avg_10 = round(sum(scores_10) / len(scores_10), 1) if scores_10 else 0
    avg_fin = round(sum(scores_final) / len(scores_final), 1) if scores_final else 0

    final_h_solo = sum(hallucination_solo_flags) > (len(JURY_MODELS) / 2)
    final_h_multi = sum(hallucination_multi_flags) > (len(JURY_MODELS) / 2)
    final_reason = " | ".join(jury_reasons)

    print(ROLE_COLORS['judge'] + "-" * 30)
    print(f"   [WERDYKT KOŃCOWY] Solo: {avg_solo} | Multi Finał: {avg_fin}")
    print(ROLE_COLORS['judge'] + "-" * 30 + Style.RESET_ALL)

    return {
        "score_solo": avg_solo,
        "score_multi_5": avg_5,
        "score_multi_10": avg_10,
        "score_multi_final": avg_fin,
        "hallucination_solo": final_h_solo,
        "hallucination_multi": final_h_multi,
        "reason": final_reason
    }


def visualize_results(df):
    if df.empty: return
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Solo vs Group Chat - Analiza Zbiorcza', fontsize=16)
    avg_scores = [df['score_solo'].mean(), df['score_multi_final'].mean()]
    axes[0, 0].bar(['Solo', 'Group Final'], avg_scores, color=['gray', 'green'])
    axes[0, 0].set_title('Średnie Punkty')
    avg_times = df[['solo_time', 'multi_time']].mean()
    axes[0, 1].bar(['Solo', 'Group'], avg_times, color=['gray', 'orange'])
    axes[0, 1].set_title('Średni Czas (s)')
    avg_efficiency = df[['solo_token_efficiency', 'multi_token_efficiency']].mean()
    axes[1, 0].bar(['Solo', 'Group'], avg_efficiency, color=['gray', 'purple'])
    axes[1, 0].set_title('Token Efficiency (Pkt / 1k Tokenów)')
    stages = ['Tura 5', 'Tura 10', 'Finał']
    scores_evolution = [df['score_multi_5'].mean(), df['score_multi_10'].mean(), df['score_multi_final'].mean()]
    axes[1, 1].plot(stages, scores_evolution, marker='o', color='b', linestyle='-', linewidth=2, markersize=8)
    axes[1, 1].set_title('Ewolucja wyników Multi (Convergence)')
    axes[1, 1].set_ylim(0, 100)
    axes[1, 1].grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()

    print("\n" + Fore.CYAN + df[['task_id', 'turns', 'score_multi_final', 'agent_path']].to_string(index=False))


def run_research():
    tasks = json.load(open('tasks.json', 'r', encoding='utf-8'))
    agents = json.load(open('agents.json', 'r', encoding='utf-8'))

    run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    sheet_name = f"Run_{run_timestamp}"
    results_data = []

    for task in tasks:
        print_header(f"ZADANIE: {task['id']}", Fore.WHITE)
        print(Fore.WHITE + ">>> Testowanie modelu SOLO...")
        solo_res = call_ollama(task['query'], "Pomocny asystent", MODEL_SOLO, "system")
        group_res = run_group_chat_loop(task['query'], agents, task['id'])
        scores = get_judge_score(
            task['query'],
            task['expected_answer'],
            solo_res['text'],
            group_res['snapshot_5'],
            group_res['snapshot_10'],
            group_res['final_answer']
        )

        solo_eff = round(scores.get('score_solo', 0) / (solo_res['tokens'] / 1000), 2) if solo_res['tokens'] > 0 else 0
        multi_eff = round(scores.get('score_multi_final', 0) / (group_res['tokens'] / 1000), 2) if group_res[
                                                                                                       'tokens'] > 0 else 0

        row = {
            "task_id": task['id'],
            "score_solo": scores.get('score_solo', 0),
            "score_multi_5": scores.get('score_multi_5', 0),
            "score_multi_10": scores.get('score_multi_10', 0),
            "score_multi_final": scores.get('score_multi_final', 0),
            "hallucination_solo": scores.get('hallucination_solo', False),
            "hallucination_multi": scores.get('hallucination_multi', False),
            "solo_time": round(solo_res['total_time'], 2),
            "multi_time": round(group_res['total_time'], 2),
            "solo_tokens": solo_res['tokens'],
            "multi_tokens": group_res['tokens'],
            "solo_token_efficiency": solo_eff,
            "multi_token_efficiency": multi_eff,
            "turns": group_res['turns'],
            "agent_path": group_res.get('agent_path', ''),
            "judge_reason": scores.get('reason', ''),
            "chat_history": group_res.get('chat_history', '')
        }

        results_data.append(row)

        with open(f"log_{run_timestamp}_{task['id']}.txt", "w", encoding="utf-8") as log_file:
            log_file.write(group_res.get('chat_history', ''))

    df = pd.DataFrame(results_data)

    if os.path.exists(EXCEL_FILE):
        with pd.ExcelWriter(EXCEL_FILE, engine='openpyxl', mode='a') as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    else:
        with pd.ExcelWriter(EXCEL_FILE, engine='openpyxl', mode='w') as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=False)

    print(Fore.GREEN + f"\n[SYSTEM] Wyniki zapisano pomyślnie w arkuszu '{sheet_name}' w pliku {EXCEL_FILE}")

    visualize_results(df)


if __name__ == "__main__":
    run_research()