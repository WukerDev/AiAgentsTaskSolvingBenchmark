import contextlib

import requests
import json
import time
import csv
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
CSV_FILE = f"badanie_groupchat_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
MAX_TURNS = 15

ROLE_COLORS = {
    "orchestrator": Fore.MAGENTA + Style.BRIGHT,
    "analyst":      Fore.BLUE + Style.BRIGHT,
    "coder":        Fore.CYAN + Style.BRIGHT,
    "solver":       Fore.GREEN + Style.BRIGHT,
    "critic":       Fore.YELLOW + Style.BRIGHT,
    "secretary":    Fore.WHITE + Style.BRIGHT,
    "judge":        Fore.LIGHTBLACK_EX + Style.BRIGHT,
    "system":       Fore.WHITE + Style.DIM,
    "tool":         Back.WHITE + Fore.BLACK
}


def execute_python_code(code):
    """Bezpieczniejsze (względnie) wykonanie kodu Python z przechwyceniem wyniku."""
    print(ROLE_COLORS["tool"] + " [TOOL] Uruchamiam kod Python... ", end="")
    buffer = io.StringIO()
    try:
        # Przechwytujemy STDOUT (to co printuje kod)
        with contextlib.redirect_stdout(buffer):
            # Ostrzeżenie: exec() jest niebezpieczne w produkcji, ale OK do lokalnych badań
            exec(code, {"__name__": "__main__", "math": __import__("math"), "random": __import__("random")})
        result = buffer.getvalue()
        if not result:
            result = "[Kod wykonany poprawnie, brak outputu (użyj print)]"
    except Exception as e:
        result = f"Błąd wykonania kodu: {e}"

    print(f"Wynik: {result[:50]}..." + Style.RESET_ALL)
    return result


def search_web(query):
    """Wyszukiwanie w DuckDuckGo"""
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
    """Drukuje ładny nagłówek w ramce"""
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
            "temperature": 0.4,  # Troszkę wyższa temperatura pomaga wybić się z pętli
            "num_ctx": 8192,
            "repeat_penalty": 1.1,
            "stop": ["<|eot_id|>", "---", "user:", "assistant:"]  # Więcej tokenów stopu
        }
    }

    full_response = ""
    start_time = time.time()
    print(f"\n{text_color}", end="")

    # HAMULEC BEZPIECZEŃSTWA (Max znaków na wypowiedź)
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

                    # --- NOWOŚĆ: SPRAWDZANIE DŁUGOŚCI ---
                    if len(full_response) > MAX_CHARS:
                        print(Fore.RED + "\n[SYSTEM: PRZERWANO - ZA DŁUGA WYPOWIEDŹ]" + Style.RESET_ALL)
                        manage_model(model, action="unload")
                        return {"text": full_response, "total_time": time.time() - start_time, "tokens": 0}
                    # ------------------------------------

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
    # --- 1. TWARDE ZASADY (Hard Rules) ---
    if last_speaker_id == "solver":
        print(Fore.MAGENTA + "\n[SYSTEM] Solver skończył. Wymuszam Krytyka.")
        # Zwracamy sztucznie, pomijając model
        return "critic", {"total_time": 0, "tokens": 0}

        # Jeśli Sekretarz już mówił, to koniec.
    if last_speaker_id == "secretary":
        print(Fore.MAGENTA + "\n[SYSTEM] Sekretarz podsumował. KONIEC.")
        return "finish", {"total_time": 0, "tokens": 0}

        # Wykrywanie zatwierdzenia przez Krytyka
    if last_speaker_id == "critic":
        try:
            last_msg = history.split("---")[-1].strip().upper()
            if "ZATWIERDZAM" in last_msg:
                print(Fore.MAGENTA + "\n[SYSTEM] Krytyk zatwierdził. Wołam Sekretarza.")
                return "secretary", {"total_time": 0, "tokens": 0}
        except:
            pass

    # --- 2. LOGIKA ORKIESTRATORA (LLM) ---
    orchestrator_cfg = agents_config[0]
    agent_names = [a['id'] for a in agents_config if a['id'] != 'orchestrator']

    # Usuwamy ostatniego mówcę z listy dostępnych dla LLM
    available_for_prompt = [name for name in agent_names if name != last_speaker_id]

    prompt = f"""
    ZADANIE: {task}
    Ostatnio mówił: {last_speaker_id}.

    Wybierz następnego krok z listy: {available_for_prompt}.
    ZASADY:
    - Po 'analyst' -> 'solver' lub 'coder'.
    - Po 'solver' -> 'critic'.
    - Po 'coder' -> 'solver'.
    - Po 'critic' (jeśli błąd) -> 'solver'.

    Odpowiedz TYLKO JEDNYM SŁOWEM (ID agenta).
    """

    print(ROLE_COLORS['orchestrator'] + f"\n[ORCHESTRATOR] Decyduje (Ostatni: {last_speaker_id})...", end="")
    res = call_ollama(prompt, orchestrator_cfg['system_prompt'], orchestrator_cfg['model'], "orchestrator")

    decision = res['text'].strip().lower()

    # --- 3. FILTROWANIE DECYZJI ---
    chosen_agent = None

    # Sprawdzamy, czy model wybrał poprawne ID
    for name in agent_names:
        # Dodatkowe zabezpieczenie: upewniamy się, że to nie jest część zdania "Dostępni: analyst"
        # Szukamy nazwy agenta, ale ignorujemy, jeśli to ten sam co ostatnio
        if name in decision and name != last_speaker_id:
            chosen_agent = name
            break

    if "finish" in decision:
        chosen_agent = "finish"

    # --- 4. ŁAŃCUCH ZAPASOWY (Fallback Chain) ---
    # Jeśli model zgłupiał (wybrał None, wybrał tego samego, albo gadał głupoty)
    # Python przejmuje stery i popycha kolejkę logicznie do przodu.

    if not chosen_agent:
        print(Fore.RED + " [SYSTEM] Orchestrator niejasny. Używam ścieżki domyślnej.")

        if last_speaker_id == "none":
            chosen_agent = "analyst"
        elif last_speaker_id == "analyst":
            chosen_agent = "solver"  # Domyślnie po analizie rozwiązuj
        elif last_speaker_id == "coder":
            chosen_agent = "solver"  # Po kodzie interpretuj
        elif last_speaker_id == "solver":
            chosen_agent = "critic"  # Po rozwiązaniu sprawdź
        elif last_speaker_id == "devil":
            chosen_agent = "solver"
        elif last_speaker_id == "critic":
            chosen_agent = "solver"  # Jeśli krytyk nie zatwierdził (bo hard rule wyżej nie zadziałał), to poprawka
        else:
            chosen_agent = "secretary"  # Ostateczność

    return chosen_agent, res


def run_group_chat_loop(task_query, agents_config):
    chat_history = ""
    total_time = 0
    total_tokens = 0
    turn_count = 0
    final_answer = ""
    last_speaker = "none"

    print_header(f"START GROUP CHAT (Max {MAX_TURNS})", Fore.MAGENTA)

    while turn_count < MAX_TURNS:
        turn_count += 1

        # 1. Orchestrator
        next_agent_id, orch_res = select_next_speaker(chat_history, task_query, agents_config, last_speaker)
        total_time += orch_res['total_time']
        total_tokens += orch_res['tokens']

        if next_agent_id == "finish":
            print(Fore.MAGENTA + "\n[ORCHESTRATOR] -> KONIEC.")
            final_answer = chat_history.split("---")[-1]
            break

        selected_agent = next((a for a in agents_config if a['id'] == next_agent_id), None)
        print_header(f"TURA {turn_count}: {selected_agent['role']}", ROLE_COLORS.get(next_agent_id, Fore.WHITE))

        # 2. Agent Generuje Odpowiedź
        agent_input = f"ZADANIE: {task_query}\n\nHISTORIA:\n{chat_history}\n\nJesteś {selected_agent['id']}. Jeśli potrzebujesz narzędzia, użyj formatu:\nSEARCH: zapytanie\n```python\nkod\n```"
        agent_res = call_ollama(agent_input, selected_agent['system_prompt'], selected_agent['model'], next_agent_id)

        total_time += agent_res['total_time']
        total_tokens += agent_res['tokens']
        response_text = agent_res['text']

        # --- DETEKCJA I UŻYCIE NARZĘDZI ---
        tool_output = ""

        # A. Detekcja Kodu (dla Codera)
        if (next_agent_id == "coder" or next_agent_id == "solver") and "```python" in response_text:
            try:
                # Wyciągamy zawartość między ```python a ```
                code_match = re.search(r"```python(.*?)```", response_text, re.DOTALL)
                if code_match:
                    code = code_match.group(1)
                    result = execute_python_code(code)
                    tool_output = f"\n[SYSTEM: Wynik uruchomienia kodu Python]:\n{result}\n"
            except Exception as e:
                tool_output = f"\n[SYSTEM ERROR]: {e}\n"

        # B. Detekcja Wyszukiwania (dla Analityka)
        if next_agent_id == "analyst" and "SEARCH:" in response_text:
            try:
                # Szukamy linii zaczynającej się od SEARCH:
                search_match = re.search(r"SEARCH:(.*)", response_text)
                if search_match:
                    query = search_match.group(1).strip()
                    result = search_web(query)
                    tool_output = f"\n[SYSTEM: Wyniki wyszukiwania dla '{query}']:\n{result}\n"
            except Exception as e:
                tool_output = f"\n[SYSTEM ERROR]: {e}\n"

        # Jeśli użyto narzędzia, dodajemy wynik do historii od razu!
        if tool_output:
            print(ROLE_COLORS['tool'] + tool_output + Style.RESET_ALL)
            response_text += tool_output

        chat_history += f"\n--- {selected_agent['role']} ---\n{response_text}\n"
        final_answer = response_text
        last_speaker = next_agent_id

    return {"final_answer": final_answer, "turns": turn_count, "total_time": total_time, "tokens": total_tokens}


def get_judge_score(task_query, expected, answer_solo, answer_multi):
    print_header("KOMISJA SĘDZIOWSKA OCENIA", ROLE_COLORS['judge'])

    # Lista sędziów (Możesz tu wpisać gpt-oss-20 jeśli go masz pobranego)
    JURY_MODELS = ["phi4", "qwen2.5:14b", "gemma2:9b"]

    # Wielkość kontekstu sędziego.
    # 8192 = dużo pamięci, wolniej.
    # 4096 = optymalnie.
    # 2048 = błyskawicznie (wystarczy do krótkich zagadek).
    JUDGE_CTX_SIZE = 4096
    solo_scores = []
    multi_scores = []
    jury_reasons = []

    prompt_template = f"""
    Jesteś profesjonalnym sędzią w konkursie logicznym.

    ZADANIE: {task_query}
    POPRAWNY WZORZEC (PRAWDA): {expected}

    ODPOWIEDŹ A (Solo Model): {answer_solo}
    ODPOWIEDŹ B (Zespół Agentów): {answer_multi}

    --- KRYTERIA OCENY (0-100 PKT) ---
    100 pkt: Odpowiedź idealna, zgodna z WZORCEM, poprawne uzasadnienie.
    75 pkt: Wynik poprawny, ale uzasadnienie mało precyzyjne lub zawiera drobne nieścisłości.
    50 pkt: Wynik częściowo poprawny (np. dobra liczba, złe jednostki) LUB wynik dobry, ale błędna logika (przypadek).
    25 pkt: Wynik błędny, ale widać próbę poprawnego rozumowania (dobry wzór, błąd rachunkowy).
    0 pkt: Wynik błędny, logika błędna, halucynacje lub zaprzeczenie WZORCOWI.

    Twoim zadaniem jest ocenić obie odpowiedzi zgodnie z WZORCEM. NIE OCENIAJ ODPOWIEDZI POPRZEDNICH AGENTÓW, TYLKO GŁÓWNĄ ODPOWIEDŹ
    Zwróć TYLKO JSON: {{ "score_solo": X, "score_multi": Y, "reason": "Krótkie uzasadnienie" }}
    """

    for judge_model in JURY_MODELS:
        print(ROLE_COLORS['judge'] + f"   [JURY: {judge_model}] Ocenia...", end="", flush=True)

        # Wywołanie modelu (krótkie, bez streamingu na ekran dla czytelności)
        try:
            # Używamy call_ollama, ale musimy przechwycić wynik 'po cichu' lub zmodyfikować call_ollama
            # Tutaj użyję prostego requesta, żeby nie spamować logami z 3 modeli naraz
            payload = {
                "model": judge_model,
                "prompt": prompt_template,
                "stream": False,
                "options": {
                    "temperature": 0.0,
                    "num_ctx": JUDGE_CTX_SIZE  # <--- TU JEST ZMIANA (np. 4096 zamiast 8192)
                },
                "format": "json"
            }
            start_t = time.time()
            resp = requests.post(f"{OLLAMA_API}/generate", json=payload).json()
            response_text = resp.get('response', '')

            # Parsowanie
            result = json.loads(response_text)

            s_solo = int(result.get("score_solo", 0))
            s_multi = int(result.get("score_multi", 0))
            reason = result.get("reason", "Brak")

            solo_scores.append(s_solo)
            multi_scores.append(s_multi)
            jury_reasons.append(f"[{judge_model}]: {reason}")

            print(f" -> Solo: {s_solo}, Multi: {s_multi}")

            # Czyścimy VRAM po każdym sędzim
            manage_model(judge_model, action="unload")

        except Exception as e:
            print(Fore.RED + f" -> Błąd sędziego {judge_model}: {e}")
            # W razie błędu dajemy neutralne 0, żeby nie psuć średniej (lub pomijamy)
            solo_scores.append(0)
            multi_scores.append(0)

    # OBLICZANIE ŚREDNIEJ
    avg_solo = round(sum(solo_scores) / len(solo_scores), 1) if solo_scores else 0
    avg_multi = round(sum(multi_scores) / len(multi_scores), 1) if multi_scores else 0

    # Łączenie uzasadnień w jeden string do CSV
    final_reason = " | ".join(jury_reasons)

    print(ROLE_COLORS['judge'] + "-" * 30)
    print(f"   [WERDYKT KOŃCOWY] Solo: {avg_solo} | Multi: {avg_multi}")
    print(ROLE_COLORS['judge'] + "-" * 30 + Style.RESET_ALL)

    return {
        "score_solo": avg_solo,
        "score_multi": avg_multi,
        "reason": final_reason
    }

def visualize_results(csv_filename):
    df = pd.read_csv(csv_filename)
    if df.empty: return

    avg_scores = df[['score_solo', 'score_multi']].mean()
    avg_times = df[['solo_time', 'multi_time']].mean()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Solo vs Group Chat', fontsize=16)

    axes[0].bar(['Solo', 'Group'], avg_scores, color=['gray', 'green'])
    axes[0].set_title('Punkty')

    axes[1].bar(['Solo', 'Group'], avg_times, color=['gray', 'orange'])
    axes[1].set_title('Czas (s)')

    plt.tight_layout()
    plt.show()

    print("\n" + Fore.CYAN + df[['task_id', 'turns', 'score_multi']].to_string(index=False))
    pass


# --- MAIN ---
def run_research():
    tasks = json.load(open('tasks.json', 'r', encoding='utf-8'))
    agents = json.load(open('agents.json', 'r', encoding='utf-8'))

    fieldnames = ["task_id", "score_solo", "score_multi", "solo_time", "multi_time", "solo_tokens", "multi_tokens",
                  "turns", "judge_reason"]

    # Tworzenie pliku
    with open(CSV_FILE, 'w', newline='', encoding='utf-8') as f:
        csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    for task in tasks:
        print_header(f"ZADANIE: {task['id']}", Fore.WHITE)

        # 1. SOLO
        print(Fore.WHITE + ">>> Testowanie modelu SOLO...")
        solo_res = call_ollama(task['query'], "Pomocny asystent", MODEL_SOLO, "system")

        # 2. GROUP
        group_res = run_group_chat_loop(task['query'], agents)

        # 3. OCENA
        scores = get_judge_score(task['query'], task['expected_answer'], solo_res['text'], group_res['final_answer'])

        row = {
            "task_id": task['id'],
            "score_solo": scores.get('score_solo', 0),
            "score_multi": scores.get('score_multi', 0),
            "solo_time": round(solo_res['total_time'], 2),
            "multi_time": round(group_res['total_time'], 2),
            "solo_tokens": solo_res['tokens'],
            "multi_tokens": group_res['tokens'],
            "turns": group_res['turns'],
            "judge_reason": scores.get('reason', '')
        }

        with open(CSV_FILE, 'a', newline='', encoding='utf-8') as f:
            csv.DictWriter(f, fieldnames=fieldnames).writerow(row)

    visualize_results(CSV_FILE)


if __name__ == "__main__":
    run_research()