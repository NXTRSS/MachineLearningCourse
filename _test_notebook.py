#!/usr/bin/env python3
"""
Test end-to-end: symuluje studenta wypełniającego ćwiczenia.

1. Kopiuje Function_Calling.ipynb → Function_Calling_TEST.ipynb
2. Ustawia port 4242 + odkomentowuje gemma-4-e4b
3. Wkleja fragmenty rozwiązań do stubów (jak student)
4. Wykonuje notebook komórka po komórce
5. Zapisuje wynik z outputami
"""

import json
import copy
import re

# ── 1. Wczytaj i skopiuj ──
with open("Function_Calling.ipynb", encoding="utf-8") as f:
    nb = json.load(f)

nb_original = copy.deepcopy(nb)  # zachowaj oryginał do porównania po przed_zajeciami

# ── 2. Modyfikuj Cell 5 — port 4242 + odkomentuj model ──
cell5 = nb["cells"][5]
src_lines = cell5["source"]
new_lines = []
for line in src_lines:
    # Zamień connect_llm na port 4242
    if "LECTURER_SERVER" in line and "ADRES_SERWERA" in line:
        # Nie zmieniamy — connect_llm i tak próbuje localhost
        new_lines.append(line)
    elif line.strip().startswith("# model="):
        # Odkomentuj model=
        new_lines.append(line.replace("# model=", "model="))
    elif line.strip().startswith("# backend="):
        # Zostaw zakomentowane
        new_lines.append(line)
    else:
        new_lines.append(line)
cell5["source"] = new_lines

# Lepiej: zamień cały blok connect_llm na bezpośrednie połączenie z port 4242
cell5["source"] = [
    "from utils import connect_llm, extract_reasoning, print_reasoning, clean_content\n",
    "\n",
    "# --- TEST: bezpośrednie połączenie na port 4242 z modelem gemma-4-e4b ---\n",
    "from openai import OpenAI\n",
    "import instructor\n",
    "\n",
    'client = OpenAI(base_url="http://localhost:4242/v1", api_key="lm-studio")\n',
    "instructor_client = instructor.from_openai(client, mode=instructor.Mode.MD_JSON)\n",
    'MODEL_NAME = [m.id for m in client.models.list().data if "gemma" in m.id.lower()][0]\n',
    "\n",
    'print(f"\\nKlient LLM gotowy!  Model: {MODEL_NAME}")\n',
    'print(f"Instructor:         {\'tak\' if instructor_client else \'nie\'}")\n',
]

# ── 3. Wypełnij stuby rozwiązaniami (jak student) ──

# Exercise 1A (Cell 40): zmień description
# Student wpisuje: test_tools[0]["function"]["description"] = "Przeszukuje bazę danych o prezydentach Polski"
cell40_src = ''.join(nb["cells"][40]["source"])
cell40_src = cell40_src.replace(
    'test_tools[0]["function"]["description"] = ...  # np. "Przeszukuje bazę prezydentów Polski"',
    'test_tools[0]["function"]["description"] = "Przeszukuje bazę danych o prezydentach Polski"'
)
nb["cells"][40]["source"] = [l + "\n" for l in cell40_src.split("\n")[:-1]] + [cell40_src.split("\n")[-1]]

# Exercise 2 (Cell 55): wklej funkcję + opis
# Student zastępuje pass swoim kodem i wpisuje opis
cell55_src = ''.join(nb["cells"][55]["source"])

# Zastąp def get_population
cell55_src = cell55_src.replace(
    '''def get_population(city: str) -> str:
    # Słownik z danymi: {"Warszawa": (1_860_000, 1), "Kraków": (800_000, 2), ...}
    # Jeśli miasto jest w słowniku → zwróć f-string, np.:
    #   "Kraków: ~800,000 mieszkańców (#2 w Polsce)"
    # Jeśli nie ma → zwróć: f"Brak danych o populacji dla: {city}"

    pass  # ← zastąp swoim kodem''',
    '''def get_population(city: str) -> str:
    dane = {
        "Warszawa": (1_860_000, 1), "Kraków": (800_000, 2), "Wrocław": (674_000, 3),
        "Gdańsk": (470_000, 6), "Poznań": (535_000, 5), "Łódź": (646_000, 4),
    }
    if city in dane:
        pop, rank = dane[city]
        return f"{city}: ~{pop:,} mieszkańców (#{rank} w Polsce)"
    return f"Brak danych o populacji dla: {city}"'''
)

# Zastąp opis
cell55_src = cell55_src.replace(
    '_population_desc = ...  # Tutaj wpisz swój kod — opis narzędzia (1 zdanie, po polsku)',
    '_population_desc = "Zwraca przybliżoną liczbę mieszkańców polskiego miasta."'
)

nb["cells"][55]["source"] = [l + "\n" for l in cell55_src.split("\n")[:-1]] + [cell55_src.split("\n")[-1]]

# Exercise 3 (Cell 63): wklej return + opis argumentu + opis narzędzia
cell63_src = ''.join(nb["cells"][63]["source"])

# Zastąp ... w return
cell63_src = cell63_src.replace(
    """        # ✏️ UZUPEŁNIJ: Zwróć f-stringa z informacjami ze strony.
        # Użyj page.title, page.summary (obetnij do 500 znaków: page.summary[:500])
        # i page.url. Połącz je w jednego f-stringa oddzielonego enterami (\\n\\n).
        # Wzór:  f"{tytuł}\\n\\n{streszczenie}\\n\\nŹródło: {url}"

        ...""",
    '        return f"{page.title}\\n\\n{page.summary[:500]}\\n\\nŹródło: {page.url}"'
)

# Zastąp description=... w args
cell63_src = cell63_src.replace(
    'query: str = Field(..., description=...  # ✏️ Tutaj wpisz opis argumentu — co to jest query?',
    'query: str = Field(..., description="Zapytanie do Wikipedii, np. \'fotosynteza\', \'Nikola Tesla\'"'
)

# Zastąp opis narzędzia
cell63_src = cell63_src.replace(
    '_wikipedia_desc = ...  # ✏️ Tutaj wpisz opis narzędzia (string) — kiedy LLM powinien go użyć?',
    '_wikipedia_desc = "Przeszukuje Wikipedię i zwraca streszczenie artykułu. Użyj gdy pytanie dotyczy wiedzy ogólnej, historii, nauki, geografii."'
)

nb["cells"][63]["source"] = [l + "\n" for l in cell63_src.split("\n")[:-1]] + [cell63_src.split("\n")[-1]]

# Exercise 4 (Cell 79): wklej pola FactCheck
nb["cells"][79]["source"] = [
    "# Ćwiczenie 4: Uzupełnij model FactCheck\n",
    "\n",
    "class FactCheck(BaseModel):\n",
    '    claim: str = Field(..., description="Sprawdzane twierdzenie")\n',
    '    evidence: str = Field(..., description="Znalezione dowody potwierdzające lub obalające")\n',
    '    verdict: Literal["prawda", "fałsz", "nie da się zweryfikować"] = Field(\n',
    '        ..., description="Werdykt na podstawie dowodów"\n',
    "    )\n",
    '    confidence: float = Field(..., ge=0, le=1, description="Pewność werdyktu (0=zgaduję, 1=pewny)")\n',
    '    source: str = Field(..., description="Skąd pochodzą dowody, np. \'baza prezydentów\', \'Wikipedia\'")',
]

# Exercise 5 (Cell 92): wpisz pytanie
cell92_src = ''.join(nb["cells"][92]["source"])
cell92_src = cell92_src.replace(
    'MOJE_PYTANIE = ...  # Tutaj wpisz swój kod — wymyśl pytanie łączące kilka narzędzi',
    'MOJE_PYTANIE = "Ile lat miał Aleksander Kwaśniewski gdy skończył kadencję? Oblicz ile to w przybliżeniu dni."'
)
nb["cells"][92]["source"] = [l + "\n" for l in cell92_src.split("\n")[:-1]] + [cell92_src.split("\n")[-1]]

# Exercise 6 (Cell 97): wpisz temat + system prompt
cell97_src = ''.join(nb["cells"][97]["source"])
cell97_src = cell97_src.replace(
    'TEMAT = "..."  # np. "asystent podróżniczy"',
    'TEMAT = "asystent podróżniczy"'
)
cell97_src = cell97_src.replace(
    'SYSTEM_PROMPT = "..."  # Tutaj wpisz swój kod — opisz rolę asystenta',
    'SYSTEM_PROMPT = "Jesteś asystentem podróżniczym. Pomagasz planować wycieczki, sprawdzasz pogodę i populację miast. Odpowiadaj po polsku."'
)
nb["cells"][97]["source"] = [l + "\n" for l in cell97_src.split("\n")[:-1]] + [cell97_src.split("\n")[-1]]

# ── 4. Zapisz jako test ──
output_path = "Function_Calling_TEST.ipynb"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
    f.write("\n")

print(f"✅ Zapisano {output_path}")
print("   Stuby wypełnione rozwiązaniami (symulacja studenta)")
print("   Port: 4242, Model: gemma-4-e4b")
print()
print("Teraz uruchom: jupyter nbconvert --to notebook --execute Function_Calling_TEST.ipynb")
