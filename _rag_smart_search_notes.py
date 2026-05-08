"""
NOTATKI DO RAG NOTEBOOK — kod przeniesiony z Function_Calling (sekcja 5b)
==========================================================================

Ten kod pokazuje porównanie PROSTY substring search vs SMART search z LLM + Pydantic.
Docelowo trafi do RAG notebooka jako sekcja porównawcza (substring vs embeddingi vs context stuffing).

Kontekst: w FC notebook zostaje TYLKO prosty substring search_presidents().
Smart search i porównanie przenosimy tutaj.
"""

# === MODELE PYDANTIC — strukturalna odpowiedź smart searcha ===

# Pydantic wymusza strukturę odpowiedzi LLM-a na TRZECH poziomach:
#
# 1. PresidentMatch  — pojedynczy wynik (prezydent + powód + pewność TEGO dopasowania)
# 2. PresidentAnswer — cała odpowiedź (lista wyników + ogólna pewność CAŁEGO wyszukania)
# 3. Field(ge=0, le=1) — walidacja: confidence MUSI być 0–1, inaczej Pydantic odrzuci i instructor wymusi retry
#
# Dzięki temu LLM nie może zwrócić byle czego — struktura, typy i zakresy są wymuszone.

# class PresidentMatch(BaseModel):
#     name: str = Field(..., description="Imię i nazwisko prezydenta")
#     reason: str = Field(..., description="Dlaczego pasuje do zapytania (1 zdanie)")
#     confidence: float = Field(..., ge=0, le=1, description="Pewność TEGO dopasowania (0=luźne skojarzenie, 1=pewne trafienie)")
#
# class PresidentAnswer(BaseModel):
#     matches: List[PresidentMatch] = Field(..., description="Prezydenci pasujący do zapytania (pusta lista jeśli nikt nie pasuje)")
#     confidence: float = Field(..., ge=0, le=1, description="Pewność CAŁEJ odpowiedzi (0=zgaduję, 1=mam dane)")


# === SMART SEARCH FUNCTION (context stuffing) ===

# _PRESIDENTS_RAW = Path("prezydenci_polski.md").read_text(encoding="utf-8") if Path("prezydenci_polski.md").exists() else ""
#
# def search_presidents_smart(query: str) -> str:
#     """
#     Przeszukuje bazę danych o prezydentach Polski (III RP) — z rozumieniem semantycznym.
#     Rozumie synonimy, kontekst i pytania zadane własnymi słowami.
#     """
#     if not _PRESIDENTS_RAW:
#         return "Brak danych — nie znaleziono pliku prezydenci_polski.md"
#     if not instructor_client:
#         return search_presidents(query)
#     try:
#         result = instructor_client.chat.completions.create(
#             model=MODEL_NAME,
#             response_model=PresidentAnswer,
#             messages=[
#                 {"role": "system", "content":
#                  "Odpowiadasz WYŁĄCZNIE na podstawie podanych danych o prezydentach. "
#                  "Jeśli pasuje więcej niż jeden prezydent — wymień WSZYSTKICH. "
#                  "Jeśli danych brak — zwróć pustą listę. Nie wymyślaj. Odpowiadaj po polsku."},
#                 {"role": "user", "content":
#                  f"Pytanie: {query}\n\nDane:\n{_PRESIDENTS_RAW}"}
#             ],
#         )
#         if not result.matches:
#             return "Nie znaleziono pasujących prezydentów."
#         return "\n".join(f"- {m.name}: {m.reason}" for m in result.matches)
#     except Exception:
#         return search_presidents(query)


# === PORÓWNANIE: PROSTY vs SMART ===

# test_queries = [
#     "Nobel",                     # odmiana: "Nobel" ≠ "Nobla" w tekście
#     "Międzynarodowa nagroda",    # peryfraza — trzeba skojarzyć z Noblem
#     "kto zbierał monety",        # semantyczne — nie ma takiej frazy w tekście
#     "najdłużej rządził",         # synonim "najdłuższa kadencja"
#     "związkowiec",               # w tekście jest "NSZZ Solidarność", nie "związkowiec"
#     "wojskowy",                  # w tekście "Wyższa Szkoła Piechoty", nie "wojskowy"
#     "zginął tragicznie",         # w tekście "katastrofa smoleńska", nie "zginął"
#     "Kwaśniewski",               # to oba znajdą
# ]
#
# print("=== PROSTY (substring) vs. SMART (LLM) ===\n")
# for q in test_queries:
#     simple = search_presidents(q)
#     simple_ok = "Nie znaleziono" not in simple
#
#     print(f"  Zapytanie: \"{q}\"")
#     print(f"    Prosty:  {'ZNALAZŁ' if simple_ok else 'NIE ZNALAZŁ'}")
#     if simple_ok:
#         print(f"             {simple[:150]}...")
#
#     if instructor_client:
#         try:
#             result = instructor_client.chat.completions.create(
#                 model=MODEL_NAME,
#                 response_model=PresidentAnswer,
#                 messages=[
#                     {"role": "system", "content":
#                      "Odpowiadasz WYŁĄCZNIE na podstawie podanych danych o prezydentach. "
#                      "Jeśli pasuje więcej niż jeden prezydent — wymień WSZYSTKICH. "
#                      "Jeśli danych brak — zwróć pustą listę. Nie wymyślaj. Odpowiadaj po polsku."},
#                     {"role": "user", "content":
#                      f"Pytanie: {q}\n\nDane:\n{_PRESIDENTS_RAW}"}
#                 ],
#             )
#             smart_ok = len(result.matches) > 0
#             print(f"    Smart:   {'ZNALAZŁ' if smart_ok else 'NIE ZNALAZŁ'}  (pewność odpowiedzi: {result.confidence:.0%})")
#             for m in result.matches:
#                 print(f"             → {m.name} ({m.confidence:.0%}): {m.reason}")
#         except Exception as e:
#             print(f"    Smart:   BŁĄD ({e})")
#     print()


# === MARKDOWN OPIS (do RAG notebooka) ===

# ### Context stuffing vs substring vs embeddingi
#
# | Metoda | Jak działa | Kiedy | Ograniczenia |
# |---|---|---|---|
# | **Substring** | `if query in text` | Małe dane, dokładne frazy | Nie ogarnia synonimów, odmiany |
# | **Context stuffing** | Cały tekst → LLM | Dane < ~4K tokenów | Nie skaluje się |
# | **RAG (embeddingi)** | Similarity search → top-K → LLM | Duże dane | Wymaga modelu embeddingowego |
