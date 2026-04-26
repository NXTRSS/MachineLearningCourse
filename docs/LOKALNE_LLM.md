# Lokalne modele językowe (LLM) — setup w domu

Na zajęciach z Function Calling i RAG używamy modeli językowych uruchomionych lokalnie na laptopie. Poniżej znajdziesz wszystko co potrzebne, żeby przygotować się w domu.

---

## Czy mój komputer da radę?

| RAM | Co możesz uruchomić |
|-----|---------------------|
| < 8 GB | Za mało — skorzystaj z serwera prowadzącego na zajęciach |
| 8 GB | Małe modele: `qwen3:4b` ⭐, `gemma3:4b`, `phi4-mini` |
| 16 GB | Komfortowe modele: `qwen3:8b` ⭐, `llama3.1:8b`, `gemma3:12b` |
| 32 GB+ | Duże modele: `qwen3:14b`, `gemma3:27b`, `llama3.1:70b` (skwantyzowany) |

> **Mac z Apple Silicon (M1/M2/M3/M4)** — GPU jest zintegrowany z CPU, więc cały RAM jest dostępny dla modelu. Laptop z 16 GB M2 pobije wiele desktopów z kartą 8 GB VRAM.

---

## Jak zacząć — notebook setup_local_llm

Otwórz **`setup_local_llm.ipynb`** — sprawdzi Twój sprzęt i przeprowadzi przez instalację krok po kroku. Wystarczy uruchomić komórki po kolei.

---

## Platformy — LM Studio vs Ollama

### LM Studio ⭐ *polecana dla początkujących*

**Czym jest:** Aplikacja desktopowa z interfejsem graficznym. Pobierasz, instalujesz, klikasz — gotowe.

**Zalety:**
- Graficzny interfejs — brak komendy w terminalu
- Wbudowana wyszukiwarka modeli (Hugging Face)
- Chat wbudowany w aplikację — możesz od razu porozmawiać z modelem
- Serwer API jednym kliknięciem (kompatybilny z OpenAI API)
- Pokazuje zużycie RAM i prędkość generacji (tokens/s)
- Działa na Windows, macOS, Linux

**Wady:**
- Ciężka aplikacja (~500 MB)
- Mniej wygodna do automatyzacji / skryptów
- Trzeba ją ręcznie uruchomić przed zajęciami

**Instalacja:** https://lmstudio.ai → Download → uruchom instalator

**Port domyślny:** `1234` → adres API: `http://localhost:1234/v1`

---

### Ollama

**Czym jest:** Lekkie narzędzie działające w tle jako usługa systemowa. Obsługuje się przez terminal.

**Zalety:**
- Bardzo lekkie (~50 MB)
- Startuje automatycznie przy uruchomieniu systemu
- Prosta składnia: `ollama run llama3.2` — pobiera i uruchamia model
- Wygodna do automatyzacji i skryptów
- Duże oficjalne repozytorium modeli (ollama.com/library)
- Open source

**Wady:**
- Brak interfejsu graficznego (tylko terminal)
- Mniej przejrzysta informacja o obciążeniu sprzętu
- Modele są w formacie GGUF — mniej wyboru niż na Hugging Face

**Instalacja:** https://ollama.com → Download

**Port domyślny:** `11434` → adres API: `http://localhost:11434`

**Przydatne komendy:**
```bash
ollama list                  # pokaż pobrane modele
ollama run llama3.2          # pobierz (jeśli brak) i uruchom
ollama run gemma3:12b        # konkretna wersja modelu
ollama ps                    # pokaż aktualnie załadowane modele
ollama rm llama3.2           # usuń model (odzyskaj miejsce)
```

---

### Inne opcje (mniej popularne)

**Jan** (jan.ai) — podobny do LM Studio, GUI, open source. Mniej dojrzały, ale aktywnie rozwijany.

**GPT4All** — prosty GUI, dobry dla zupełnych początkujących. Mały wybór modeli, wolniejszy.

**llama.cpp** — surowe narzędzie CLI, maksymalna kontrola i wydajność. Dla zaawansowanych.

---

## Modele — co pobrać?

### Mac z Apple Silicon — szukaj formatu MLX

Na Macu z chipem M1/M2/M3/M4 używaj modeli w formacie **MLX** zamiast GGUF. Są zoptymalizowane pod Apple Silicon i działają znacznie szybciej (nawet 2–3x).

**Gdzie szukać:** Hugging Face → wyszukaj np. `mlx-community/llama-3.2-3b-instruct-4bit`

W LM Studio: przy wyszukiwaniu modelu wybierz zakładkę **MLX** lub filtruj po `mlx-community`.

> Modele GGUF też działają na Macu, ale wolniej. Jeśli widzisz wersję MLX — wybierz ją.

### Rekomendowane modele do zajęć

| Model | Rozmiar | RAM | Jakość | Uwagi |
|-------|---------|-----|--------|-------|
| `qwen3:4b` | ~3 GB | 8 GB | ★★★★★ | ⭐ Najlepsza jakość w klasie 4B, polecany |
| `gemma3:4b` | ~3 GB | 8 GB | ★★★★ | Dobra alternatywa dla Qwen3:4b |
| `phi4-mini` | ~2.5 GB | 8 GB | ★★★★ | Lżejszy, szybszy, nieco słabszy |
| `llama3.2:3b` | ~2 GB | 8 GB | ★★★☆ | Minimum, gdy brakuje miejsca |
| `qwen3:8b` | ~5 GB | 16 GB | ★★★★★ | ⭐ Polecany dla 16 GB RAM |
| `llama3.1:8b` | ~5 GB | 16 GB | ★★★★★ | Równie dobry, popularna alternatywa |
| `gemma3:12b` | ~8 GB | 16 GB | ★★★★★ | Świetny jeśli masz zapas RAM |
| `qwen3:14b` | ~9 GB | 32 GB | ★★★★★ | Bardzo mocny model dla entuzjastów |

> Na zajęciach używamy głównie modeli 8B — jeśli masz 16 GB RAM, pobierz `qwen3:8b` lub `llama3.1:8b`.

#### O rodzinie Qwen3

Qwen3 (Alibaba, 2025) to aktualnie jedna z najsilniejszych rodzin otwartych modeli. Kilka ciekawych cech:
- **Tryb myślenia** — model może "myśleć na głos" (jak chain-of-thought) zanim odpowie; można to wyłączyć dodając `/no_think` w prompcie, gdy zależy Ci na szybkości
- **128K kontekst** — znacznie więcej niż starsze modele (zwykle 4–8K)
- **119 języków** — w tym polski
- Dostępny na Ollama (`qwen3:4b`, `qwen3:8b` itd.) i w MLX na Apple Silicon

---

## Najczęstsze problemy

### Model działa bardzo wolno
Sprawdź czy aplikacja używa GPU:
- **LM Studio:** na dole okna widać `GPU Layers` — powinno być > 0
- **Ollama:** `ollama ps` → kolumna `PROCESSOR` powinna pokazywać `100% GPU`

Jeśli używa tylko CPU — model jest za duży na Twój RAM. Pobierz mniejszy lub bardziej skwantyzowany (np. `Q4_K_M` zamiast `Q8_0`).

### "Nie mam miejsca na dysku"
Modele zajmują 2–10 GB. Domyślne lokalizacje:
- **LM Studio:** `~/lmstudio-models` (można zmienić w ustawieniach)
- **Ollama (macOS):** `~/.ollama/models`
- **Ollama (Windows):** `C:\Users\<user>\.ollama\models`

### Port jest zajęty
Jeśli masz uruchomione i LM Studio i Ollamę równocześnie, ich porty (1234 i 11434) nie kolidują — możesz mieć oba aktywne.

### Windows: model się nie uruchamia
Upewnij się że masz zainstalowane [Visual C++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe).

---

## Podsumowanie — co wybrać?

- **Nie lubisz terminala** → LM Studio
- **Mac z Apple Silicon** → LM Studio (MLX) lub Ollama
- **Chcesz automatyzować / skryptować** → Ollama
- **Mało RAM (8 GB)** → `qwen3:4b` w LM Studio lub Ollama
- **Dużo RAM (16 GB+)** → `qwen3:8b` lub `llama3.1:8b` w Ollamie lub LM Studio

Na zajęciach serwer prowadzącego jest zawsze dostępny jako backup — nie musisz mieć lokalnego modelu, żeby uczestniczyć w zajęciach.
