"""
Chat UI — interaktywny chat z narzędziami w stylu ChatGPT.

Uruchamiany z notebooka jedną linijką:
    from chat_ui import launch_chat
    launch_chat(client, MODEL_NAME, tools_definition, AVAILABLE_TOOLS, DEFAULT_SYSTEM_PROMPT)

Otwiera w przeglądarce interfejs z:
  - dymkami wiadomości (user / assistant)
  - collapsible reasoning (🧠 tok myślenia)
  - wyświetlaniem tool calls z wynikami
  - pamięcią konwersacji (multi-turn)
  - agent loop (multi-step tool calling)

Wymaga: pip install gradio (>=6.0)
"""

import json


def launch_chat(
    client,
    model_name,
    tools_definition,
    available_tools,
    system_prompt,
    instructor_client=None,
    max_steps=6,
    share=False,
):
    """
    Uruchamia Gradio chat UI z Function Calling.

    Args:
        client: OpenAI client
        model_name: Nazwa modelu (np. "gemma-4-e4b-it-mlx")
        tools_definition: Lista narzędzi w formacie OpenAI JSON Schema
        available_tools: Dict {nazwa: funkcja} — mapowanie tool calls na Python
        system_prompt: System prompt (np. DEFAULT_SYSTEM_PROMPT)
        instructor_client: Opcjonalny instructor client (do wymuszonego reasoning)
        max_steps: Maks. liczba kroków agenta na jedną wiadomość (domyślnie 6)
        share: True = publiczny link (np. do pokazania na zajęciach przez ngrok)
    """
    from utils import extract_reasoning, clean_content

    try:
        import gradio as gr
    except ImportError:
        from utils import ensure_package
        ensure_package("gradio")
        import gradio as gr

    # ── Nazwy narzędzi do wyświetlania ──
    tool_names = [t["function"]["name"] for t in tools_definition]

    def respond(message, history):
        """
        Generator — Gradio wywołuje go na każde pytanie użytkownika.
        yield-ujemy historię po każdym kroku, żeby UI aktualizowało się na żywo.
        """
        if not message.strip():
            yield history
            return

        # ── Budujemy messages z historii Gradio ──
        # Format Gradio 6: [{"role": "user/assistant", "content": "..."}]
        # Filtrujemy "thinking" bloki (metadata) — LLM ich nie potrzebuje
        messages = [{"role": "system", "content": system_prompt}]
        for msg in history:
            if msg.get("role") in ("user", "assistant") and not msg.get("metadata"):
                messages.append({"role": msg["role"], "content": msg["content"]})

        messages.append({"role": "user", "content": message})

        # Pokaż pytanie użytkownika
        history = history + [{"role": "user", "content": message}]
        yield history

        # ── Agent loop (multi-step tool calling z pamięcią) ──
        for step in range(max_steps):
            # Wywołaj API w osobnym wątku, żeby animacja "myślę" działała
            from concurrent.futures import ThreadPoolExecutor, as_completed
            import time as _time

            future_pool = ThreadPoolExecutor(max_workers=1)
            future = future_pool.submit(
                client.chat.completions.create,
                model=model_name,
                messages=messages,
                tools=tools_definition,
                temperature=0.1,
            )

            # Animacja oczekiwania — aktualizujemy co 0.8s
            # Po 8s bez odpowiedzi → prawdopodobnie stoi w kolejce
            QUEUE_THRESHOLD = 8
            dots = ["⏳", "⏳.", "⏳..", "⏳..."]
            tick = 0
            _t0 = _time.monotonic()
            base_msg = "Myślę" if step == 0 else "Analizuję wyniki narzędzia"
            while not future.done():
                elapsed = _time.monotonic() - _t0
                if elapsed > QUEUE_THRESHOLD:
                    indicator = dots[tick % len(dots)]
                    queue_pos = f" ({int(elapsed)}s)"
                    status = f"🕐 *Czekam w kolejce...{queue_pos}*"
                else:
                    indicator = dots[tick % len(dots)]
                    status = f"{indicator} *{base_msg}...*"
                # Dodaj/zamień wskaźnik
                if history and history[-1].get("content", "").startswith(("⏳", "🕐")):
                    history[-1] = {"role": "assistant", "content": status}
                else:
                    history.append({"role": "assistant", "content": status})
                yield history
                _time.sleep(0.8)
                tick += 1

            response = future.result()
            future_pool.shutdown(wait=False)

            # Usuń wskaźnik "myślę" / "czekam w kolejce"
            if history and history[-1].get("content", "").startswith(("⏳", "🕐")):
                history.pop()
                yield history

            msg = response.choices[0].message

            # ── Natywny reasoning (Qwen3, DeepSeek-R1, Gemma-4 channel) ──
            reasoning = extract_reasoning(msg)
            msg_content = clean_content(msg)
            if reasoning:
                history.append(
                    {
                        "role": "assistant",
                        "content": str(reasoning)[:800],
                        "metadata": {"title": "🧠 Tok myślenia"},
                    }
                )
                yield history

            # ── LLM mówi coś przed tool callem ──
            if msg_content and msg.tool_calls:
                history.append(
                    {
                        "role": "assistant",
                        "content": msg_content[:500],
                        "metadata": {"title": "💬 Komentarz modelu"},
                    }
                )
                yield history

            # ── Brak tool calls → finalna odpowiedź ──
            if not msg.tool_calls:
                content = msg_content or "(brak odpowiedzi)"
                history.append({"role": "assistant", "content": content})
                yield history
                return

            # ── Wykonanie tool calls ──
            messages.append(msg)

            for tc in msg.tool_calls:
                func_name = tc.function.name
                func_args = json.loads(tc.function.arguments)

                # Wyświetl tool call
                args_str = json.dumps(func_args, ensure_ascii=False)
                tool_content = f"**Wywołanie:** `{func_name}({args_str})`\n\n"

                # Wykonaj narzędzie
                try:
                    if func_name in available_tools:
                        result = available_tools[func_name](**func_args)
                    else:
                        result = f"❌ Nieznane narzędzie: {func_name}"
                except Exception as e:
                    result = f"❌ Błąd: {e}"

                # Pokaż wynik (skrócony w UI, pełny do LLM-a)
                display_result = result[:300] + "..." if len(result) > 300 else result
                tool_content += f"**Wynik:** {display_result}"

                history.append(
                    {
                        "role": "assistant",
                        "content": tool_content,
                        "metadata": {"title": f"🔧 Narzędzie: {func_name}"},
                    }
                )
                yield history

                messages.append(
                    {"role": "tool", "tool_call_id": tc.id, "content": result}
                )

        # ── Limit kroków ──
        history.append(
            {
                "role": "assistant",
                "content": f"⚠️ Osiągnięto limit {max_steps} kroków agenta.",
            }
        )
        yield history

    # ── Przykładowe pytania ──
    examples = [
        "Jaka jest pogoda we Wrocławiu?",
        "Opowiedz mi jakiś mało znany fakt o Andrzeju Dudzie.",
        "Jaki jest aktualny kurs dolara do złotówki?",
        "Znajdź informacje o Nikoli Tesli na Wikipedii.",
    ]

    # ── Budujemy UI (Gradio 6) ──
    with gr.Blocks(
        title="🤖 Chat z narzędziami — Function Calling",
    ) as demo:

        gr.Markdown(
            "# 🤖 Chat z narzędziami\n"
            f"*Model: `{model_name}` · "
            f"Narzędzia: {', '.join(tool_names)}*"
        )

        chatbot = gr.Chatbot(
            height=550,
            buttons=["copy"],
            placeholder="Zadaj pytanie — model użyje narzędzi jeśli potrzeba! 🚀",
        )

        with gr.Row():
            textbox = gr.Textbox(
                placeholder="Napisz pytanie... (np. 'Ile to 17 * 23?')",
                show_label=False,
                scale=9,
                autofocus=True,
            )
            submit_btn = gr.Button("Wyślij ➤", variant="primary", scale=1)

        with gr.Row():
            clear_btn = gr.ClearButton([chatbot, textbox], value="🗑️ Wyczyść chat")

        gr.Examples(examples=examples, inputs=textbox, label="Przykładowe pytania")

        # ── Logika przycisków ──
        textbox.submit(respond, [textbox, chatbot], [chatbot]).then(
            lambda: "", None, textbox
        )
        submit_btn.click(respond, [textbox, chatbot], [chatbot]).then(
            lambda: "", None, textbox
        )

    print("\n🚀 Uruchamiam chat UI...")
    print("   Otwieram w nowej karcie przeglądarki...\n")
    demo.launch(
        share=share,
        quiet=False,          # pokaż URL w output
        inbrowser=True,       # automatycznie otwórz w nowej karcie
        height=300,           # mały inline preview w notebooku
        theme=gr.themes.Soft(),
        css="""
            .gradio-container { max-width: 900px !important; margin: auto; }
            footer { display: none !important; }
        """,
    )
