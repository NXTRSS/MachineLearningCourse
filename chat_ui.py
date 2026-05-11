"""
Chat UI — interaktywny chat z narzędziami w stylu ChatGPT.

Uruchamiany z notebooka jedną linijką:
    from chat_ui import launch_chat
    launch_chat(client, MODEL_NAME, tools_definition, AVAILABLE_TOOLS, DEFAULT_SYSTEM_PROMPT)

Otwiera w przeglądarce interfejs z:
  - dymkami wiadomości (user / assistant)
  - collapsible reasoning (🧠 tok myślenia)
  - wyświetlaniem tool calls z wynikami
  - przełącznik "pod maską" — pokaż/ukryj szczegóły (reasoning, tool calls)
  - ekran logowania (użytkownik + opcjonalne hasło)
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
    reasoning="all",
    auth_password=None,
    auth_fn=None,
    ask_username=False,
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
        auth_password: Hasło stałe — Gradio pokaże ekran logowania z hasłem.
        auth_fn: Funkcja (username, password) → nowy client lub False.
        ask_username: True = pokaż ekran z polem "Użytkownik" (bez hasła).
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

    # ── Mutowalne kontenery (auth_fn może podmienić klienta) ──
    _client = [client]
    _username = [""]  # nazwa studenta → nagłówek X-Student-Name
    _show_debug = [True]  # toggle "Pod maską" — mutowalny, żeby respond czytał na bieżąco

    # ── Login config ──
    needs_login = ask_username or auth_fn is not None or auth_password is not None
    needs_password = auth_fn is not None or auth_password is not None

    # ── Helper: filtruj historię wg trybu debug ──
    def _display(full_hist, show):
        """Zwróć historię do wyświetlenia — pełną lub bez metadanych."""
        if show:
            return list(full_hist)
        return [m for m in full_hist if not m.get("metadata")]

    def respond(message, _chatbot, full_hist):
        """
        Generator — Gradio wywołuje go na każde pytanie użytkownika.
        yield-ujemy (display_history, full_history) po każdym kroku.
        """
        if not message.strip():
            yield _display(full_hist, _show_debug[0]), full_hist
            return

        # ── Budujemy messages z pełnej historii ──
        messages = [{"role": "system", "content": system_prompt}]
        for msg in full_hist:
            if msg.get("role") in ("user", "assistant") and not msg.get("metadata"):
                messages.append({"role": msg["role"], "content": msg["content"]})

        messages.append({"role": "user", "content": message})

        # Pokaż pytanie użytkownika
        full_hist = full_hist + [{"role": "user", "content": message}]
        yield _display(full_hist, _show_debug[0]), full_hist

        # ── Agent loop (multi-step tool calling z pamięcią) ──
        tool_log = []

        for step in range(max_steps):
            from concurrent.futures import ThreadPoolExecutor
            import time as _time

            future_pool = ThreadPoolExecutor(max_workers=1)
            _extra = {"X-Student-Name": _username[0]} if _username[0] else {}
            future = future_pool.submit(
                _client[0].chat.completions.create,
                model=model_name,
                messages=messages,
                tools=tools_definition,
                temperature=0.1,
                extra_headers=_extra,
            )

            # Animacja oczekiwania
            QUEUE_THRESHOLD = 8
            dots = ["⏳", "⏳.", "⏳..", "⏳..."]
            tick = 0
            _t0 = _time.monotonic()
            base_msg = "Myślę" if step == 0 else "Analizuję wyniki narzędzia"
            anim_active = False
            while not future.done():
                elapsed = _time.monotonic() - _t0
                if elapsed > QUEUE_THRESHOLD:
                    indicator = dots[tick % len(dots)]
                    queue_pos = f" ({int(elapsed)}s)"
                    status = f"🕐 *Czekam w kolejce...{queue_pos}*"
                else:
                    indicator = dots[tick % len(dots)]
                    status = f"{indicator} *{base_msg}...*"
                anim_msg = {"role": "assistant", "content": status}
                display = _display(full_hist, _show_debug[0])
                if anim_active:
                    display[-1] = anim_msg
                else:
                    display.append(anim_msg)
                    anim_active = True
                yield display, full_hist
                _time.sleep(0.8)
                tick += 1

            response = future.result()
            future_pool.shutdown(wait=False)

            yield _display(full_hist, _show_debug[0]), full_hist

            msg = response.choices[0].message

            # ── Natywny reasoning ──
            show_this = (reasoning == "all") or (reasoning == "first" and step == 0)
            r = extract_reasoning(msg)
            msg_content = clean_content(msg)
            if r and show_this:
                full_hist = full_hist + [
                    {
                        "role": "assistant",
                        "content": str(r)[:800],
                        "metadata": {"title": "🧠 Tok myślenia"},
                    }
                ]
                yield _display(full_hist, _show_debug[0]), full_hist

            # ── LLM mówi coś przed tool callem ──
            if msg_content and msg.tool_calls:
                full_hist = full_hist + [
                    {
                        "role": "assistant",
                        "content": msg_content[:500],
                        "metadata": {"title": "💬 Komentarz modelu"},
                    }
                ]
                yield _display(full_hist, _show_debug[0]), full_hist

            # ── Brak tool calls → finalna odpowiedź ──
            if not msg.tool_calls:
                if tool_log:
                    lines = []
                    for i, (name, a, res) in enumerate(tool_log, 1):
                        short_res = res[:150] + "..." if len(res) > 150 else res
                        lines.append(f"{i}. `{name}({a})`\n   → {short_res}")
                    summary = "\n\n".join(lines)
                    full_hist = full_hist + [
                        {
                            "role": "assistant",
                            "content": summary,
                            "metadata": {"title": "📋 Użyte narzędzia"},
                        }
                    ]
                    yield _display(full_hist, _show_debug[0]), full_hist

                content = msg_content or "(brak odpowiedzi)"
                full_hist = full_hist + [{"role": "assistant", "content": content}]
                yield _display(full_hist, _show_debug[0]), full_hist
                return

            # ── Wykonanie tool calls ──
            messages.append(msg)

            for tc in msg.tool_calls:
                func_name = tc.function.name
                func_args = json.loads(tc.function.arguments)

                args_str = json.dumps(func_args, ensure_ascii=False)
                tool_content = f"**Wywołanie:** `{func_name}({args_str})`\n\n"

                try:
                    if func_name in available_tools:
                        result = available_tools[func_name](**func_args)
                    else:
                        result = f"❌ Nieznane narzędzie: {func_name}"
                except Exception as e:
                    result = f"❌ Błąd: {e}"

                tool_log.append((func_name, args_str, result))

                display_result = result[:300] + "..." if len(result) > 300 else result
                tool_content += f"**Wynik:** {display_result}"

                full_hist = full_hist + [
                    {
                        "role": "assistant",
                        "content": tool_content,
                        "metadata": {"title": f"🔧 Narzędzie: {func_name}"},
                    }
                ]
                yield _display(full_hist, _show_debug[0]), full_hist

                messages.append(
                    {"role": "tool", "tool_call_id": tc.id, "content": result}
                )

        # ── Limit kroków ──
        full_hist = full_hist + [
            {
                "role": "assistant",
                "content": f"⚠️ Osiągnięto limit {max_steps} kroków agenta.",
            }
        ]
        yield _display(full_hist, _show_debug[0]), full_hist

    # ── Toggle debug ──
    def toggle_debug(full_hist):
        _show_debug[0] = not _show_debug[0]
        label = "🔍 Pod maską: ON" if _show_debug[0] else "🔍 Pod maską: OFF"
        return _display(full_hist, _show_debug[0]), gr.update(value=label)

    # ── Przykładowe pytania ──
    examples = [
        ["Jaka jest pogoda we Wrocławiu?"],
        ["Opowiedz mi jakiś mało znany fakt o Andrzeju Dudzie."],
        ["Jaki jest aktualny kurs dolara do złotówki?"],
        ["Znajdź informacje o Nikoli Tesli na Wikipedii."],
    ]

    # ── Budujemy UI (Gradio 6) ──
    with gr.Blocks(
        title="🤖 Chat z narzędziami — Function Calling",
        theme=gr.themes.Soft(),
        css="""
            .gradio-container { max-width: 900px !important; margin: auto; }
            footer { display: none !important; }
        """,
    ) as demo:

        # Stan
        full_history = gr.State(value=[])

        # ════════════════════════════════════════════════════════════
        #  EKRAN LOGOWANIA (widoczny tylko gdy needs_login)
        # ════════════════════════════════════════════════════════════
        with gr.Column(visible=needs_login) as login_screen:
            gr.Markdown(
                "# 🤖 Chat z narzędziami\n"
                f"*Model: `{model_name}`*"
            )
            with gr.Group():
                login_user = gr.Textbox(
                    label="Użytkownik",
                    placeholder="Twoje imię...",
                    autofocus=True,
                )
                login_pass = gr.Textbox(
                    label="Hasło",
                    type="password",
                    placeholder="Hasło od prowadzącego...",
                    visible=needs_password,
                )
                login_error = gr.Markdown(visible=False)
                login_btn = gr.Button("Wejdź ➤", variant="primary")

        # ════════════════════════════════════════════════════════════
        #  EKRAN CHATU (ukryty do logowania, widoczny od razu gdy brak loginu)
        # ════════════════════════════════════════════════════════════
        with gr.Column(visible=not needs_login) as chat_screen:
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
                clear_btn = gr.ClearButton(
                    [chatbot, textbox], value="🗑️ Wyczyść chat"
                )
                toggle_btn = gr.Button("🔍 Pod maską: ON", variant="secondary")

            gr.Examples(examples=examples, inputs=textbox, label="Przykładowe pytania")

        # ── Logika logowania ──
        def do_login(username, password):
            if not username.strip():
                return (
                    gr.update(),                         # login_screen
                    gr.update(),                         # chat_screen
                    gr.update(visible=True, value="❌ Wpisz swoją nazwę użytkownika"),
                )
            if auth_fn:
                result = auth_fn(username, password)
                if not result or result is False:
                    return (
                        gr.update(),
                        gr.update(),
                        gr.update(visible=True, value="❌ Nieprawidłowe hasło"),
                    )
                # auth_fn zwraca nowego klienta
                if result is not True:
                    _client[0] = result
            elif auth_password:
                if password != auth_password:
                    return (
                        gr.update(),
                        gr.update(),
                        gr.update(visible=True, value="❌ Nieprawidłowe hasło"),
                    )
            # Sukces → zapamiętaj username, ukryj login, pokaż chat
            _username[0] = username.strip()
            return (
                gr.update(visible=False),
                gr.update(visible=True),
                gr.update(visible=False),
            )

        if needs_login:
            login_btn.click(
                do_login,
                [login_user, login_pass],
                [login_screen, chat_screen, login_error],
            )
            # Enter w polu użytkownik lub hasło → loguj
            login_user.submit(
                do_login,
                [login_user, login_pass],
                [login_screen, chat_screen, login_error],
            )
            login_pass.submit(
                do_login,
                [login_user, login_pass],
                [login_screen, chat_screen, login_error],
            )

        # ── Logika chatu ──
        textbox.submit(
            respond,
            [textbox, chatbot, full_history],
            [chatbot, full_history],
        ).then(lambda: "", None, textbox)

        submit_btn.click(
            respond,
            [textbox, chatbot, full_history],
            [chatbot, full_history],
        ).then(lambda: "", None, textbox)

        toggle_btn.click(
            toggle_debug,
            [full_history],
            [chatbot, toggle_btn],
        )

        clear_btn.click(lambda: [], None, full_history)

    print("\n🚀 Uruchamiam chat UI...")
    print("   Otwieram w nowej karcie przeglądarki...\n")
    demo.launch(
        share=share,
        quiet=False,
        inbrowser=True,
        height=300,
    )
