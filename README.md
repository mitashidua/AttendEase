# AttendEase Secure v9 — Random Blink + Smile Liveness

This version uses only **blink** and **smile** for liveness.

## Liveness flow
Every attendance attempt gets a fresh two-step server-generated sequence containing exactly:
- 2 × **Blink twice**
- 2 × **Smile, then relax**

The four steps are shuffled using a cryptographically secure random generator. The browser does **not** receive the full sequence in advance; the server reveals only the current action and sends the next action only after the current one passes. A short random prompt-to-capture delay is also used.

This makes a fixed ordinary prerecorded video harder to reuse, but a normal RGB webcam cannot guarantee protection against sophisticated adaptive replay/deepfake attacks.

## Everything else remains
- Firebase Realtime Database
- one-time face enrollment
- class-specific LBPH face recognition
- teacher-controlled classroom kiosk
- duplicate attendance prevention
- multilingual UI
- light/dark mode
- IST display and CSV export
- port `5055`

## Run
1. Extract the ZIP.
2. Run `setup_windows.bat`.
3. Open `http://127.0.0.1:5055`.
4. On first run, enter the same Firebase Realtime Database URL and service-account JSON.
5. Login, open a class, start attendance, and open the secure kiosk.
6. Complete the 4 random prompts — 2 blink + 2 smile — then face recognition runs automatically.

If you already have a configured older version, you can copy its private `config/` folder into this project before starting. Never commit that folder or your service-account JSON to GitHub.
