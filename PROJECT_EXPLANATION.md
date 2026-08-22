# AttendEase Secure v9 — Interview Explanation

## Core idea
A teacher-controlled classroom attendance kiosk with one-time face enrollment, Firebase-backed class data, randomized challenge-response liveness, and class-specific LBPH face recognition.

## Liveness
Each verification attempt uses exactly four server-generated prompts: **2 blink** and **2 smile**. Their order is shuffled for every attempt using a cryptographically secure random generator. The server reveals only the current action; the next prompt appears only after the current one passes.

This is designed to reduce basic photo and fixed prerecorded-video spoofing while keeping the challenge practical on a normal laptop webcam. It is not equivalent to dedicated depth/IR or certified anti-spoofing hardware.
