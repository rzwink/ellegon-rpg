# Ellegon

Ellegon is a voice friendly, AI powered Dungeon Master engine.
It runs short, guided fantasy adventures with clear rules, structured campaigns, and a definite ending.

Ellegon is built to feel like a real tabletop experience:
- Spoken narration
- Dice driven outcomes
- Inventory and currency tracking
- Acts, locations, NPCs, and completion conditions
- Friendly, age appropriate storytelling

This project is **source available**, not open source.

---

## What Ellegon Is

Ellegon is:
- A narrative engine for running tabletop style adventures
- Optimized for text to speech and speech to text
- Designed for kids around ages 8 to 12
- Structured so campaigns can be completed, not endless
- Built for extensibility and future commercialization

Ellegon is intentionally:
- Deterministic enough to finish a story
- Flexible enough to allow creativity
- Lightweight enough to run locally

---

## What Ellegon Is Not

Ellegon is not:
- A replacement for human Dungeon Masters
- A full D and D rules engine
- A virtual tabletop
- A hosted service
- An open source project

---

## Repository Structure

```text
ellegon-rpg/
├── app.py
├── system_prompt.txt
├── ellegon-campaign.schema.json
├── campaigns/
│   ├── 001/
│   │   └── campaign.json
│   └── 002/
│       └── campaign.json
├── ellegon/
│   ├── cli.py
│   ├── apps/
│   ├── config.py
│   ├── campaigns/
│   ├── llm/
│   ├── prompts/
│   ├── service/
│   ├── sessions/
│   └── speech/
├── tests/
├── requirements.txt
├── README.md
└── LICENSE
```

### Key Files

* app.py
  Entry point for the CLI (calls `ellegon.cli.main`).

* ellegon/cli.py
  Argument parsing, save loading, and the interactive play loop.

* ellegon/service/engine.py
  Session orchestration and LLM calls.

* ellegon/prompts/instructions.py
  Builds the prompt payload given the campaign and save state.

* ellegon-campaign.schema.json
  JSON Schema used to validate campaign files.

* system_prompt.txt
  Defines Ellegon’s personality, tone, and Dungeon Master behavior.

* campaigns/
  Campaign definitions (each folder contains `campaign.json` with inline intro text).

* saves/
  Generated automatically at runtime as `saves/<campaign>/<instance>.json`.

---

## Campaign Format

Campaigns are defined in JSON and validated against `ellegon-campaign.schema.json`.

A campaign includes:

* Campaign overview and goals
* A clear completion condition
* A starting state
* Acts with success and failure guidance
* Locations and NPCs
* Rewards and endings
* Optional intro text stored inline in `campaign.json`

This allows:

* Static validation
* AI friendly context loading
* Predictable story pacing
* Deterministic completion

---

## Running Ellegon

### Requirements

* Python 3.10+
* An OpenAI API key
* Optional microphone and speaker for voice interaction

### Setup

```bash
pip install -r requirements.txt
```

Optional extras:

```bash
pip install ".[api]"
pip install ".[pygame]"
```

Set your API key (or place in `.env`):

```bash
export OPENAI_API_KEY="your_key_here"
```

Optional overrides:

```bash
export ELLEGON_CAMPAIGNS_ROOT="/path/to/campaigns"
export ELLEGON_SAVES_ROOT="/path/to/saves"
```

### Run a Campaign

```bash
python app.py --campaign 001 --instance kid1 --players 1
```

Arguments:

* --campaign
  Folder name under `campaigns/` (for example `001`).

* --instance
  Unique identifier for this playthrough; used as the save file name.

* --players
  Number of players currently participating.

* --model
  OpenAI model name (defaults to `gpt-5.2-2025-12-11`).

* --fake-gateway
  Use a deterministic fake LLM gateway (no network calls).

Progress is saved automatically after each turn.

---

## FastAPI Service

Create a FastAPI app for web sessions:

```bash
uvicorn ellegon.apps.api_app:create_app --factory --reload
```

---

## Pygame Voice Client

Install pygame extras and run the desktop client:

```bash
python -m ellegon.apps.pygame_app --campaign 001 --instance kid1 --players 1
```

The Pygame client uses OpenAI speech APIs for transcription and narration, so it requires
`OPENAI_API_KEY`.

---

## Voice Interaction

Ellegon is designed for voice first play:

* Short responses
* Clear prompts
* Dice roll instructions
* Simple questions

Speech to text and text to speech are provided via OpenAI speech APIs in the Pygame client.

---

## Tests

```bash
pytest
```

---

## License

This project is licensed under the **Business Source License (BSL 1.1)**.

### Important

* This is **not open source**
* Commercial use is **not permitted**
* Hosting this as a service is **not permitted**
* Redistribution is **not permitted**

You may:

* View the source
* Run it locally for personal use
* Modify it for personal experimentation

Commercial licenses are available.

See LICENSE for full terms.

---

## Roadmap

Planned and possible future work:

* Structured state updates from the model
* Voice input and output helpers
* Campaign generation tools
* Multi session campaign arcs
* Commercial campaign packs
* Hosted offerings

None of the above are guaranteed.

---

## Legal and Attribution

Ellegon is an original project.

Campaigns inspired by existing fantasy works are **fan created and non canonical**.
No copyrighted text or characters from published works are included.

---

## Contact

For commercial licensing, partnerships, or questions:

Rob Zwink
Email: rzwink@gmail


sudo apt-get update
sudo apt-get install -y libportaudio2 portaudio19-dev
pip install -U sounddevice
