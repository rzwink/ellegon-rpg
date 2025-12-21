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
ellegon/
├── app.py
├── system_prompt.txt
├── ellegon-campaign.schema.json
├── campaigns/
│   └── goblin_cave/
│       └── campaign.json
├── saves/
│   └── (generated at runtime)
├── README.md
├── LICENSE
└── CONTRIBUTING.md
````

### Key Files

* app.py
  The main CLI application that runs Ellegon.

* system_prompt.txt
  Defines Ellegon’s personality, tone, and Dungeon Master behavior.

* ellegon-campaign.schema.json
  JSON Schema used to validate campaign files.

* campaigns/
  Contains individual campaign definitions.

* saves/
  Generated automatically. Stores progress per campaign instance.

---

## Campaign Format

Campaigns are defined entirely in JSON and validated against a schema.

A campaign includes:

* Campaign overview and goals
* A clear completion condition
* A starting state
* Acts with success and failure guidance
* Locations and NPCs
* Rewards and endings
* Optional spoken intro text

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

Set your API key (or place in .env):

```bash
export OPENAI_API_KEY="your_key_here"
```

### Run a Campaign

```bash
python app.py --campaign goblin_cave --instance kid1 --players 1
```

Arguments:

* --campaign
  Folder name under campaigns/

* --instance
  Unique identifier for this playthrough

* --players
  Number of players currently participating

Progress is saved automatically after each turn.

---

## Voice Interaction

Ellegon is designed for voice first play:

* Short responses
* Clear prompts
* Dice roll instructions
* Simple questions

Speech to text and text to speech are intentionally left out of this repo so you can choose the tools that fit your platform.

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

## Contributing

Contributions are welcome for:

* Bug fixes
* Campaign examples
* Documentation improvements

All contributions must agree to the terms in CONTRIBUTING.md.

By contributing, you agree that your contributions may be relicensed by the project owner.

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
