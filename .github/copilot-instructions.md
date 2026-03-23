# Project Guidelines

## Scope and Structure
- This repository is a multi-project sandbox, not a single deployable app.
- Treat each top-level project folder as independent unless the user asks to integrate across folders.
- Before changing code, confirm the target subproject from the path and keep edits local to that area.

## Code Style
- Preserve existing style in each subproject; do not run broad reformatting across unrelated folders.
- Mixed Chinese/English comments and strings are intentional in several projects; keep language consistent with nearby code.
- Prefer minimal, surgical changes over large refactors.

## Architecture
- `P_Tianjian_Holistic_governance/`: RAG and workflow orchestration (FAISS + LangGraph + LLM calls).
- `P_Dimai_Parasite_testing/`: YOLOv8 training/evaluation scripts for microscopy tasks.
- `Docker_GitLab/` and `Docker_wiki/`: standalone Docker Compose infrastructure.
- Tutorial and notebook-focused folders (for example `LangGraph_toturial/`, `RAG_toturial/`, `test/`) may be exploratory and not production-hardened.

## Build and Test
- No repo-wide build or test command exists; run commands only within the relevant subproject.
- Common commands used in this workspace:
  - `python P_Dimai_Parasite_testing/train_and_eval_yolov8.py`
  - `python P_Dimai_Parasite_testing/yolo_test.py`
  - `python P_Tianjian_Holistic_governance/buildFaiss.py`
  - `python P_Tianjian_Holistic_governance/handlerFinder.py`
  - `docker-compose -f Docker_GitLab/docker-compose.yml up -d`
  - `docker-compose -f Docker_wiki/docker-compose.yaml up -d`
- Prefer lightweight validation first (import checks, targeted script runs) before long GPU training or container startup.

## Environment and Safety
- Python environments are typically Conda-based in this workspace.
- Many scripts depend on local files, GPU/CUDA, and `.env` keys; warn early if these are missing.
- Do not hardcode or expose secrets. Keep API keys in environment variables.
- Some scripts include machine-specific absolute paths (for example under `/data/...` or `/srv/...`); avoid assuming portability.

## Conventions and Pitfalls
- Keep Chinese domain terminology intact in governance and vision projects.
- Avoid creating global dependency assumptions: there is no single `requirements.txt` for the whole repo.
- For notebooks, prefer editing only requested cells/files and avoid unnecessary kernel-side changes.

## Reference Docs
- Root overview: `README.md`
- Image insertion workflow: `InsertImages/README.md`
- Road map generation notes: `InsertImages/Road_map_generation/README.md`
- Wallpaper notes: `InsertImages/Wall_paper/README.md`
