# Tasks: Aura v2 (Memory Redesign)

- [x] **Phase 1: The Observer Architecture**
    - [x] Create `FactExtractor` class in separate module (or `aura.py` if small).
    - [x] Implement `extract_facts(text)` method using Llama 3.2 with strict JSON prompting.
    - [x] Create background worker `_memory_observer_loop` in `AuraAssistant`.
- [x] **Phase 2: RAG Pipeline Upgrade**
    - [x] Modify `KnowledgeBase.add()` to accept tagged entries (Fact vs Document).
    - [x] Remove old `_update_long_term_memory` (Session Summary) logic.
    - [x] Connect User Input -> Observer -> KnowledgeBase.
- [ ] **Phase 3: Verification (The "Life" Test)**
    - [ ] Test Passive Preference: "I like snowboards".
    - [ ] Test Relationship Update: "Giana broke up with me".
    - [ ] Test Future Intent: "Thinking of a black cat".
