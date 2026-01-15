# Walkthrough - Advanced Capabilities 🚀

I have implemented the follow-up features to turn Aura into a more capable local agent.

## New Features

### 1. Filesystem & OS Control 🛠️
Aura can now interact with your computer in more ways:
- **Directory Listing**: Ask "What's in my current folder?" to see your project structure.
- **File Reading**: Ask "Read the content of requirements.txt" to have her analyze local files.
- **Volume Control**: Ask "Set my volume to 50%" (uses `amixer` on Linux).
- **Expanded System Info**: Now includes **Battery Status** alongside CPU and RAM.

### 2. Self-Learning Memory (Aura v2) 🧠
Aura now has a subconscious "Observer" that listens for facts:
- **Passive Memory**: You don't need to say "Remember". Just speak naturally.
- **Background Extraction**: If you say "My girlfriend Giana broke up with me," the Observer silently notes: `Fact: Jeff broke up with Giana`.
- **Immediate Indexing**: Facts are available to the RAG system *immediately* for the next sentence, not just after restart.

## Final Polish & Logic Tuning 💎
To ensure Aura is professional yet fun, we implemented three key logic layers:
1.  **Chatter Filter**: Aura now ignores the database for simple greetings ("hi", "ok") to prevent "hallucinating" context for small talk.
2.  **Banter Mode**: The system prompt now explicitly tells her to prioritize sarcasm and wit over database searches when you are making jokes.

## Code Changes

### [aura.py](file:///home/jk/Projects/aura-local/aura.py)
- Added 5 new tool definitions.
- Implemented tool execution logic in `_execute_tool`.
- Integrated `_update_long_term_memory()` into the main execution loop.

### [system_prompt.txt](file:///home/jk/Projects/aura-local/system_prompt.txt)
- Added rules for properly using the new memory context.

## Verification

The script has been verified to start correctly. Due to the interactive nature of the memory and volume tools, I recommend you try the following:

1.  **Memory Test**: Tell Aura "My favorite color is deep forest green."
2.  **Exit**: Type `exit`. Observe the "Updating Aura's long-term memory..." message.
3.  **Restart**: Run `python3 aura.py` again.
4.  **Confirm**: Ask "What's my favorite color?" and she should pull it from her memory!
