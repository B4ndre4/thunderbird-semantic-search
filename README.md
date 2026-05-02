# Thunderbird Semantic Search

> Index and search your Thunderbird emails using hybrid search (vector embeddings + full-text search) and LLM classification.

## About

These scripts were created to solve a real work problem while serving as a first experience in coding with AI agent assistance. Most of the code was written using [Opencode](https://opencode.ai), which was also used to review code I wrote myself. All code written by the agent and modifications to my code were checked and tested to the best of my abilities. For exploring implementation options, the Sonnet LLM was used via free web browser chat, while Opencode used another model. An LLM was also used to organize comments in the code and to edit this README. For me, the experiment was a success: I have a tool that solves my problem and I learned a lot.

The utility scripts in `utils/` were created directly by an LLM from a single prompt and are used for testing and downloading embedding models.

### My Use Case

I have approximately 180,000 emails spread across a dozen mbox files covering 11 years. After the initial backlog processing (which took considerable time), I now run `censor_agent.py` before opening the Thunderbird client so the database gets updated with emails received since the last time the client was used. I then use the MCP server with an AI agent (e.g., AnythingLLM or Opencode itself) to perform semantic searches in my mail. Now I can ask the agent things like "years ago I had contact with someone who solved problem X but I don't remember who or when, search my mail and summarize everything you find."

## Features

- Parse Thunderbird Mbox files and extract email metadata and content (note: attachments are not processed)
- Classify emails as work, private, or mixed using configurable LLM prompts
- Extract and clean work-related content, removing signatures and boilerplate using configurable LLM prompts
- Generate vector embeddings using fastembed models
- Store embeddings in LanceDB for efficient semantic search
- Full-text search (FTS) on email subjects and content
- Hybrid search combining semantic similarity and keyword matching via RRF (Reciprocal Rank Fusion)
- Resume interrupted processing from where it left off (essential for large mbox files that take considerable time to process)
- MCP server for integration with AI agents (uses stdio transport)
- Support for multiple LLM providers: Ollama (local), Ollama Cloud, or Llama.cpp
- Deduplication using SHA-256 hashes (prevents re-indexing duplicates when processing files multiple times or when emails exist in multiple mbox files)
- Retry mechanism for failed emails

## Getting Started

### Prerequisites

- Python 3.11 or higher
- For local LLM: Ollama or Llama.cpp with a suitable model
- For Ollama Cloud: an API key (can be generated from the Ollama website)
- For embeddings: fastembed models downloaded locally

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/B4ndre4/thunderbird-semantic-search.git
   cd thunderbird-semantic-search
   ```

2. Install dependencies:
   ```bash
   pip install lancedb fastembed ollama llama-cpp-python mcp pyarrow rich click
   ```

3. Download an embedding model locally:
   ```bash
   python utils/fastembed_download_model.py
   ```
   
4. Copy the example configuration:
   ```bash
   cp config/config.example.toml config/config.toml
   ```

5. Edit `config/config.toml` with your settings:
   - Set absolute paths to your Mbox files
   - Configure the embedding model and cache path
   - Choose your LLM provider and model (for Ollama Cloud, add your API key)
   - Replace the English prompts and patterns in the example configuration with those from your chosen language in `config/languages.txt`
   - Adjust chunk size and overlap as needed (avoid setting chunk size too large if your emails are generally short)

**Note:** The prompts in German, French, and Spanish in `languages.txt` are AI-generated. I do not know these languages and cannot be sure they are correct. I personally use the Italian prompt because the emails I need to search are in that language, and I get better results from LLM models using prompts in the same language as the emails.

## Usage

### Indexing Emails

Process a single mbox file:
```bash
python censor_agent.py --file /path/to/your.mbox
```

Process all mbox files configured in config.toml:
```bash
python censor_agent.py
```

Retry failed emails for a specific mbox:
```bash
python censor_agent.py --retry /path/to/your.mbox
```

Silent mode (no console output):
```bash
python censor_agent.py --silent
```

Debug mode (verbose logging to file):
```bash
python censor_agent.py --debug
```

#### Understanding Processing Statistics

After processing, you will see statistics like:
- **Processed**: Total emails examined in the current run
- **Indexed**: Emails successfully classified, extracted, and stored in the vector database
- **Skipped**: Emails already processed previously (deduplication based on content hash)
- **Discarded**: Emails classified as "private" or with no substantive work content after extraction
- **Errors**: Emails that failed during classification, extraction, chunking, embedding, or storage (these are tracked and can be retried)

### Running the MCP Server

The MCP server provides hybrid search capabilities (semantic + full-text) to AI agents:

```bash
python mcp_server.py
```

With debug logging:
```bash
python mcp_server.py --debug
```

The MCP server must be configured in your chosen AI client. For example, in AnythingLLM the configuration is done via a JSON file:

```json
{
  "mcpServers": {
    "email-search-mcp": {
      "command": "python",
      "args": [
        "D:/myfolder/thunderbird-semantic-search/mcp_server.py",
        // Note: --debug is recommended only for troubleshooting issues
        "--debug"
      ],
      "env": {
        "PYTHONPATH": "D:/myfolder/thunderbird-semantic-search"
      },
      "anythingllm": {
        "autoStart": true,
        "suppressedTools": []
      }
    }
  }
}
```

The server exposes two tools:
- `search_emails`: Perform hybrid search over indexed emails combining vector similarity and full-text search (FTS) using Reciprocal Rank Fusion (RRF)
- `get_system_status`: Get indexing statistics and configuration info

#### How Hybrid Search Works

When you search for emails, the system combines two approaches:
1. **Semantic search**: Finds emails with similar meaning using vector embeddings
2. **Full-text search**: Finds emails matching exact keywords in subject and content

Results are merged using **Reciprocal Rank Fusion (RRF)**, which ranks documents based on their positions in both result sets. This ensures you get relevant results whether you use specific keywords or natural language queries.

### LLM Recommendations

For the classification and indexing phase (`censor_agent.py`), using a local model is recommended if you have many emails to process, to avoid the token costs of cloud models. I have obtained good results with `qwen3.5:9b-q8_0` in acceptable times on an RX 7800XT GPU, but your mileage may vary depending on your hardware and specific use case.

## Project Structure

```
thunderbird-semantic-search/
├── censor_agent.py          # Main indexing agent
├── mcp_server.py            # MCP server for semantic search
├── src/                     # Core modules
│   ├── config.py            # Configuration loading
│   ├── mbox_parser.py       # Mbox parsing and email extraction
│   ├── chunker.py           # Text chunking for embeddings
│   ├── embedder.py          # fastembed wrapper
│   ├── vector_store.py      # LanceDB interface
│   ├── state_db.py          # SQLite state tracking
│   ├── llm_provider.py      # LLM provider interface
│   ├── ollama_provider.py   # Ollama implementation
│   └── llamacpp_provider.py # Llama.cpp implementation
├── config/
│   ├── config.example.toml  # Example configuration
│   └── languages.txt        # Prompts in multiple languages
├── data/                    # Vector DB and state DB (gitignored)
├── models/                  # fastembed cache (gitignored)
└── utils/                   # Utility scripts
```

## Important: Backup Your Data

**It is strongly recommended to BACKUP your Mbox files before using this tool.** While the scripts only read from Mbox files and never write to them, and no corruption has ever been observed, having a backup ready is always a good idea.

## License

Distributed under the MIT License. See `LICENSE` for full text.
