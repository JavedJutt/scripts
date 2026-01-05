## Setup

1.  **Install Dependencies**:

    ```bash
    pip install -r requiremets.txt
    ```

    *(Note: The requirements file is named `requiremets.txt` in this repo).*

2.  **Environment Variables**:
    You can set your API keys in a `.env` file in the same directory, or export them in your terminal.
    
    **Option A: .env file**
    Create a file named `.env` and add:
    ```env
    PINECONE_SOURCE_API_KEY="your-source-api-key"
    PINECONE_DEST_API_KEY="your-destination-api-key"
    ```

    **Option B: Terminal Export**
    ```bash
    export PINECONE_SOURCE_API_KEY="your-source-api-key"
    export PINECONE_DEST_API_KEY="your-destination-api-key"
    ```

## Usage

Run the script as a module from the `pinecone_index_migration` directory.

### Interactive Mode
If you run without arguments, the script will prompt you for the index names.

```bash
python -m pinecone_migration
```


## How it works

1.  **Connects to Source**: Verifies the source index exists and reads its configuration (dimension, metric).
2.  **Connects to Destination**: Checks if the destination index exists.
    *   If **Yes**: Verifies that dimensions match.
    *   If **No**: Creates a new Serverless Index (AWS us-east-1) with the correct configuration.
3.  **Migrates Data**: correctly iterates through all namespaces in the source index, downloads vectors in batches, and upserts them into the destination index.

## Notes

- The script uses `pinecone_manager.py` for client connections.
- Ensure your source index supports listing IDs (standard Serverless and Pod interactions usually support this via the `list` endpoint in the new SDK).
- If you need to use the `openai` features of the manager elsewhere, ensure `OPENAI_API_KEY` is set; otherwise, it is skipped for migration.
