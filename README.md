# RAG Mistral

## Usage

### Installation

```bash

git clone https://github.com/adddddddddddddddd/rag_doc_mistral.git --recurse-submodules

cd rag_doc_mistral

uv sync

docker compose up vespa -d

uv run python backend/vespa_utils.py deploy_app ./my-vespa-app

# Wait for Vespa to start running, about 30 seconds.
# While waiting, create a .env file and add it you MISTRAL_API_KEY
#then run the following command to feed the documents

cd backend

# Make sure you have correctly clone the sub-repo of mistral docs api -
uv run python feed.py --include_folders=["getting_started", "capabilities", "agents", "deployment", "mistral-vibe", "deprecated"]


```

In a terminal dedicated to backend

```bash

cd backend

uv run python server.py
```

In a terminal dedicated to frontend

```bash

cd chat_interface

pnpm install

pnpm dev
```

### How it works

It goes through all the .md files of the cited subsections and parse the information according to the following logic :
-> Seperate the file into chunks based on the h2 tags
-> if number of token of one chunk is superior than a fixed limit (constant) then seperates again thanks to h3 headers.
-> If too long, break into chunks of approximately 800 tokens
-> do not touch to code, unless it is bigger than 8192 tokens, the Mistral API limit

## Evaluation

To evaluate the performance of the RAG, I focused on the Agents subsection. To evaluate, I first use Recall@1 3 5 for the retrieval and faithfullness@1 .

### Methodology

To create such metrics, I only used sunthetic annotations. I took every chunk, made an LLM generate a question that is only about the chunk. Then I would see if the chunk is in the top-k retrieved chunks and validate or not the recall
Faithfullness is done with an LLM-as-a-judge. First I used Mistral Large because it is powerful and that what I always used. Here, reasoning suits better that tasks so I ended up making it with magistral

### Experiments

1. Raw RAG to start with a baseline (using hybrid vector search from vespa) :
- Recall@5 0.91
- Recall@3 0.82
- Recall@1 0.62
- Faithfullness@5 0.72

Faithfullness and recall@1 were a little low so I tried to improve it.
-> By adding HYDE (make the similarity search more meaningful - make recall@1 go up)
-> By deleting duplicates (tends to make the judge not overstimulated)
-> By changing judging model (I saw errors in the judgement of the judge)

Results :
Deleting the duplicates did not improved the recalls so no need to compute anything else.
- Recall@5 (hybrid): 0.89
- Recall@3 (hybrid): 0.83
- Recall@1 (hybrid): 0.61


Adding HYDE did not improved significantly enough the results to integrate it
- Recall@5 (hybrid): 0.89
- Recall@3 (hybrid): 0.80
- Recall@1 (hybrid): 0.557

Changing the judge to magistral-small actually gives us a faithfullness close to 1 (0.95), which is very good indeed

## Conclusion

Basic approche works best, I introduced segmentation of distinct part not to tmi.
