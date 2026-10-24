# DADA2 — Agentic RAG over SEC filings

**Fill in both sections at the bottom before you submit.** Keep the deployed
app line exactly in this form; it is how your app gets found and graded.

## Getting started

There is no separate spec. The instructions are in the notebooks, and
`DADA2_00_Start_Here.ipynb` is the one to open first: it explains the
assignment, the setup, the grading, and what each of the four notebooks after
it asks for. Work through them in order.

Get the files the same way as every class day: pull, then work in a copy:

```bash
cd lecture-files
git pull
cd ..
cp -r lecture-files/DADA2 DADA2     # work in the copy, not in lecture-files/
cd DADA2
cp example.env .env                  # then fill in your keys
```

Then open `DADA2_00_Start_Here.ipynb` from your `DADA2` copy and run its
setup check.

Deploying is `DEPLOY.md`.

---

## Submission

```
Deployed app: https://sec-filings-agent-evlc.onrender.com
```

**Name** : **HARSHA PRAKASH** (forgot to add the name to the deployment in the beginning and later added it, but the app was already deployed so couldn't change the link later. But did change the file name.)


### Write-up

---

**Conditions.**
The corpus is 27 section files (Business, Risk Factors and MD&A) from nine 10-K filings by NVIDIA, AMD and Micron, plus their XBRL financials as JSON. The sections were split with RecursiveCharacterTextSplitter (chunk size 1200, overlap 150) into 2,124 chunks, and each chunk carries source, ticker, fiscal_year and section metadata. Chunks were embedded with @cf/qwen/qwen3-embedding-0.6b on Cloudflare Workers AI (1024 dimensions) and stored in a Pinecone serverless index with cosine similarity. Retrieval uses k = 5 (k = 10 on the comparative set), a rerank pool of 10 and an RRF constant of 60, and reranking uses Cohere rerank-v3.5. The chat model is gemini-3.5-flash-lite and the decision model is clef-flash on Cloudflare. The retrieval strategies in notebook 02 were developed on a six-question slice of the specific set, and on five comparative questions at k = 10. The routers were compared on all 24 labelled questions (14 specific, 10 comparative). The agents were compared on 14 questions (8 specific, 6 comparative), with answers judged by the chat model.

**The routers.**
On all 24 labelled questions, the decision model routed 20 correctly at 0.40 s per call, and the chat model routed 18 correctly at 9.40 s per call. The accuracy gap is two questions. The speed gap is larger than a model difference would explain, and it isn’t a fair measure of the chat model itself. chat_scope calls Gemini through retry, so any backoff from the free tier’s 15-requests-a-minute limit is counted as model time, while the decision model runs on Cloudflare with no such limit. Speed aside, the decision model earns its place for two reasons. First, it returns a probability for every option, and the agent acts on those numbers directly: route falls back to DEFAULT below SURE, and diagnose re-labels a low-confidence absent as wrong_words below ABSENT_SURE. In notebook 03, the TSMC search was diagnosed incomplete at only 0.38, while the dress-code search was absent at 0.90. A structured-output chat call would have returned only a label in both cases. Second, its calls don’t count against Gemini’s 500-a-day limit, which matters because the agents already spend 2.4 to 4.7 chat calls per question.

**The agents.**
On 14 questions, the diagnosis agent answered 0.86 correctly against 0.79 for the ladder, a difference of one question. It did this with about half the chat calls (2.4 per question against 4.7) and in less time (20.8 s per question against 37.3 s), plus 2.1 decision-model calls per question that don’t count against the Gemini limit. On the specific questions the judge sampled, the diagnosis agent took the shortest path in the graph every time: one filtered search, a sufficient verdict, and an answer, for 2 chat and 2 decision calls. The ladder can’t stop that early, because it has no verdict to stop on. The 4.2 run reports only averages, so it doesn’t show which questions the agents split on. The clearest difference I measured directly is from notebook 03, on the out-of-scope dress-code question. There, the ladder climbed to rung 2 and spent 9 chat calls over 3 attempts before saying the filings don’t mention it. The diagnosis agent stopped after a single search, because diagnose returned absent at 0.90, above ABSENT_SURE, and decide sends absent straight to answer. On the TSMC comparison, the ladder used 6 chat calls. The diagnosis agent’s first hybrid search came back incomplete, and because plan is told which sources it already has, it wrote only one new step for the missing company before the second diagnosis came back sufficient. The difference between the agents is what happens after a failure: the ladder retries the same way every time, while the diagnosis agent sends each verdict to a different node.

**What did not help.**
Multi-query RRF on top of the metadata filter. On the six-question slice in notebook 02, the filter alone reached recall@5 1.00 and MRR 0.917. Adding four rewrites fused with RRF kept recall at 1.00 but dropped MRR to 0.792. Adding the reranker only brought it back to 0.806, still below the filter alone, at roughly five extra calls per question. The reason is the order of the pipeline. Once the filter has narrowed the search to a single filing, there is almost nothing left for a different phrasing to find, and the rewrites can drift off-topic (one was “regulatory compliance hazards in global shipping”). RRF then rewards passages that the drifting rewrites agree on, which can push the single best passage down a rank. For questions that already name a company and a year, the filter does all the useful work.


**What I would ship.**
I would ship the diagnosis agent. It was slightly more accurate than the ladder (0.86 against 0.79 on 14 questions) and cost roughly half as much in chat calls and time. On specific questions I would drop multi-query fusion and use the plain filter, since it lowered MRR without improving recall. Before shipping, I would want to know what mix of questions users ask. The filter raised recall@5 from 0.67 to 1.00 on specific questions but dropped recall@10 from 0.83 to 0.57 on comparative ones, so the router’s thresholds should be tuned on real traffic rather than on 24 labelled questions. I would also want to know whether users expect the latest filing by default. When I asked the deployed app “Is anyone worried about China?”, it answered only from fiscal 2024 filings for all three companies, because similarity search has no notion of recency.