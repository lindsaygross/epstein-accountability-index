# Deployment Guide — Google Cloud Run

## Prerequisites

1. Install Google Cloud CLI: https://cloud.google.com/sdk/docs/install
2. Login: `gcloud auth login`
3. Set project: `gcloud config set project YOUR_PROJECT_ID`
4. Enable required APIs:
   ```bash
   gcloud services enable run.googleapis.com artifactregistry.googleapis.com cloudbuild.googleapis.com
   ```

---

## One-Command Deploy (recommended)

From the project root:

```bash
gcloud run deploy impunity-index \
  --source . \
  --region us-central1 \
  --platform managed \
  --allow-unauthenticated \
  --memory 4Gi \
  --cpu 2 \
  --timeout 300 \
  --concurrency 10 \
  --min-instances 1 \
  --max-instances 3 \
  --port 8080
```

`--source .` uses Cloud Build to build the Docker image automatically from your `Dockerfile`. No local Docker needed.

`--min-instances 1` keeps one instance warm so there's no cold start when the grader visits.

---

## Test Locally First (optional but recommended)

```bash
# Build the image
docker build -t impunity-index .

# Run locally
docker run -p 8080:8080 -e PORT=8080 -e FLASK_ENV=production impunity-index

# Verify
curl http://localhost:8080/api/people | python3 -c "import json,sys; d=json.load(sys.stdin); print(len(d), 'people')"
```

---

## Expected Build Time

| Step | Time |
|------|------|
| Docker build (first time) | ~10-15 min (torch + transformers are large) |
| Docker build (subsequent, cached) | ~2-3 min |
| Deploy to Cloud Run | ~2 min |
| App startup (first request) | ~10-15 sec |

---

## Cost Estimate ($50 credit)

With `--min-instances 1` (always-on, 4GB RAM, 2 CPU):
- ~$0.048/hour = ~$34/month
- Your $50 credit covers ~6 weeks easily

With `--min-instances 0` (scale-to-zero, free when idle):
- Free when not in use, but ~15s cold start on first request
- For grading: set min-instances 1 during the grading week

---

## After Deploy

You'll get a URL like:
```
https://impunity-index-xxxxxxxxxx-uc.a.run.app
```

Add this to your README.md as the live app URL.

---

## Monitoring

```bash
# View logs
gcloud run services logs read impunity-index --region us-central1

# Check service status
gcloud run services describe impunity-index --region us-central1

# Update (redeploy after code changes)
gcloud run deploy impunity-index --source . --region us-central1
```

---

## Troubleshooting

**Out of memory error:**
- Increase to `--memory 8Gi` (torch + chromadb can hit 3-4GB under load)

**Build timeout:**
- Cloud Build default is 10 min; torch download may exceed this
- Increase: `gcloud builds submit --timeout=1800`

**Citations returning empty:**
- ChromaDB (`chroma_db/`) must be present before build
- Verify: `ls chroma_db/` shows files before running `gcloud run deploy`

**Port mismatch:**
- Cloud Run always injects PORT=8080; our app reads `os.environ.get("PORT", 5001)`
- Dockerfile sets `EXPOSE 8080` — this is correct
