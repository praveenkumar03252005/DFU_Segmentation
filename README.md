# Diabetic Foot Ulcer (DFU) Live Segmentation Backend

FastAPI microservice for live Diabetic Foot Ulcer boundary segmentation and domain validation.

## Architecture
- **Inference Hardware:** CPU-only mode
- **Certified Deployment Candidate:** Stage 16 Boundary-only U-Net (Seed 42, $0.50$ Frozen Threshold)
- **Upstream Validation:** CLIP Zero-Shot Domain Validator (rejects non-DFU inputs before segmentation)
- **Model Distribution:** Automated high-speed retrieval from Hugging Face CDN upon startup

## API Endpoints
- `GET /api/health` - System health check & CPU runtime status
- `POST /api/validate-image` - Upstream domain integrity validation
- `POST /api/segment` - Live single-image inference
- `GET /api/models` - Certified model catalog
- `POST /api/export` - Comprehensive audit package export

## Run Locally
```bash
pip install -r requirements.txt
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```
