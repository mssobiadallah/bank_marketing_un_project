# Streamlit Deployment Guide

## Deploying to Streamlit Community Cloud

### Quick Start

1. **Push your code to GitHub** (already done! ✅)
   ```bash
   git push origin main
   ```

2. **Go to [share.streamlit.io](https://share.streamlit.io)**

3. **Click "New app"**

4. **Fill in the deployment form:**
   - **Repository**: `mssobiadallah/bank_marketing_un_project`
   - **Branch**: `main`
   - **Main file path**: `app/streamlit_app.py`

5. **Click "Deploy!"**

The app will take 2-3 minutes to build and deploy.

---

## What's Included in This Repo

✅ **All required files for deployment:**
- `requirements.txt` — All Python dependencies
- `app/streamlit_app.py` — Main app entry point
- `models/*.joblib` — Pre-trained model files (652KB total)
- `data/raw/*.csv` — Dataset files (11MB total)
- `.streamlit/config.toml` — App configuration
- `.python-version` — Python 3.12 specification

---

## File Sizes (Streamlit Cloud Limits)

| Item | Size | Limit | Status |
|------|------|-------|--------|
| Models | 652KB | 200MB | ✅ Well under limit |
| Data | 11MB | 200MB | ✅ Well under limit |
| Total repo | ~12MB | 1GB | ✅ No issues |

---

## App Structure

```
app/
├── streamlit_app.py          ← Main entry (7 pages)
└── pages/
    ├── 1_EDA_Dashboard.py
    ├── 2_Hypothesis_Testing.py
    ├── 3_Model_Performance.py
    ├── 4_Predict_New_Client.py
    ├── 5_Batch_Prediction.py
    └── 6_Business_Recommendations.py
```

---

## Environment Variables

No secrets or API keys needed! Everything runs locally.

---

## Testing Locally Before Deploy

```bash
# Activate virtual environment
source venv/bin/activate

# Run the app
streamlit run app/streamlit_app.py

# Or use make
make app
```

Open browser to `http://localhost:8501`

---

## Troubleshooting Deployment

### Issue: "Module not found"
**Fix**: Check that all imports in `requirements.txt` are listed

### Issue: "Model files not found"
**Fix**: Ensure `models/*.joblib` files are tracked in git:
```bash
git add models/*.joblib
git commit -m "Add model files for deployment"
git push
```

### Issue: App crashes on startup
**Check Streamlit logs**:
1. Go to your app dashboard on share.streamlit.io
2. Click "Manage app" → "Logs"
3. Look for error messages

---

## Performance Notes

- **Cold start**: 2-3 minutes (first deployment or after inactivity)
- **Warm start**: 5-10 seconds (after first load)
- **Model loading**: Cached after first page load
- **Memory usage**: ~200MB (well within free tier 1GB limit)

---

## Free Tier Limits (Streamlit Community Cloud)

✅ **What you get:**
- 1 public app
- Unlimited visitors
- 1 GB RAM
- 1 CPU core
- Auto-sleep after inactivity (wakes on visit)

📊 **This app uses:**
- ~200MB RAM
- 0.5 CPU cores average
- Models: 652KB
- Data: 11MB

**Result**: ✅ Perfect fit for free tier!

---

## Post-Deployment Checklist

After deployment:
- [ ] Test all 7 pages load correctly
- [ ] Try single prediction (page 4)
- [ ] Upload CSV for batch prediction (page 5)
- [ ] Check model metrics display (page 3)
- [ ] Verify charts render (pages 1-2)

---

## URL

Your app will be available at:
```
https://share.streamlit.io/mssobiadallah/bank_marketing_un_project/main/app/streamlit_app.py
```

Or a shorter custom URL if you configure it in settings.

---

## Updates

To update your deployed app:
```bash
git add .
git commit -m "Update app"
git push origin main
```

Streamlit Cloud will **automatically redeploy** in ~2 minutes.

---

## Need Help?

- [Streamlit Docs](https://docs.streamlit.io/streamlit-community-cloud/get-started)
- [Community Forum](https://discuss.streamlit.io)
- [GitHub Issues](https://github.com/mssobiadallah/bank_marketing_un_project/issues)
