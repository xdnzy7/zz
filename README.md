# Live Momentum Scanner

Streamlit web app for continuous small-cap momentum scanning and live trade setup planning.

## What It Does

- Starts scanning automatically when the app loads.
- Runs a background scanner thread every 5-10 seconds by default.
- Keeps results in memory instead of Excel files.
- Shows top momentum candidates and analyzed trade plans in a browser.
- Supports fast mode, scan interval control, and scanner restart from the sidebar.

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy On Streamlit Cloud

1. Push this folder to GitHub.
2. Go to https://streamlit.io/cloud.
3. Connect your GitHub repo.
4. Select `app.py` as the Streamlit app file.
5. Deploy.

## Notes

- The app does not require local Excel files.
- Data is fetched from public market data endpoints and `yfinance`.
- Streamlit Cloud may throttle or sleep free apps depending on platform limits.
- This tool is for trade planning and education only, not financial advice.
