# Meteorology Web App Frontend

This React frontend uploads a rainfall chart image to the FastAPI backend, displays extracted values in a table, and renders a line chart.

## Setup

1. Open a terminal in `frontend/frontend`
2. Install dependencies:
   ```bash
   npm install
   ```
3. Start the development server:
   ```bash
   npm run dev
   ```

## Notes

- The frontend expects the backend API to run at `http://localhost:8000`.
- Use the upload card to select a chart image and save the extracted data to MySQL.
- The table and chart views refresh automatically after a successful upload.
