# Meteorology Web Application

## Overview

The Meteorology Web Application is a full-stack web application that allows users to upload meteorological chart images, extract rainfall data from the chart, store the extracted data in a MySQL database, and visualize the data in both tabular and graphical formats.

The application uses computer vision techniques to identify the rainfall trace on the chart image and convert it into structured time-series data.

---

## Features

* Upload meteorological chart images.
* Extract rainfall values from chart images using OpenCV.
* Convert extracted chart information into structured data (Time, Value).
* Store extracted data in a MySQL database.
* Display extracted data in a table.
* Generate a line chart from stored data.
* REST API endpoints for uploading and retrieving chart data.
* Duplicate image upload detection.
* Graceful handling of invalid images and extraction failures.

---

## Technology Stack

### Frontend

* React
* Vite
* Axios
* Chart.js
* React ChartJS 2

### Backend

* Python
* FastAPI
* OpenCV
* NumPy

### Database

* MySQL

---

## Project Structure

```text
Meteorology-Web-App/
│
├── backend/
│   ├── main.py
│   ├── requirements.txt
│
├── frontend/
│   ├── src/
│   │   ├── App.jsx
│   │   ├── App.css
│   │   └── main.jsx
│   ├── package.json
│
└── README.md
```

---

## Installation

### Prerequisites

Install the following software:

* Node.js 20 or later
* Python 3.12 or later
* MySQL Server

---

## Backend Setup

### 1. Navigate to the backend directory

```bash
cd backend
```

### 2. Install dependencies

```bash
pip install fastapi uvicorn opencv-python numpy mysql-connector-python python-multipart
```

### 3. Configure MySQL

Create the database tables.

```sql
CREATE TABLE rainfall_records (
    id INT AUTO_INCREMENT PRIMARY KEY,
    time_stamp VARCHAR(10),
    value DOUBLE
);

CREATE TABLE uploaded_images (
    id INT AUTO_INCREMENT PRIMARY KEY,
    image_hash VARCHAR(64) UNIQUE,
    uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

Update the database configuration inside `main.py`.

```python
DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "your_password",
    "database": "sys"
}
```

### 4. Start the backend server

```bash
uvicorn main:app --reload
```

The backend server will run on:

```text
http://localhost:8000
```

API documentation is available at:

```text
http://localhost:8000/docs
```

---

## Frontend Setup

### 1. Navigate to the frontend directory

```bash
cd frontend
```

### 2. Install dependencies

```bash
npm install
```

### 3. Start the React application

```bash
npm run dev
```

The frontend will run on:

```text
http://localhost:5173
```

---

## API Endpoints

### Upload Chart

```http
POST /api/charts/upload
```

Uploads a chart image, extracts rainfall data, and stores it in the database.

#### Request

```text
multipart/form-data
file: image file
```

#### Response

```json
{
  "message": "Chart uploaded successfully",
  "records": 25
}
```

---

### Retrieve Chart Data

```http
GET /api/charts/data
```

Returns all stored rainfall records.

#### Response

```json
[
  {
    "id": 1,
    "time_stamp": "08:30",
    "value": 0
  },
  {
    "id": 2,
    "time_stamp": "09:00",
    "value": 1.2
  }
]
```

---

## Workflow

1. User uploads a meteorological chart image.
2. Backend validates the image.
3. OpenCV extracts the rainfall trace.
4. Extracted values are converted into time-series data.
5. Data is stored in MySQL.
6. React frontend retrieves data using REST APIs.
7. Data is displayed in a table and line chart.

---

## Error Handling

The application handles the following scenarios:

* Invalid image uploads.
* Corrupted image files.
* Images without a detectable rainfall trace.
* Duplicate image uploads.
* Database connection failures.
* Extraction failures.

Appropriate HTTP status codes and error messages are returned to the user.

---

## Sample Database Output

| ID | Time  | Value |
| -- | ----- | ----- |
| 1  | 08:30 | 0     |
| 2  | 23:00 | 5     |
| 3  | 00:00 | 7     |

---

## Future Improvements

* OCR-based extraction of axis labels.
* Support for multiple chart formats.
* User authentication.
* Export extracted data to CSV or Excel.
* Historical chart storage and comparison.

---

## Author

Developed as part of the Meteorology Web Application Assessment project.
