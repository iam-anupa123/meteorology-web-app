import { useState, useEffect } from "react";
import axios from "axios";

import { Line } from "react-chartjs-2";

import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Tooltip,
  Legend
} from "chart.js";

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Tooltip,
  Legend
);

function App() {
  const [file, setFile] = useState(null);
  const [records, setRecords] = useState([]);
  const [metadata, setMetadata] = useState({});
  const [loading, setLoading] = useState(false);

  const loadData = async () => {
    try {
      const res = await axios.get(
        "http://localhost:8000/api/charts/data"
      );

      setRecords(res.data);
    } catch (err) {
      console.log(err);
    }
  };

  const loadMetadata = async () => {
    try {
      const res = await axios.get(
        "http://localhost:8000/api/charts/metadata"
      );

      setMetadata(res.data);
    } catch {
      setMetadata({});
    }
  };

  useEffect(() => {
    loadData();
    loadMetadata();
  }, []);

  const upload = async () => {
    if (!file) {
      alert("Please select an image.");
      return;
    }

    setLoading(true);

    const formData = new FormData();
    formData.append("file", file);

    try {
      await axios.post(
        "http://localhost:8000/api/charts/upload",
        formData
      );

      await loadData();
      await loadMetadata();

      alert("Chart uploaded successfully.");
    } catch (err) {
      alert(
        err.response?.data?.detail ||
          "Upload failed."
      );
    }

    setLoading(false);
  };

  const chartData = {
    labels: records.map(
      (r) => r.time_stamp
    ),
    datasets: [
      {
        label: "Rainfall",
        data: records.map(
          (r) => r.value
        ),
        borderColor: "#c0c0c0",
        backgroundColor: "#c0c0c0",
        borderWidth: 1,
        pointRadius: 2,
        pointHoverRadius: 4,
        tension: 0,
        fill: false
      }
    ]
  };

  const chartOptions = {
    responsive: true,
    plugins: {
      legend: {
        display: true
      }
    },
    scales: {
      y: {
        beginAtZero: true,
        title: {
          display: true,
          text: "Rainfall Value"
        }
      },
      x: {
        title: {
          display: true,
          text: "Time"
        }
      }
    }
  };

  return (
    <div className="container">
      <h1>
        Meteorology Web Application
      </h1>

      <div className="upload-box">
        <input
          type="file"
          accept="image/*"
          onChange={(e) =>
            setFile(
              e.target.files[0]
            )
          }
        />

        <button
          onClick={upload}
          disabled={loading}
        >
          {loading
            ? "Uploading..."
            : "Upload"}
        </button>
      </div>

      <div className="metadata-box">
        <h2>Chart Metadata</h2>

        <div className="metadata-grid">
          <p>
            <strong>
              Station Name:
            </strong>{" "}
            {metadata.station_name ||
              "N/A"}
          </p>

          <p>
            <strong>
              Chart Set At:
            </strong>{" "}
            {metadata.chart_set_at ||
              "N/A"}
          </p>

          <p>
            <strong>Set Date:</strong>{" "}
            {metadata.set_date || "N/A"}
          </p>

          <p>
            <strong>
              Chart Removed At:
            </strong>{" "}
            {metadata.chart_removed_at ||
              "N/A"}
          </p>

          <p>
            <strong>
              Removed Date:
            </strong>{" "}
            {metadata.removed_date ||
              "N/A"}
          </p>

          <p>
            <strong>Time On:</strong>{" "}
            {metadata.time_on || "N/A"}
          </p>

          <p>
            <strong>Time Off:</strong>{" "}
            {metadata.time_off || "N/A"}
          </p>

          <p>
            <strong>
              Duration of Rainfall:
            </strong>{" "}
            {metadata.duration_rainfall ||
              "N/A"}
          </p>
        </div>

        {metadata.raw_text && (
          <>
            <hr />

            <h3>Extracted Text Content (Top → Bottom)</h3>

            <div className="raw-text-box">
              <pre style={{ whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}>
                {metadata.raw_text}
              </pre>
            </div>
          </>
        )}
      </div>

      <div className="chart">
        <Line
          data={chartData}
          options={chartOptions}
        />
      </div>

      <h2>Extracted Data</h2>

      <table>
        <thead>
          <tr>
            <th>ID</th>
            <th>Time</th>
            <th>Value</th>
          </tr>
        </thead>

        <tbody>
          {records.length === 0 ? (
            <tr>
              <td
                colSpan="3"
                style={{
                  textAlign:
                    "center"
                }}
              >
                No data available
              </td>
            </tr>
          ) : (
            records.map((r) => (
              <tr key={r.id}>
                <td>{r.id}</td>
                <td>
                  {r.time_stamp}
                </td>
                <td>{r.value}</td>
              </tr>
            ))
          )}
        </tbody>
      </table>
    </div>
  );
}

export default App;