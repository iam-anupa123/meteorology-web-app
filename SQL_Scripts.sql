USE sys;
CREATE TABLE IF NOT EXISTS rainfall_records (
    id INT AUTO_INCREMENT PRIMARY KEY,
    time_stamp VARCHAR(10),
    value DOUBLE
);

CREATE TABLE IF NOT EXISTS uploaded_images (
    id INT AUTO_INCREMENT PRIMARY KEY,
    image_hash VARCHAR(64) UNIQUE,
    uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS chart_metadata (
    id INT AUTO_INCREMENT PRIMARY KEY,
    station_name VARCHAR(255),
    chart_set_at VARCHAR(255),
    set_date VARCHAR(255),
    chart_removed_at VARCHAR(255),
    removed_date VARCHAR(255),
    time_on VARCHAR(255),
    time_off VARCHAR(255),
    duration_rainfall VARCHAR(255),
    raw_text LONGTEXT
);