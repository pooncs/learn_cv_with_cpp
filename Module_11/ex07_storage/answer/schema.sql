-- 1. Cameras Table
-- Stores configuration for each camera stream
CREATE TABLE Cameras (
    camera_id INT PRIMARY KEY AUTO_INCREMENT,
    name VARCHAR(100) NOT NULL,
    rtsp_url VARCHAR(255) NOT NULL,
    location VARCHAR(100),
    resolution VARCHAR(20) -- e.g., "1920x1080"
);

-- 2. Frames Table
-- Stores metadata about processed frames. 
-- The actual image data is in Object Storage (S3), referenced by 'storage_path'.
CREATE TABLE Frames (
    frame_id BIGINT PRIMARY KEY AUTO_INCREMENT,
    camera_id INT,
    timestamp DATETIME(3) NOT NULL, -- Millisecond precision
    storage_path VARCHAR(255) NOT NULL, -- s3://bucket/date/cam_id/frame_123.jpg
    FOREIGN KEY (camera_id) REFERENCES Cameras(camera_id)
);

-- 3. Detections Table
-- Stores individual objects found in a frame.
-- Normalized for efficient querying (e.g., "Count all cars seen today").
CREATE TABLE Detections (
    detection_id BIGINT PRIMARY KEY AUTO_INCREMENT,
    frame_id BIGINT,
    class_name VARCHAR(50) NOT NULL, -- e.g., "person", "car"
    confidence FLOAT NOT NULL,
    bbox_x INT,
    bbox_y INT,
    bbox_w INT,
    bbox_h INT,
    FOREIGN KEY (frame_id) REFERENCES Frames(frame_id)
);

-- Indexing for Performance
CREATE INDEX idx_timestamp ON Frames(timestamp);
CREATE INDEX idx_class ON Detections(class_name);
