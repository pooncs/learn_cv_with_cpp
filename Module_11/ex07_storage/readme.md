# Exercise 07: Storage and Databases

## Goal
Understand the difference between various storage types (Relational, NoSQL, Time-Series) and when to use them in a Computer Vision system.

## Learning Objectives
1.  **Relational (SQL):** Structured data (User accounts, Camera configurations).
2.  **NoSQL (MongoDB):** Unstructured/JSON data (Detection metadata, Events).
3.  **Blob Storage (S3):** Large binary files (Images, Videos).
4.  **Time-Series (InfluxDB):** Metrics over time (Temperature, FPS).

## Practical Motivation
You don't store 4K video files in a MySQL database. It would crash. You store the *path* to the video in MySQL, and the *actual video* in Blob Storage (like AWS S3 or MinIO).

**Analogy:**
*   **SQL (Warehouse Inventory):** Strictly organized rows and columns. "Item ID 123 is on Shelf A."
*   **NoSQL (Documents):** A filing cabinet with varied folders. "Here is the report for Incident #5."
*   **Blob Storage (Self-Storage Unit):** Big empty rooms where you throw large furniture (videos) because they don't fit in the filing cabinet.

## Theory: Data Architecture for CV
1.  **Camera** captures frame.
2.  **App** runs inference -> gets Bounding Boxes.
3.  **Save Image:** To Disk/S3 -> Get Path `s3://bucket/img_123.jpg`.
4.  **Save Metadata:** To DB -> `{timestamp: 12:00, path: s3://..., detections: [car, person]}`.

## Step-by-Step Instructions

### Task 1: Design Schema (Conceptual)
1.  Open `schema.sql` (or `schema.json`).
2.  Design a table/document structure for storing:
    *   Camera ID
    *   Timestamp
    *   Image Path
    *   List of Detections (Class, Confidence, Box)

## Code Hints
```sql
-- Relational Approach
CREATE TABLE Detections (
    id INT PRIMARY KEY,
    camera_id INT,
    timestamp DATETIME,
    image_path VARCHAR(255),
    -- Detections might need a separate table or JSON column
    detection_data JSON
);
```

## Verification
Review your schema. Does it handle 1 million detections? Does it handle efficient querying by time range?
