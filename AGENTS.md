# Project rules

Build only this independent YOLO project. Do not import or copy legacy application code, environments, weights, databases, or logs. Source datasets are read-only. Derivatives belong under data/derived. Keep credentials and data out of Git. Never report generated fixtures, smoke runs, desktop measurements, or offline image evaluation as camera/laptop acceptance. Run contract tests after changes to geometry, tracking, event timing, or storage. All queues and frame buffers must be bounded. Use original-pixel xyxy coordinates and monotonic observation time.
