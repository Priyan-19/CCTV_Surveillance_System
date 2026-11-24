# Phase 1 Performance Benchmark Results

| Clip | Model | ImgSz | Conf | Inference FPS | Throughput FPS | Latency (ms) | VRAM Alloc | VRAM Res | Max People | Max Bags |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| sample_walking_01.mp4 | yolo26s.pt | 640 | 0.25 | **19.39** | 19.09 | 52.26 ms | 131.36 MB | 154.0 MB | 2 | 0 |
| sample_walking_01.mp4 | yolo26s.pt | 640 | 0.4 | **31.31** | 30.58 | 32.59 ms | 157.86 MB | 172.0 MB | 2 | 0 |
| sample_walking_01.mp4 | yolo26s.pt | 960 | 0.25 | **26.77** | 26.14 | 38.15 ms | 236.6 MB | 288.0 MB | 2 | 0 |
| sample_walking_01.mp4 | yolo26s.pt | 960 | 0.4 | **28.85** | 28.09 | 35.48 ms | 273.65 MB | 330.0 MB | 2 | 0 |
| sample_walking_01.mp4 | yolo11s.pt | 640 | 0.25 | **34.23** | 33.37 | 29.87 ms | 268.71 MB | 348.0 MB | 1 | 0 |
| sample_walking_01.mp4 | yolo11s.pt | 640 | 0.4 | **34.35** | 33.49 | 29.75 ms | 237.99 MB | 362.0 MB | 1 | 0 |
| sample_walking_01.mp4 | yolo11s.pt | 960 | 0.25 | **32.35** | 31.58 | 31.57 ms | 199.9 MB | 362.0 MB | 1 | 0 |
| sample_walking_01.mp4 | yolo11s.pt | 960 | 0.4 | **33.45** | 32.57 | 30.6 ms | 235.89 MB | 362.0 MB | 1 | 0 |
| sample_abandoned_bag_01.mp4 | yolo26s.pt | 640 | 0.25 | **30.58** | 29.88 | 33.36 ms | 233.92 MB | 362.0 MB | 1 | 0 |
| sample_abandoned_bag_01.mp4 | yolo26s.pt | 640 | 0.4 | **32.07** | 30.81 | 31.81 ms | 270.77 MB | 362.0 MB | 1 | 0 |
| sample_abandoned_bag_01.mp4 | yolo26s.pt | 960 | 0.25 | **27.75** | 27.14 | 36.74 ms | 198.48 MB | 362.0 MB | 2 | 0 |
| sample_abandoned_bag_01.mp4 | yolo26s.pt | 960 | 0.4 | **30.54** | 29.76 | 33.5 ms | 235.72 MB | 362.0 MB | 2 | 0 |
| sample_abandoned_bag_01.mp4 | yolo11s.pt | 640 | 0.25 | **34.43** | 33.56 | 29.69 ms | 230.34 MB | 362.0 MB | 1 | 0 |
| sample_abandoned_bag_01.mp4 | yolo11s.pt | 640 | 0.4 | **33.65** | 32.79 | 30.39 ms | 267.58 MB | 362.0 MB | 1 | 0 |
| sample_abandoned_bag_01.mp4 | yolo11s.pt | 960 | 0.25 | **30.48** | 29.74 | 33.52 ms | 199.9 MB | 362.0 MB | 1 | 0 |
| sample_abandoned_bag_01.mp4 | yolo11s.pt | 960 | 0.4 | **31.99** | 31.09 | 32.05 ms | 236.26 MB | 362.0 MB | 1 | 0 |
