# 🚨 SENTINEL AI — AI-Powered Police Patrol & Crime Management System

> **DSA Project | Data Structures & Algorithms in Action**  
> A complete browser-based police operations system demonstrating 8 core DSA concepts with real working code.

[![Live Demo](https://img.shields.io/badge/🌐_Live_Demo-GitHub_Pages-blue?style=for-the-badge)](https://your-username.github.io/sentinel-ai-police)
[![DSA Concepts](https://img.shields.io/badge/DSA_Concepts-8-green?style=for-the-badge)](https://github.com/your-username/sentinel-ai-police)
[![No Install](https://img.shields.io/badge/No_Install_Needed-Open_in_Browser-orange?style=for-the-badge)](https://github.com/your-username/sentinel-ai-police)

---

## 📸 Screenshots

| Dashboard | Surveillance AI | DSA Algorithms |
|-----------|----------------|----------------|
| Crime stats, hotspot heatmap | Live webcam + object detection | Dijkstra, Graph, BST, Heap |

---

## 🧠 DSA Concepts Used

This project was built as a DSA demonstration. Every feature is backed by a real algorithm:

| # | Algorithm / Data Structure | Used For | Time Complexity |
|---|---------------------------|----------|----------------|
| 1 | **Graph (Adjacency List)** | City road network representation | O(V + E) space |
| 2 | **Dijkstra's Algorithm** | Shortest patrol route to crime scene | O(V²) |
| 3 | **BFS (Breadth First Search)** | Minimum-hop route finding | O(V + E) |
| 4 | **DFS (Depth First Search)** | Path exploration demonstration | O(V + E) |
| 5 | **Hash Table + Chaining** | O(1) case lookup by ID | O(1) average |
| 6 | **Min-Heap (Priority Queue)** | Emergency incident dispatch | O(log n) insert/extract |
| 7 | **Binary Search Tree (BST)** | Case search by year/date | O(log n) search |
| 8 | **Neural Network (COCO-SSD)** | Real-time weapon/person detection | 30 FPS |

---

## ✨ 20 Features

| Feature | Technology Used |
|---------|----------------|
| 🗺️ Dijkstra Shortest Path | Graph + Dijkstra Algorithm |
| 🗺️ A* / BFS / DFS Routing | Graph Traversal Algorithms |
| 🔍 O(1) Case Lookup | Hash Table with Chaining |
| 🚨 Emergency Dispatch | Min-Heap Priority Queue |
| 📅 Date-based Case Search | Binary Search Tree |
| 📷 Live AI Surveillance | TensorFlow.js COCO-SSD (80 classes) |
| 🔪 Weapon Detection | Neural Network + Threat Hash Map |
| 👤 Person Tracking | Real-time Bounding Box Detection |
| 📝 FIR Registration | Dynamic Array + Auto ID Generation |
| 🗺️ India Map + Routing | Leaflet.js + OpenStreetMap |
| 🔥 Crime Heatmap | Circle layers + risk scoring |
| 🚗 License Plate Lookup | String matching + NCIT database |
| 👮 Officer Dashboard | Data table + Chart.js radar |
| 📊 NCRB Crime Charts | Chart.js bar/line/doughnut |
| 🤖 Humanoid Robot UI | Animated SVG |
| 🚨 Police Siren | Web Audio API oscillator |
| 🔊 Automated Voice Alert | Speech Synthesis API |
| 📡 Crime Prediction | ML feature scoring engine |
| 🛰️ Satellite Feed Status | UI dashboard panel |
| 📈 Real-time Stats | Live counters and timers |

---

## 🚀 How to Run

### Option 1: Open Directly (Fastest)
```
1. Download index.html
2. Double-click to open in Chrome or Edge
3. Done! (Camera requires HTTPS or localhost — see note below)
```

### Option 2: Deploy on GitHub Pages (Recommended for class)
```
1. Create a GitHub account at github.com
2. Click + → New repository → Name it "sentinel-ai-police"
3. Make it Public → Create repository
4. Upload index.html → Commit changes
5. Go to Settings → Pages → Deploy from main branch
6. Wait 2 minutes → Your site is live at:
   https://[your-username].github.io/sentinel-ai-police
```

> ⚠️ **Camera Note:** `getUserMedia` (webcam access) requires HTTPS or localhost.  
> GitHub Pages automatically provides HTTPS — so your live site will have working cameras!

---

## 📁 File Structure

```
sentinel-ai-police/
│
├── index.html          ← THE ENTIRE PROJECT (single file!)
└── README.md           ← This file
```

Yes, the **entire project is just one HTML file**. No npm install, no build step, no server. Just open and run.

---

## 🔬 Algorithm Details

### 1. Graph Representation
```javascript
// 7 Delhi locations as nodes
const GRAPH_NODES = [
  { id: 0, label: 'CP',  city: 'Connaught Place' },
  { id: 1, label: 'KB',  city: 'Karol Bagh'      },
  // ...
];

// Roads between locations with distances (weights)
const GRAPH_EDGES = [
  { u: 0, v: 1, w: 4 }, // CP -- KB, 4 km
  { u: 0, v: 2, w: 6 }, // CP -- DB, 6 km
  // ...
];
```

### 2. Dijkstra's Algorithm
```javascript
function runDijkstra() {
  const dist = new Array(n).fill(Infinity);
  dist[0] = 0; // Start node distance = 0

  for (let i = 0; i < n; i++) {
    let u = pickSmallest(dist, visited); // Greedy choice
    visited.add(u);

    adj[u].forEach(({ v, w }) => {
      if (dist[u] + w < dist[v]) {
        dist[v] = dist[u] + w; // Relax edge
        prev[v] = u;           // Track path
      }
    });
  }
}
// Time: O(V²)  |  Space: O(V)
```

### 3. Hash Table with Chaining
```javascript
const TABLE_SIZE = 17; // Prime number

function hashFn(key) {
  let hash = 0;
  for (let ch of key)
    hash = (hash * 31 + ch.charCodeAt(0)) % TABLE_SIZE;
  return hash;
}

// insert('IB-2024-0891', caseData) → bucket[3]
// lookup('IB-2024-0891') → O(1) instant!
```

### 4. Min-Heap (Priority Queue)
```javascript
function heapInsert(item) {
  heap.push(item);
  let i = heap.length - 1;
  while (i > 0) {
    const parent = Math.floor((i - 1) / 2);
    if (heap[parent].priority <= heap[i].priority) break;
    [heap[parent], heap[i]] = [heap[i], heap[parent]]; // Swap
    i = parent;
  }
}
// Armed Robbery (priority 1) always dispatched before Noise Complaint (priority 9)
```

### 5. Binary Search Tree
```javascript
//        2022
//       /    \
//    2020    2024
//    /  \   /  \
//  2019 2021 2023 2025

function bstSearch(node, target) {
  if (!node) return null;
  if (node.val === target) return node; // Found!
  if (target < node.val) return bstSearch(node.left, target);
  return bstSearch(node.right, target);
}
// Find year 2024: only 2 comparisons instead of scanning all records!
```

### 6. Real-Time AI Surveillance
```javascript
// Load COCO-SSD neural network (80 object classes, ~5MB)
const model = await cocoSsd.load({ base: 'mobilenet_v2' });

// Detect objects in live webcam frame
const predictions = await model.detect(videoElement);
// Returns: [{ class: 'knife', score: 0.94, bbox: [x, y, w, h] }, ...]

// O(1) threat classification via hash map lookup
const threat = SV_THREATS['knife']; // → { level: 'DANGER', color: '#ff2233' }
```

---

## 🛠️ Tech Stack

| Technology | Purpose |
|------------|---------|
| **Vanilla JavaScript (ES6+)** | All DSA implementations |
| **TensorFlow.js v4.15** | ML inference in browser |
| **COCO-SSD MobileNetV2** | 80-class object detection |
| **Leaflet.js v1.9.4** | India map + routing |
| **Chart.js** | Crime statistics visualization |
| **HTML5 Canvas** | Graph drawing, camera overlay |
| **Web Audio API** | Police siren synthesis |
| **Speech Synthesis API** | Automated voice messages |

---

## 🎓 For DSA Students — Key Learning Points

1. **Why Hash Table over Array?**  
   Array search = O(n) = check every case. Hash table = O(1) = go directly to the right bucket. For 10,000 cases: array = 10,000 comparisons, hash table = 1 comparison.

2. **Why Dijkstra over BFS for routing?**  
   BFS finds fewest hops (turns), Dijkstra finds shortest distance. If roads have different lengths, BFS gives wrong answer, Dijkstra gives correct shortest path.

3. **Why Min-Heap for dispatch?**  
   Sorted array insert = O(n) (shift all elements). Heap insert = O(log n). For 1000 incidents: sorted array = 1000 operations, heap = 10 operations.

4. **Why BST for date search?**  
   Linear scan for year 2024 in 1000 records = 1000 checks. BST search = log₂(1000) ≈ 10 checks. BST is 100x faster!

---

## 🏆 Project Highlights

- ✅ **Real AI** — actual neural network running in your browser, not simulation
- ✅ **Single file** — entire project is one HTML file, no setup needed
- ✅ **India-specific** — uses NCRB data, Indian cities, IPC sections, FIR system
- ✅ **Works offline** — algorithms run locally; only map tiles need internet
- ✅ **Presentation-ready** — police siren, voice alerts, animated dashboard

---

## 🔴 Live Demo Instructions

When presenting to class:

1. **Dashboard** → Show crime stats, siren, voice message
2. **Map & Routing** → Run Dijkstra, show shortest path highlighted
3. **DSA Viewer** → Show Graph, click Dijkstra, show path animation
4. **DSA Viewer** → Show Hash Table, lookup a case ID
5. **DSA Viewer** → Show Min-Heap, add incidents, dispatch highest priority
6. **Surveillance** → Click START CAMERA → show knife/scissors → RED alert!
7. **Surveillance** → Show book/phone → correct classification

---

## 📄 License

This project was created as a DSA academic project. Free to use for educational purposes.

---

*Built with ❤️ for DSA class presentation | SENTINEL AI — Protecting with Algorithms*
