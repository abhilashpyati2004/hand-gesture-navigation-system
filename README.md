# Hand Gesture Controlled Navigation System 🚀

An AI-powered computer vision application that allows you to control your system's navigation, volume, brightness, and more using intuitive hand gestures.

## 🌟 Overview
This project bridges the gap between human interaction and digital interfaces. Using a standard webcam, it detects hand landmarks in real-time and translates specific gestures into system commands, enabling a "touchless" computing experience.

## ✨ Key Features
- **Real-time Hand Tracking:** Uses Google's MediaPipe for high-performance 21-point landmark detection.
- **Dynamic Mouse Control:** Move the cursor, left-click, right-click, and drag with high precision.
- **Media & System Control:**
    - Adjust **System Volume** and **Screen Brightness** via intuitive hand rotations.
    - **Scrolling:** Smooth vertical scrolling using finger movements.
    - **Browser Navigation:** Swipe gestures for moving back and forward in your browser.
- **Advanced Gestures:**
    - **Pinch to Zoom:** Intuitive zooming capabilities.
    - **Screenshot:** Capture your screen with a specific hand pattern.
- **On-Screen Display (HUD):** Visual feedback (Heads-Up Display) directly on the video feed to show current mode and actions.

## 🛠️ Tech Stack
- **Language:** Python 3.x
- **Computer Vision:** OpenCV
- **AI/ML:** MediaPipe (Hand Landmarking)
- **Automation:** 
    - `PyAutoGUI` & `pynput` (Input simulation)
    - `screen-brightness-control` (Display adjustment)
    - `pycaw` (Windows Audio control)
- **Data Handling:** NumPy & Deque for smooth signal processing.

## 🖐️ Gesture Reference
| Action | Gesture |
| :--- | :--- |
| **Move Cursor** | Point index finger |
| **Left Click** | Index and Middle finger together (Pinch/Tap) |
| **Right Click** | Specific finger configuration |
| **Scroll** | Vertical movement with specific fingers raised |
| **Volume/Brightness** | Two-handed mode (adjust by rotating or distance) |
| **Screenshots** | Specific 5-finger pattern |

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- A webcam

### Installation
1. **Clone the repository:**
   ```bash
   git clone https://github.com/abhilashpyati2004/hand-gesture-navigation-system.git
   cd hand-gesture-navigation-system
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application:**
   ```bash
   python handgesturenav.py
   ```

---
*Created by [Abhilash Pyati](https://github.com/abhilashpyati2004) for personal project and placement showcase.*
