# Getting Started with Speedman ⚡

Speedman is designed to make high-speed speech (5×–6×) easy to understand rather than a chore.

---

## 🚀 3 Ways to Use Speedman

### 1. Windows Desktop Studio (Recommended)
Double-click **`Speedman.lnk`** on your Windows Desktop (or run `D:\Workspace\speedman\Speedman.bat`).
- Opens a dedicated app window at `http://127.0.0.1:8081/`.
- **Drag & drop** any audio file, or enter a workstation path like `D:\Audio\Speed\sample.wav`.
- Pick your speed ($1.0\times$ to $10.0\times$) and DSP preset (`fast` is default).
- Click **Process with Speedman** to scrub and listen to the result!
- When you close the app window, Speedman automatically shuts down after 60 seconds of idle time, freeing 100% of RAM.

### 2. Windows Explorer Right-Click (Instant 1-Click)
1. In Windows File Explorer, right-click any audio file (`.wav`, `.mp3`, `.m4a`, `.aac`, `.flac`, `.ogg`).
2. Select **`Send to`** $\rightarrow$ **`Speedman (5x Fast)`**.
3. A small background terminal compresses the file on workstation E-cores and saves `<filename>_5x_fast.wav` to `D:\Audio\Speed\`.

### 3. Blind A/B Intelligibility Evaluation
Want to test whether Speedman actually sounds clearer than your normal media player?
1. Open the Web Studio and click the **Blind A/B Evaluation** tab.
2. Drop your audio clip and click **Generate Blind Evaluation Set**.
3. Speedman renders 12 randomized versions across speeds (4×, 5×, 6×) and presets (`natural`, `fast`, `aggressive`, plus standard uniform stretch).
4. Listen to the blind tracks without confirmation bias.
5. Click **Reveal Mapping** to see which preset won!

---

## 🎛️ Choosing a Preset

- **`natural`**: Lightest pause compression, smooth transitions. Best for casual speech and audiobooks.
- **`fast` (Default)**: Balanced pause compression and consonant transient protection. The sweet spot for 5× listening on podcasts and lectures.
- **`aggressive`**: Maximum intelligible speed for rapid information intake at 6×.
- **`max`**: Extreme pause compression and peak limiting for ultra-fast audio scanning up to 8×–10×.
- **`uniform` (Control)**: Plain scalar speed-up (what standard media players do), used to compare effort against Speedman.

---

## 💻 CLI Usage

From WSL terminal or PowerShell:
```bash
# Check system health & dependencies
speedman doctor

# Compress a file
speedman yourfile.mp3 --speed 5 --preset fast

# Generate blind comparison folder
speedman compare yourfile.mp3 --speeds 4,5,6
```
