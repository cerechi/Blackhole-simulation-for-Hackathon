# Schwarzschild Black Hole Relativistic Telemetry Deck

An interactive, cinematic, and high-performance real-time physics simulation of a Schwarzschild black hole with a granular accretion disk. This project utilizes a cutting-edge hybrid GPU-accelerated architecture, combining **Taichi Lang** for massive parallel physics computation and **ModernGL (GLSL)** for high-fidelity relativistic rendering.

Designed as a futuristic spacecraft navigation panel and diagnostic telemetry deck, the simulator visualizes complex general relativity phenomena while maintaining a smooth, hardware-accelerated 60 FPS on dedicated graphics hardware.

<img width="1427" height="1106" alt="image" src="https://github.com/user-attachments/assets/8ceb3d1c-adf6-4139-9147-a8c5a6bfabc5" />

---

## Visualized Physics & Emergent Phenomena

* **Gravitational Lensing (Null Geodesics):** Real-time analytic approximation of photon path bending under severe spacetime curvature, warping background stars into perfect, continuous, non-pixelated **Einstein Rings**.
* **Relativistic Doppler Beaming:** Matter streaming toward the observer's viewpoint undergoes immense irradiance modulation ($g^4$), resulting in a stark, realistic brightness asymmetry on the approaching side of the disk.
* **Gravitational Redshift & Doppler Shift:** Thermal emissions from the disk are dynamically shifted in color wavelength based on local relativistic orbital velocity and gravitational proximity to the event horizon.
* **Emergent Fluid Dynamics (Shear Stress):** The accretion disk undergoes physical differential Keplerian rotation, naturally breaking a smooth, homogeneous particle ring over time into gorgeous, turbulent gaseous filaments and density waves.
* **Dynamic Boundary Protection:** Strictly enforces the Innermost Stable Circular Orbit (**ISCO** at $3.0 \cdot r_s$) across both the particle physics runtime and raymarching shader checks, preventing geometric feedback artifacts during real-time mass updates.

---

## Tech Stack & Architecture

The project eliminates CPU-GPU bottlenecks by implementing a direct data stream architecture entirely inside the VRAM video memory layout:

* **Taichi Lang:** Manages the parallel physics backend loop, resolving timelike geodesics and tracking the positions and velocities of over 60,000 active particles via Semi-Implicit Euler Integration.
* **ModernGL & GLSL:** Executes backward raymarching from the camera's viewport, processing absolute ray termination at the event horizon ($r \le r_s$), strict disk opacity depth occlusion over the background cosmos, and custom bilinear texture filtering for the skybox.
* **Dear PyGui:** Powers the fully responsive UI overlay (HUD), rendering an intuitive spaceship control deck that adjusts dynamically to fullscreen mode without distorting the simulation rendering aspect ratio.

---

## Spaceship Panel Controls

* **Physics Manipulators:** Real-time interactive sliders to alter the Black Hole Mass ($M$), Disk Keplerian Rotation Speed, and Accretion Drag (friction/viscosity).
* **Optical & Aesthetic Tweaks:** Dynamic adjustments for Doppler Beaming strength, global disk glow intensity, and custom dual-temperature color palette pickers for the inner and outer boundaries of the gas.
* **Navigation Deck:** Left mouse drag for smooth orbital camera rotation around the singularity, mouse scroll wheel for distance range adjustments, and an interactive **Navigation Sensitivity** slider to calibrate rotation speeds.
* **Time-Control Multimedia Deck:** Integrated video-player style playback control panel supporting real-time physics pausing (`||`) while keeping the 3D orbital camera fully active, fast-forward speed scaling (`>> 2x/4x`), and fully inverted runtime integration (`<< REW`) for smoothly retracing particle paths in reverse.

---

## Requirements & Execution

Due to Taichi compilation guardrails, this simulation requires **Python 3.10, 3.11, or 3.12** and a dedicated GPU graphics backend compatible with Vulkan, CUDA, or DirectX 12.

To clone and run the simulation safely with isolated dependency management, utilizing the `uv` toolchain is highly recommended:

```bash
# Run the simulation directly, installing all necessary packages automatically
uv run --python 3.12 --with taichi --with moderngl --with dearpygui --with numpy sim.py
