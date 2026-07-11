# Schwarzschild Black Hole Relativistic Telemetry Deck (Hackathon Edition)

An interactive, cinematic, and high-performance real-time physics simulation of a Schwarzschild black hole with a granular accretion disk. This project utilizes a cutting-edge hybrid GPU-accelerated architecture, combining **Taichi Lang** for massive parallel physics computation and **ModernGL (GLSL)** for high-fidelity relativistic rendering.

Designed as a futuristic spacecraft navigation panel and diagnostic telemetry deck, the simulator visualizes complex general relativity phenomena while maintaining a smooth, hardware-accelerated 60+ FPS on dedicated graphics hardware.

<img width="1427" height="1106" alt="image" src="https://github.com/user-attachments/assets/8ceb3d1c-adf6-4139-9147-a8c5a6bfabc5" />

---

## Applied Physical & Rendering Models

1. **Equatorial Accretion Disk (Particle-Cloud Representation)**:
   The accretion disk is modeled as a flat particle cloud situated at the $z=0$ equatorial plane. This ensures that individual particles from the Taichi simulation are represented as sharp, glowing points and concentric arcs of light (just like dust in a physical accretion disk) rather than vertical columns or hairy structures, and preserves a mathematically sharp, smooth, spherical event horizon shadow.

2. **Planckian Blackbody Radiation Mapping**:
   The disk local temperature is modeled using the classical **Shakura-Sunyaev profile**:
   $$T(r) = T_{\text{inner}} \cdot \left(\frac{r_{\text{in}}}{r}\right)^{0.75}$$
   The color temperature observed by the camera is relativistically shifted by the local redshift factor $g$:
   $$T_{\text{obs}}(r) = g \cdot T(r)$$
   This temperature is mapped to physical RGB values using a Planckian blackbody locus function. Relativistic Doppler beaming and gravitational redshift are evaluated in real-time, automatically blueshifting and brightening gas moving towards the observer (making the left side white-hot and brilliant) and redshifting gas moving away (making the right side dark-red/orange and dimmer).

3. **Lensed Procedural Starfield (Anti-Aliased)**:
   In addition to a high-resolution skybox texture, the fragment shader computes a sub-pixel lensed starfield. Stars are mapped to escaping null geodesic directions $\vec{k}_{\infty}$ and anti-aliased relative to screen pixel dimensions to prevent aliasing, shimmering, or pixelation when stretched. Near the event horizon, starlight stretches into highly defined lensed filaments, Einstein rings, and arcs.

4. **Circular Orbits in Schwarzschild Spacetime**:
   Accretion disk particles are initialized and stepped in Taichi using General Relativistic circular geodesic velocity relative to a local static observer:
   $$v_\phi = \sqrt{\frac{M}{r - 2M}}$$
   At the ISCO ($r = 6M$), the velocity is exactly $0.5c$.

5. **First-Order Post-Newtonian (1PN) Einstein Correction**:
   Particle motion is integrated under the relativistic Schwarzschild precession acceleration formula:
   $$\mathbf{a} = -\frac{GM}{r^2} \mathbf{\hat{r}} \left( 1 + \frac{3 L^2}{c^2 r^2} \right)$$
   Viscosity angular momentum loss causes particles to slowly spiral inward until they hit the ISCO boundary and fall into the horizon.

---

## Physical Approximations & Limitations

To achieve real-time rendering performance (60+ FPS on modern GPUs) and allow for interactive stylistic control, several physical approximations were made compared to a rigorous scientific simulation:

1. **Schwarzschild vs. Kerr Spacetime**: The raymarching and particle orbital dynamics assume a non-rotating (Schwarzschild) black hole metric. Real astrophysical black holes typically have angular momentum (Kerr metric), which introduces frame-dragging (the Lense-Thirring effect) and causes the event horizon and photon sphere to become asymmetric.
2. **1PN Particle Dynamics**: The accretion disk particles in Taichi are integrated using a First-Order Post-Newtonian (1PN) approximation for Schwarzschild precession, rather than fully solving the nonlinear timelike geodesic equations. Viscosity is modeled via a simple linear Newtonian drag rather than magnetorotational instability (MRI).
3. **Semi-Implicit Euler Integration**: Numerical integration of the particle orbits uses a semi-implicit Euler method. While this is sufficient for visual stability in real-time applications, it is not symplectic and thus does not strictly conserve orbital energy over very long timescales like high-order Runge-Kutta methods would.
4. **2D Equatorial Thin Disk**: The accretion disk is represented as an infinitely thin plane at $z=0$. Real accretion flows possess 3D volumetric structure (e.g., pressure gradients, flaring, or a hot corona) governed by General Relativistic Magnetohydrodynamics (GRMHD) equations.
5. **Local Static Observer Beaming**: Doppler beaming and gravitational redshift assume the camera is a static observer at infinity, and velocities are computed in the local static frame rather than using a fully covariant radiative transport equation along the geodesic.

---

## Tech Stack & Architecture

The project eliminates CPU-GPU bottlenecks by implementing a direct data stream architecture entirely inside the VRAM video memory layout:

* **Taichi Lang:** Manages the parallel physics backend loop, resolving timelike geodesics and tracking the positions and velocities of over 80,000 active particles via Semi-Implicit Euler Integration.
* **ModernGL & GLSL:** Executes backward raymarching from the camera's viewport, processing absolute ray termination at the event horizon ($r \le r_s$), strict disk opacity depth occlusion over the background cosmos, and custom bilinear texture filtering for the skybox.
* **Dear PyGui:** Powers the fully responsive UI overlay (HUD), rendering an intuitive spaceship control deck that adjusts dynamically to fullscreen mode without distorting the simulation rendering aspect ratio.

---

## Spaceship Panel Controls

* **Physics Manipulators:** Real-time interactive sliders to alter the Black Hole Mass ($M$), Disk Rotation Speed, and Accretion Drag (friction/viscosity).
* **Optical & Aesthetic Tweaks:** Dynamic adjustments for Doppler Beaming strength, global disk glow intensity, and custom dual-temperature color palette pickers. Toggle between physical Blackbody mode and custom Artistic presets.
* **Navigation Deck:** Left mouse drag for smooth orbital camera rotation around the singularity, mouse scroll wheel for distance range adjustments, and an interactive **Navigation Sensitivity** slider to calibrate rotation speeds.
* **Time-Control Multimedia Deck:** Integrated video-player style playback control panel supporting real-time physics pausing (`||`), fast-forward speed scaling (`>> 2x/4x`), and fully inverted runtime integration (`<< REW`) for smoothly retracing particle paths in reverse.

---

## Setup & Running the Simulation

Ensure you have a GPU with OpenGL 3.3+ support (ModernGL) and a Vulkan/CUDA compatible GPU for Taichi (if no GPU is available, Taichi will automatically fall back to CPU mode).

### Installation

Install the required Python packages:
```bash
pip install -r requirements.txt
```

### Run Simulation

Execute the main program:
```bash
python blackhole_sim.py
```

Alternatively, to clone and run the simulation safely with isolated dependency management, utilizing the `uv` toolchain is highly recommended:
```bash
uv run --python 3.12 --with taichi --with moderngl --with dearpygui --with numpy blackhole_sim.py
```
