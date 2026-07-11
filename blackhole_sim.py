"""
=====================================================================
 SCHWARZSCHILD BLACK HOLE — RELATIVISTIC CINEMATIC SIMULATION
=====================================================================
Physics Backend:  Taichi (timelike disk geodesics, 1PN Einstein correction)
Rendering:        ModernGL (null geodesics / flat-disk raymarching in GLSL)
UI & Dashboard:   Dear PyGui (spaceship telemetry deck)

This file contains a general relativistic visualization of a Schwarzschild
black hole and its accretion disk. It features:
1) Flat accretion disk rendering at the z=0 equatorial plane to ensure a sharp,
   star-like particle cloud representation and a perfectly smooth horizon shadow.
2) Planckian blackbody radiation color mapping based on shifted local temperature
   (T_obs = g * T) with a toggle for custom artistic color presets.
3) A highly optimized, anti-aliased procedural lensed starfield that stretches
   into Einstein rings and arcs near the shadow boundary without pixelation.
4) 1PN Einstein orbital correction and General Relativistic circular Keplerian
   velocity fields for disk particles.
5) Dynamic controls for astrophysics parameters.

=====================================================================
 PHYSICAL APPROXIMATIONS & LIMITATIONS
=====================================================================
For real-time performance and stylistic control, the following physical 
approximations are employed:
1. Schwarzschild vs. Kerr Spacetime: The raymarching and orbit dynamics 
   assume a non-rotating (Schwarzschild) black hole. Astrophysical black 
   holes typically possess angular momentum (Kerr metric), which would 
   induce frame-dragging (Lense-Thirring effect) and an asymmetric horizon.
2. 1PN Particle Dynamics: The accretion disk particles in Taichi are 
   integrated using a First-Order Post-Newtonian (1PN) approximation for 
   Schwarzschild precession, rather than the full timelike geodesic equation.
   Viscosity is modeled via a simple linear Newtonian drag.
3. Semi-Implicit Euler Integration: Numerical integration of particle 
   orbits uses semi-implicit Euler rather than high-order Runge-Kutta, 
   which is sufficient for visual stability but not for strict long-term 
   orbital conservation.
4. 2D Equatorial Thin Disk: The accretion disk is represented as an infinitely 
   thin plane at z=0. Real accretion flows possess 3D volumetric structure 
   (e.g., pressure gradients, flaring, or corona) governed by GRMHD equations.
5. Local Static Observer Beaming: Doppler beaming and gravitational redshift 
   assume the camera is a static observer at infinity, and velocities are 
   computed in the local static frame rather than a fully covariant ray 
   transport equation.
=====================================================================
"""

import dearpygui.dearpygui as dpg
import moderngl
import numpy as np
import taichi as ti
import time

# ---------------------------------------------------------------------------
# Console Welcome & Telemetry Deck Startup
# ---------------------------------------------------------------------------
print("=====================================================================")
print("     SCHWARZSCHILD BLACK HOLE CINEMATIC RELATIVISTIC SIMULATION      ")
print("=====================================================================")
print("Physics Backend: Taichi (timelike geodesics, GR Keplerian velocity)")
print("Rendering Pipeline: ModernGL (Flat equatorial null geodesic raymarching)")
print("UI & Dashboard: Dear PyGui (Spaceship Telemetry Deck)")
print("\nInteractive Controls:")
print("  - [LEFT MOUSE DRAG] over viewport to rotate camera view")
print("  - [MOUSE SCROLL] over viewport to zoom in / out")
print("  - [SPACE BAR] to Pause / Resume particle simulation")
print("  - [R Key] to Reset accretion disk particles")
print("=====================================================================")


# ---------------------------------------------------------------------------
# Far-field Background Skybox (Procedural Milky Way Texture)
# ---------------------------------------------------------------------------
def generate_milky_way_texture(w=1024, h=512):
    y = np.linspace(-0.5, 0.5, h, dtype=np.float32)
    x = np.linspace(-0.5, 0.5, w, dtype=np.float32)
    X, Y = np.meshgrid(x, y)

    theta = Y * np.pi
    phi = X * 2.0 * np.pi

    # Galactic plane band
    galaxy_band = np.exp(-np.abs(theta) / 0.15)

    # Warm galactic bulge core
    bulge = np.exp(-(theta ** 2 + phi ** 2) / (2.0 * 0.25 ** 2)) * 2.5

    # Procedural dust lanes
    dust_pattern = 0.35 + 0.65 * np.sin(phi * 6.0 + theta * 12.0)
    dust_pattern = np.clip(dust_pattern, 0.1, 1.0)

    milky_way = np.zeros((h, w, 3), dtype=np.float32)
    milky_way[..., 0] = galaxy_band * 0.4 * dust_pattern + bulge * 1.0  # R
    milky_way[..., 1] = galaxy_band * 0.45 * dust_pattern + bulge * 0.8  # G
    milky_way[..., 2] = galaxy_band * 0.6 * dust_pattern + bulge * 0.6  # B

    # Seed stars in background texture
    star_mask = np.random.rand(h, w) > 0.9992
    star_intensity = np.random.rand(h, w) * star_mask
    for c in range(3):
        milky_way[..., c] += star_intensity

    return np.clip(milky_way, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Taichi Physics Engine Initialization
# ---------------------------------------------------------------------------
try:
    ti.init(arch=ti.gpu)
except Exception as e:
    print(f"Taichi GPU initialization failed: {e}. Falling back to CPU.")
    ti.init(arch=ti.cpu)

# Particle Physics Parameters
N_PARTICLES = 80000
G_const = 1.0
c_const = 1.0  # Geometric units (G = c = 1)
GRID_SIZE = 1024

# Taichi Fields
particle_pos = ti.Vector.field(3, dtype=ti.f32, shape=N_PARTICLES)
particle_vel = ti.Vector.field(3, dtype=ti.f32, shape=N_PARTICLES)
disk_grid = ti.Vector.field(4, dtype=ti.f32, shape=(GRID_SIZE, GRID_SIZE))

speed_factor_field = ti.field(dtype=ti.f32, shape=())
accretion_rate_field = ti.field(dtype=ti.f32, shape=())

speed_factor_field[None] = 1.0
accretion_rate_field[None] = 0.05


@ti.kernel
def init_particles(M: ti.f32):
    r_eh = 2.0 * G_const * M / (c_const * c_const)
    r_isco = 3.0 * r_eh  # Innermost Stable Circular Orbit = 6M = 3 r_s
    r_inner = r_isco * 1.35
    for p in range(N_PARTICLES):
        u = ti.random()
        r = r_inner + u * (17.5 - r_inner)
        theta = ti.random() * 2.0 * 3.14159265
        z_offset = (ti.random() - 0.5) * 0.12  # Fine disk height offset

        particle_pos[p] = ti.Vector([r * ti.cos(theta), r * ti.sin(theta), z_offset])

        # Circular Keplerian velocity in Schwarzschild metric (GR): v = sqrt(M / (r - 2M))
        # With G = c = 1, M = r_eh / 2.
        v_mag = ti.sqrt((r_eh * 0.5) / (r - r_eh + 1e-4))
        particle_vel[p] = ti.Vector(
            [-v_mag * ti.sin(theta), v_mag * ti.cos(theta), (ti.random() - 0.5) * 0.02]
        )


@ti.kernel
def step_simulation(dt: ti.f32, M: ti.f32):
    r_eh = 2.0 * G_const * M / (c_const * c_const)
    r_isco = 3.0 * r_eh

    speed_factor = speed_factor_field[None]
    accretion_rate = accretion_rate_field[None]

    for p in range(N_PARTICLES):
        r_vec = particle_pos[p]
        r2 = r_vec.norm_sqr() + 1e-4
        r = ti.sqrt(r2)

        v_vec = particle_vel[p]
        L_vec = r_vec.cross(v_vec)
        L2 = L_vec.norm_sqr()

        # Relativistic First-Order Post-Newtonian (1PN) Acceleration (Schwarzschild precession)
        # a = -GM/r^2 * r_hat * (1 + 3 L^2 / (c^2 r^2))
        inv_r3 = 1.0 / (r2 * r + 1e-4)
        acc = - (r_eh * 0.5) * inv_r3 * r_vec * (1.0 + 3.0 * L2 / (c_const * c_const * r2 + 1e-4))

        # Viscosity angular momentum drag
        drag = -accretion_rate * v_vec

        # Semi-implicit Euler integration
        new_v = v_vec + (acc + drag) * dt
        new_pos = r_vec + new_v * speed_factor * dt
        r_new = new_pos.norm()

        # If particles fall inside inner edge or escape outer boundary, respawn
        r_inner = r_isco * 1.35
        if r_new < r_inner or r_new > 22.0:
            r_spawn = ti.random() * (17.5 - r_inner) + r_inner
            theta = ti.random() * 2.0 * 3.14159265
            z_val = (ti.random() - 0.5) * 0.12
            new_pos = ti.Vector([r_spawn * ti.cos(theta), r_spawn * ti.sin(theta), z_val])

            v_mag = speed_factor * ti.sqrt((r_eh * 0.5) / (r_spawn - r_eh + 1e-4))
            new_v = ti.Vector(
                [-v_mag * ti.sin(theta), v_mag * ti.cos(theta), (ti.random() - 0.5) * 0.02]
            )

        particle_pos[p] = new_pos
        particle_vel[p] = new_v


@ti.kernel
def rasterize_particles(disk_max_r: ti.f32):
    # Clear grid
    for i, j in disk_grid:
        disk_grid[i, j] = ti.Vector([0.0, 0.0, 0.0, 0.0])

    speed_factor = speed_factor_field[None]
    for p in range(N_PARTICLES):
        pos = particle_pos[p]
        vel = particle_vel[p]

        r = pos.norm()
        if r < 20.0:
            x_norm = (pos[0] + disk_max_r) / (2.0 * disk_max_r)
            y_norm = (pos[1] + disk_max_r) / (2.0 * disk_max_r)

            if 0.0 <= x_norm <= 1.0 and 0.0 <= y_norm <= 1.0:
                center_x = x_norm * GRID_SIZE
                center_y = y_norm * GRID_SIZE

                ix = int(ti.floor(center_x))
                iy = int(ti.floor(center_y))

                # 3x3 Gaussian splatting
                for offset_x in range(-1, 2):
                    for offset_y in range(-1, 2):
                        gx = ix + offset_x
                        gy = iy + offset_y
                        if 0 <= gx < GRID_SIZE and 0 <= gy < GRID_SIZE:
                            dx = (gx + 0.5) - center_x
                            dy = (gy + 0.5) - center_y
                            dist2 = dx * dx + dy * dy
                            weight = ti.exp(-dist2 / 0.8)

                            disk_grid[gx, gy][0] += weight
                            disk_grid[gx, gy][1] += weight * vel[0] * speed_factor
                            disk_grid[gx, gy][2] += weight * vel[1] * speed_factor
                            disk_grid[gx, gy][3] += weight * vel[2] * speed_factor


init_particles(1.0)

# ---------------------------------------------------------------------------
# Viewport / OpenGL Context Setup
# ---------------------------------------------------------------------------
width, height = 1280, 800
dpg.create_context()
ctx = moderngl.create_context(standalone=True)

# Blend configuration for gaseous scattering
ctx.enable(moderngl.BLEND)
ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE)

color_tex = ctx.texture((width, height), 4, dtype='f4')
fbo = ctx.framebuffer(color_attachments=color_tex)

disk_tex = ctx.texture((GRID_SIZE, GRID_SIZE), 4, dtype='f4')
disk_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)

skybox_data = generate_milky_way_texture()
skybox_tex = ctx.texture((1024, 512), 3, dtype='f4')
skybox_tex.write(skybox_data.tobytes())
skybox_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)

# ---------------------------------------------------------------------------
# GLSL Shader Program Setup
# ---------------------------------------------------------------------------
prog = ctx.program(
    vertex_shader="""
    #version 330
    in vec2 in_vert;
    out vec2 uv;
    void main() {
        uv = in_vert;
        gl_Position = vec4(in_vert, 0.0, 1.0);
    }
    """,
    fragment_shader="""
    #version 330
    in vec2 uv;
    out vec4 frag_color;

    uniform vec3 camera_pos;
    uniform mat3 camera_rot;
    uniform float r_s;
    uniform float aspect;
    
    // UI control uniforms
    uniform float disk_intensity;
    uniform float doppler_strength;
    uniform float noise_scale;
    uniform float time;
    uniform vec3 disk_color_inner;
    uniform vec3 disk_color_outer;
    
    // Physics extension uniforms
    uniform bool u_physical_mode;
    uniform bool u_cinematic_mode;
    uniform float u_star_intensity;
    uniform float u_skybox_intensity;
    uniform float u_view_height;
    uniform float u_disk_glow_intensity;
    uniform float u_temperature;

    uniform sampler2D disk_tex;
    uniform sampler2D u_skybox;

    const int MAX_STEPS = 180;
    const float min_h = 0.02;
    const float max_h = 0.35;

    // Fast 2D/3D Hash functions
    vec2 hash22(vec2 p) {
        p = vec2(dot(p, vec2(127.1, 311.7)), dot(p, vec2(269.5, 183.3)));
        return fract(sin(p) * 43758.5453123);
    }
    
    vec3 hash33(vec3 p) {
        p = fract(p * vec3(443.8975, 397.2973, 491.1871));
        p += dot(p.xyz, p.yzx + 19.19);
        return fract(p.xxy * p.yzz);
    }

    // 2D Gradient Noise
    float noise2d(vec2 p) {
        vec2 i = floor(p);
        vec2 f = fract(p);
        vec2 u = f * f * (3.0 - 2.0 * f);
        return mix(mix(dot(hash22(i + vec2(0.0,0.0)), f - vec2(0.0,0.0)),
                       dot(hash22(i + vec2(1.0,0.0)), f - vec2(1.0,0.0)), u.x),
                   mix(dot(hash22(i + vec2(0.0,1.0)), f - vec2(0.0,1.0)),
                       dot(hash22(i + vec2(1.0,1.0)), f - vec2(1.0,1.0)), u.x), u.y) * 0.5 + 0.5;
    }

    // Fractional Brownian Motion for gas detail
    float fbm(vec2 p) {
        float v = 0.0;
        float a = 0.5;
        mat2 rot = mat2(cos(0.5), sin(0.5), -sin(0.5), cos(0.5));
        for (int i = 0; i < 3; ++i) {
            v += a * noise2d(p);
            p = rot * p * 2.1;
            a *= 0.5;
        }
        return v;
    }

    // High-quality, anti-aliased procedural star function (sub-pixel rendering)
    float star(vec3 ray_dir, float cell_density, float base_size, float pixel_angle) {
        vec3 p = ray_dir * cell_density;
        vec3 grid = floor(p);
        vec3 f = fract(p) - 0.5;
        
        vec3 rand = hash33(grid);
        vec3 offset = (rand - 0.5) * 0.7;
        
        vec3 star_pos = grid + 0.5 + offset;
        vec3 star_dir = normalize(star_pos);
        
        float angle_dist = acos(clamp(dot(ray_dir, star_dir), -1.0, 1.0));
        
        // Anti-aliasing threshold to avoid pixelation
        float star_size = base_size * (rand.z * 0.7 + 0.3);
        float render_size = max(star_size, pixel_angle * 0.6);
        
        float val = smoothstep(render_size, 0.0, angle_dist);
        val *= pow(star_size / render_size, 2.0); // Dim clamped star to preserve total energy
        
        return val * rand.y;
    }

    // Lensed procedural starfield generator (low density, high aesthetic quality)
    vec3 get_starfield(vec3 ray_dir, float pixel_angle) {
        float stars = 0.0;
        stars += star(ray_dir, 45.0, 0.0015, pixel_angle) * 1.5;
        stars += star(ray_dir, 20.0, 0.0025, pixel_angle) * 3.5;
        
        vec3 star_color = vec3(stars);
        vec3 rand = hash33(floor(ray_dir * 20.0));
        
        if (rand.x > 0.85) {
            star_color *= vec3(0.7, 0.9, 1.5); // Hot O-type star (Blue giant)
        } else if (rand.x < 0.15) {
            star_color *= vec3(1.5, 0.8, 0.5); // Cool M-type star (Red dwarf)
        } else if (rand.x < 0.35) {
            star_color *= vec3(1.3, 1.1, 0.8); // Warm G-type star (Yellow-orange)
        }
        
        return star_color * u_star_intensity;
    }

    // Planckian Blackbody Radiation color mapping (Kelvin to RGB fit)
    vec3 blackbody(float Temp) {
        float t = clamp(Temp / 100.0, 10.0, 400.0);
        float r, g, b;
        
        if (t <= 66.0) {
            r = 255.0;
            g = 99.4708025861 * log(t) - 161.1195681661;
            b = (t <= 19.0) ? 0.0 : (138.5177312231 * log(t - 10.0) - 305.0447927307);
        } else {
            r = 329.698727446 * pow(t - 60.0, -0.1332047592);
            g = 288.1221695283 * pow(t - 60.0, -0.0755148492);
            b = 255.0;
        }
        
        return clamp(vec3(r, g, b) / 255.0, 0.0, 1.0);
    }

    // Standard relativistic color shift (blueshift/redshift) for artistic colors
    vec3 shift_color(vec3 base_col, float g) {
        vec3 col = base_col;
        if (g > 1.0) {
            float factor = (g - 1.0) * doppler_strength;
            col = mix(col, vec3(0.4, 0.7, 1.5) * (col + vec3(0.1)), clamp(factor, 0.0, 1.0));
        } else {
            float factor = (1.0 - g) * doppler_strength;
            col = mix(col, vec3(1.5, 0.35, 0.12) * col, clamp(factor, 0.0, 1.0));
        }
        float beaming = pow(max(g, 1e-3), 4.0); // Bolometric beaming flux scaling
        return col * clamp(beaming, 0.01, 12.0);
    }

    // CINEMATIC GAS DISK: volumetric sampling is done per-step inside the main loop below.
    // (This function placeholder kept for reference; actual logic is in main())

    // Skybox spherical projection
    vec3 sample_skybox(vec3 r_dir) {
        float phi = atan(r_dir.z, r_dir.x);
        float theta = acos(clamp(r_dir.y, -0.999, 0.999));
        vec2 bg_uv = vec2((phi + 3.14159265) / (2.0 * 3.14159265), theta / 3.14159265);
        return texture(u_skybox, bg_uv).rgb;
    }

    void main() {
        vec3 dir = camera_rot * normalize(vec3(uv.x * aspect, uv.y, -1.5));
        vec3 pos = camera_pos;

        // Specific angular momentum L = pos x dir (conserved quantity)
        vec3 L_vec = cross(pos, dir);
        float L2 = dot(L_vec, L_vec);

        vec4 acc_color = vec4(0.0);
        float photon_glow = 0.0;
        float r_min = 1e9;

        vec3 prev_pos = pos;
        bool hit_horizon = false;
        
        float pixel_angle = 2.0 / (u_view_height * 1.5);

        // Raymarching null geodesics in Schwarzschild spacetime
        for (int step = 0; step < MAX_STEPS; step++) {
            prev_pos = pos;
            float r2 = dot(pos, pos) + 1e-4;
            float r = sqrt(r2);

            if (r <= r_s) {
                hit_horizon = true;
                break;
            }

            // Adaptive step size: smaller steps near event horizon
            float h = clamp(0.07 * (r - r_s), min_h, max_h);
            r_min = min(r_min, r);

            // Schwarzschild null geodesic acceleration
            float r5 = r2 * r2 * r;
            vec3 acc = -1.5 * r_s * L2 * pos / (r5 + 1e-4);

            dir += acc * h;
            dir = normalize(dir);
            pos += dir * h;

            // Accumulated photon sphere visual impact at r = 1.5 r_s
            float dist_to_ps = abs(r - 1.5 * r_s);
            photon_glow += exp(-dist_to_ps * 10.0) * h;

            // ---- CINEMATIC MODE: thin inner-rim volumetric glow (soft vertical extent) ----
            if (u_cinematic_mode) {
                float r_vol_g  = length(pos.xy);
                float r_isco_g = 3.0 * r_s;
                float disk_hg  = 0.028 * r_vol_g;   // very thin — only for edge glow
                // Expanded bounds to avoid hard clipping artifact, use smoothstep for fade
                if (abs(pos.z) < disk_hg && r_vol_g >= r_isco_g * 1.0 && r_vol_g <= 16.0) {
                    float vert_wg   = exp(-0.5 * pow(pos.z / (disk_hg * 0.45), 2.0));
                    float r_norm_g  = (r_vol_g - r_isco_g * 1.3) / (16.0 - r_isco_g * 1.3);
                    float inner_sm_g = smoothstep(r_isco_g * 1.15, r_isco_g * 1.35, r_vol_g);
                    float inner_glw = exp(-clamp(r_norm_g, 0.0, 1.0) * 14.0) * vert_wg * inner_sm_g;
                    
                    float g_grav_g  = sqrt(max(1.0 - r_s / (r_vol_g + 1e-4), 1e-4));
                    vec3 glow_col   = u_physical_mode
                        ? blackbody(u_temperature * 1.4 * g_grav_g)
                        : mix(disk_color_inner, vec3(1.0), 0.6); // bright version of inner color
                    
                    float opc_g  = 1.0 - exp(-inner_glw * h * 6.0);
                    acc_color.rgb += (1.0 - acc_color.a) * glow_col * disk_intensity * opc_g * 1.8;
                    acc_color.a   += (1.0 - acc_color.a) * opc_g;
                }
            }

            // Flat equatorial plane intersection at z = 0
            if (prev_pos.z * pos.z < 0.0) {
                float t = -prev_pos.z / (pos.z - prev_pos.z + 1e-5);
                vec3 intersect = mix(prev_pos, pos, t);
                float r_int = length(intersect.xy);
                float r_isco = 3.0 * r_s;
                // Particle mode texture only goes up to r=18.0, cinematic goes to 20.0
                float outer_limit = u_cinematic_mode ? 20.0 : 18.0;

                // Expand the hard cutoff bound to 1.0 to prevent jagged raymarching aliasing
                // The actual fade out is handled smoothly by inner_sm and density texture
                if (r_int >= r_isco * 1.0 && r_int <= outer_limit) {

                    // ---- CINEMATIC MODE: curl-fluid advected accretion disk ----
                    if (u_cinematic_mode) {
                        float phi_c     = atan(intersect.y, intersect.x);
                        float omega_c   = pow(r_s * 0.5 / max(r_int, r_s * 0.5 + 0.01), 1.5);
                        float phi_rot_c = phi_c - omega_c * time * 0.55;
                        float r_norm_c  = (r_int - r_isco * 1.3) / (20.0 - r_isco * 1.3);

                        // Smooth inner edge (fixes pixelation at ISCO ring)
                        float inner_sm  = smoothstep(0.0, 0.04, r_norm_c);
                        float outer_sm  = 1.0 - smoothstep(0.72, 1.0, r_norm_c);
                        // Softer inner rim (spread over larger range, no sharp spike)
                        float hot_rim_c = smoothstep(0.0, 0.02, r_norm_c)
                                        * exp(-r_norm_c * 6.5) * 1.8;
                        float disk_body = (1.0 - smoothstep(0.0, 0.88, r_norm_c)) * 0.85;

                        // Log-radial UV: natural coordinate for Keplerian shear
                        // In log(r) space, the shear rate is uniform across radii
                        float log_r_c  = log(r_int / (r_isco * 1.3) + 1.0);
                        vec2 base_uv   = vec2(log_r_c * 4.2, phi_rot_c * 6.5);

                        // ---- CURL FLUID SIMULATION ----
                        // Sample a scalar potential field, then take its 2D curl
                        // to get a divergence-free (incompressible) velocity field.
                        // This naturally creates swirling eddies without sinks/sources.
                        float eps = 0.035;
                        // Potential evaluated at offset points for finite difference
                        float pot_c  = fbm(base_uv * 0.55 + vec2(time * 0.045, time * 0.018));
                        float pot_px = fbm((base_uv + vec2(eps,  0.0)) * 0.55 + vec2(time * 0.045, time * 0.018));
                        float pot_py = fbm((base_uv + vec2(0.0,  eps)) * 0.55 + vec2(time * 0.045, time * 0.018));
                        // 2D curl: vel = (dP/dy, -dP/dx)
                        vec2 curl_vel = vec2(pot_py - pot_c, -(pot_px - pot_c)) / eps;

                        // Advect the base UV with the curl velocity
                        // Scale by r to keep inner advection tighter
                        vec2 adv_uv = base_uv + curl_vel * (0.55 + r_norm_c * 0.35);

                        // ---- MULTI-SCALE TURBULENT GAS ----
                        // Layer 1: large eddies (slow, ~MHD magneto-rotational instability)
                        float large = fbm(adv_uv * 1.1 + vec2(time * 0.05, 0.0));
                        // Layer 2: medium turbulence, advected by large
                        float medium = fbm(adv_uv * 2.6 + vec2(large * 1.4, -time * 0.11));
                        // Layer 3: fine-scale turbulence (fastest evolution)
                        float fine   = fbm(adv_uv * 6.2 + vec2(medium * 1.1, time * 0.22));

                        // Combine with contrast boost for visible structure
                        float gas = large * 0.48 + medium * 0.32 + fine * 0.20;
                        gas = pow(smoothstep(0.18, 0.88, gas), 0.85);

                        // ---- PROPAGATING SPIRAL DENSITY WAVES ----
                        // Real disks have acoustic/spiral density waves that propagate
                        // outward. This adds radial ripple motion.
                        float wave_phase = r_norm_c * 18.0 - phi_rot_c * 1.2 - time * 0.55;
                        float density_wave = 0.82 + 0.18 * sin(wave_phase);

                        // Final density: fluid gas texture × radial profile × density waves
                        float density_c = (disk_body * gas * density_wave + hot_rim_c)
                                        * inner_sm * outer_sm;
                        density_c = clamp(density_c, 0.0, 3.0);

                        // Relativistic Doppler factor
                        float beta_c   = clamp(sqrt(r_s * 0.5 / max(r_int - r_s, 0.01)), 0.0, 0.99);
                        float gamma_c  = 1.0 / sqrt(1.0 - beta_c * beta_c + 1e-5);
                        vec2  v_tan_c  = normalize(vec2(-intersect.y, intersect.x));
                        float cos_t_c  = dot(v_tan_c, -dir.xy);
                        float g_grav_c = sqrt(max(1.0 - r_s / (r_int + 1e-4), 1e-4));
                        float g_c      = g_grav_c / (gamma_c * (1.0 - beta_c * cos_t_c) + 1e-4);

                        // Color: Planckian or artistic palette
                        vec3 cine_col;
                        if (u_physical_mode) {
                            float tn_c = pow(6.0 / (r_int + 1e-4), 0.75);
                            cine_col   = blackbody(u_temperature * tn_c * g_c);
                        } else {
                            vec3 c_white = vec3(1.00, 0.98, 0.95);
                            vec3 c_cream = mix(disk_color_inner, c_white, 0.5);
                            vec3 c_amber = disk_color_inner;
                            vec3 c_copp  = disk_color_outer;
                            
                            float heat_c = clamp(1.0 - r_norm_c * 1.5, 0.0, 1.0);
                            vec3  mid2   = mix(c_amber, c_cream, clamp(heat_c * 2.0 - 1.0, 0.0, 1.0));
                            vec3  mid3   = mix(c_cream, c_white, clamp(heat_c * 3.0 - 2.0, 0.0, 1.0));
                            
                            cine_col = mix(c_copp,
                                mix(c_amber, mix(mid2, mid3,
                                    clamp(heat_c * 3.0 - 2.0, 0.0, 1.0)),
                                    clamp(heat_c * 2.0, 0.0, 1.0)),
                                clamp(heat_c * 1.5, 0.0, 1.0));
                                
                            float blue_c = clamp((g_c - 1.0) * 1.2, 0.0, 1.0);
                            float red_c  = clamp((1.0 - g_c) * 1.2, 0.0, 1.0);
                            cine_col = mix(cine_col, cine_col * vec3(0.6, 0.82, 1.5), blue_c);
                            cine_col = mix(cine_col, cine_col * vec3(1.45, 0.58, 0.18), red_c);
                        }

                        float beaming_c = pow(max(g_c, 1e-3), 3.5);
                        float opacity_c = 1.0 - exp(-density_c * 1.8);
                        vec3 emission_c = cine_col * beaming_c * disk_intensity * 2.2;

                        acc_color.rgb += (1.0 - acc_color.a) * emission_c * opacity_c;
                        acc_color.a   += (1.0 - acc_color.a) * opacity_c;
                        if (acc_color.a > 0.98) break;

                    // ---- PARTICLE MODE ----
                    } else {
                    vec2 tex_coord = (intersect.xy + 18.0) / 36.0;
                    
                    // Coordinates jittering with fade-out near the inner boundary to keep the edge sharp
                    float jitter_fade = smoothstep(r_isco * 1.35, r_isco * 1.6, r_int);
                    vec2 jitter_uv = intersect.xy * 8.0 + vec2(time * 0.8, -time * 0.5);
                    vec2 jitter = vec2(fbm(jitter_uv), fbm(jitter_uv + vec2(17.4, 31.2))) - vec2(0.5);
                    vec2 tex_coord_jittered = clamp(tex_coord + jitter * 0.008 * jitter_fade, 0.0, 1.0);

                    // Sample particle density and momentum from simulation texture directly
                    vec4 disk_sample = texture(disk_tex, tex_coord_jittered);
                    float density = disk_sample.r;
                    vec3 disk_vel = disk_sample.gba;

                    if (density > 1e-4) {
                        // Relativistic Doppler and gravitational redshift
                        vec3 v_coord = disk_vel / (density + 1e-4);
                        float beta = length(v_coord);
                        beta = clamp(beta, 0.0, 0.99);
                        float gamma = 1.0 / sqrt(1.0 - beta * beta + 1e-4);
                        float cos_theta = dot(normalize(v_coord), -dir);
                        
                        float g_grav = sqrt(max(1.0 - r_s / (r_int + 1e-4), 1e-4));
                        float g = g_grav / (gamma * (1.0 - beta * cos_theta) + 1e-4);

                        // Shakura-Sunyaev thermal profile T(r) ~ r^-0.75
                        float temp_norm = pow(6.0 / (r_int + 1e-4), 0.75);
                        vec3 emit_color;
                        
                        if (u_physical_mode) {
                            // Local Keplerian gas temperature driven by slider uniform
                            float T_local = u_temperature * temp_norm;
                            // Relativistic Doppler shift: observer sees T_obs = T_local * g
                            float T_obs = T_local * g;
                            emit_color = blackbody(T_obs);
                        } else {
                            emit_color = mix(disk_color_outer, disk_color_inner, clamp(temp_norm, 0.0, 1.0));
                            emit_color = shift_color(emit_color, g);
                        }

                        // Sheared gas noise for high-frequency detail
                        float angle_rot = atan(intersect.y, intersect.x) - 0.7 * time * pow(6.0 / (r_int + 1e-4), 1.5);
                        vec2 uv_noise = vec2(r_int * noise_scale * 0.4, angle_rot * 5.0);
                        
                        float n1 = noise2d(uv_noise);
                        float n2 = noise2d(uv_noise * 2.5 + vec2(time, -time));
                        float gas_noise = (n1 * 0.65 + n2 * 0.35);
                        float fibrous_noise = 1.0 - abs(gas_noise - 0.5) * 2.0;
                        gas_noise = smoothstep(0.2, 0.8, gas_noise) * 0.6 + pow(fibrous_noise, 3.0) * 0.4;

                        float active_density = density * (0.15 + 0.85 * gas_noise) * 4.0;
                        
                        // Thin plane alpha blending
                        float opacity = 1.0 - exp(-active_density * 2.5);
                        // Relativistic beaming: g^3.5 preserves correct brightness asymmetry
                        float beaming = pow(max(g, 1e-3), 3.5);
                        vec3 plane_color = emit_color * beaming * disk_intensity;

                        acc_color.rgb += (1.0 - acc_color.a) * plane_color * opacity;
                        acc_color.a += (1.0 - acc_color.a) * opacity;

                        if (acc_color.a > 0.98) break;
                    }
                    } // end cinematic/particle branch
                } // end disk radius check
            }
        }

        vec3 final_rgb;
        if (hit_horizon) {
            // Disk accumulated before reaching the horizon must still appear in front of the shadow
            final_rgb = acc_color.rgb; // pure black horizon behind any disk emission
        } else {
            float disk_opacity = clamp(acc_color.a, 0.0, 1.0);
            
            // Render far-field lensed skybox & lensed procedural stars
            vec3 bg_color = sample_skybox(dir) * u_skybox_intensity;
            vec3 stars = get_starfield(dir, pixel_angle);
            bg_color += stars;
            
            // Blend background with flat accretion disk
            final_rgb = acc_color.rgb + (1.0 - disk_opacity) * bg_color;

            // Thin Einstein photon ring: bordering the shadow boundary
            float dist_to_photon_sphere = abs(r_min - 1.5 * r_s);
            float photon_ring_glow = exp(-pow(dist_to_photon_sphere / 0.015, 2.0)) * 6.0;
            
            // Relativistic Doppler asymmetry for photon ring based on horizontal coordinate
            float ring_doppler = clamp(1.0 - uv.x * 0.5, 0.3, 2.2);
            vec3 ring_color = u_physical_mode ? blackbody(6500.0 * ring_doppler) : vec3(1.0, 0.8, 0.5);
            final_rgb += ring_color * (photon_ring_glow * u_disk_glow_intensity * ring_doppler);

            // Secondary glow of photons wrapping near horizon
            vec3 glow_color = u_physical_mode ? blackbody(6000.0) : disk_color_inner * 1.2;
            float final_glow = pow(photon_glow, 2.0) * 0.025 * disk_intensity;
            final_rgb += final_glow * glow_color;
        }

        // Cinematic tonemapping and gamma correction
        final_rgb = final_rgb / (final_rgb + vec3(1.0));
        final_rgb = pow(final_rgb, vec3(1.0 / 2.2));

        // Screen vignette
        vec2 d_uv = uv * 0.5;
        float vignette = 1.0 - dot(d_uv, d_uv) * 0.45;
        frag_color = vec4(final_rgb * vignette, 1.0);
    }
    """,
)

# Fullscreen Quad setup
vbo = ctx.buffer(np.array([-1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0], dtype='f4').tobytes())
vao = ctx.simple_vertex_array(prog, vbo, 'in_vert')

texture_data = np.zeros((width * height * 4,), dtype=np.float32)
with dpg.texture_registry(show=False):
    dpg.add_dynamic_texture(width=width, height=height, default_value=texture_data, tag="texture_tag")

# Global Orbital Camera Variables
camera_phi = 0.12
camera_theta = -0.5
cam_radius = 13.0
time_rate = 1.0
view_w = 1280
view_h = 800


# ---------------------------------------------------------------------------
# Interactive Callback Functions
# ---------------------------------------------------------------------------
def mouse_drag_callback(sender, app_data):
    global camera_phi, camera_theta
    if dpg.is_item_hovered("render_image") or dpg.is_item_hovered("hud_window") or dpg.is_item_hovered("viewport_panel"):
        button = app_data[0]
        dx = app_data[1]
        dy = app_data[2]
        if button == 0:
            sens = dpg.get_value("nav_sensitivity")
            camera_theta -= dx * 0.02 * sens
            camera_phi = max(-1.4, min(1.4, camera_phi + dy * 0.02 * sens))


def mouse_wheel_callback(sender, app_data):
    global cam_radius
    if dpg.is_item_hovered("render_image") or dpg.is_item_hovered("hud_window") or dpg.is_item_hovered("viewport_panel"):
        cam_radius = max(5.0, min(32.0, cam_radius - app_data * 0.6))


def key_press_callback(sender, app_data):
    global time_rate
    if app_data == 32:  # Space key
        time_rate = 1.0 if time_rate == 0.0 else 0.0
        update_time_status()
    elif app_data == 82:  # R key
        reset_particles()


def reset_particles():
    M_val = dpg.get_value("bh_mass")
    init_particles(M_val)
    print("Accretion disk particles reset successfully.")


def update_time_status():
    global time_rate
    if time_rate == 0.0:
        dpg.set_value("time_status", "STATUS: PAUSED")
        dpg.configure_item("time_status", color=(255, 100, 100))
    elif time_rate == 1.0:
        dpg.set_value("time_status", "STATUS: PLAYING")
        dpg.configure_item("time_status", color=(0, 255, 128))
    elif time_rate == -1.0:
        dpg.set_value("time_status", "STATUS: REWINDING (1x)")
        dpg.configure_item("time_status", color=(255, 180, 0))
    elif time_rate == 2.0:
        dpg.set_value("time_status", "STATUS: FAST-FORWARD (2x)")
        dpg.configure_item("time_status", color=(0, 229, 255))
    elif time_rate == 4.0:
        dpg.set_value("time_status", "STATUS: FAST-FORWARD (4x)")
        dpg.configure_item("time_status", color=(0, 229, 255))


def time_rewind_callback():
    global time_rate
    time_rate = -1.0
    update_time_status()


def time_pause_callback():
    global time_rate
    time_rate = 0.0
    update_time_status()


def time_play_callback():
    global time_rate
    time_rate = 1.0
    update_time_status()


def time_fwd2_callback():
    global time_rate
    time_rate = 2.0
    update_time_status()


def time_fwd4_callback():
    global time_rate
    time_rate = 4.0
    update_time_status()


def resize_callback(*args):
    global view_w, view_h
    vw, vh = dpg.get_viewport_width(), dpg.get_viewport_height()
    panel_w = 320
    view_w = max(400, vw - panel_w - 20)
    view_h = max(300, vh - 40)

    dpg.configure_item("viewport_panel", width=view_w, height=view_h)
    dpg.configure_item("render_image", width=view_w, height=view_h)
    dpg.configure_item("control_panel", width=panel_w, height=view_h)

    cx, cy = view_w / 2, view_h / 2
    dpg.configure_item("hud_drawlist", width=view_w, height=view_h)
    dpg.configure_item("hud_circle", center=(cx, cy))
    dpg.configure_item("hud_l1", p1=(cx - 30, cy), p2=(cx - 15, cy))
    dpg.configure_item("hud_l2", p1=(cx + 15, cy), p2=(cx + 30, cy))
    dpg.configure_item("hud_l3", p1=(cx, cy - 30), p2=(cx, cy - 15))
    dpg.configure_item("hud_l4", p1=(cx, cy + 15), p2=(cx, cy + 30))

    # Corner brackets for HUD
    dpg.configure_item("hud_c1", p1=(10, 10), p2=(35, 10))
    dpg.configure_item("hud_c2", p1=(10, 10), p2=(10, 35))
    dpg.configure_item("hud_c3", p1=(view_w - 10, 10), p2=(view_w - 35, 10))
    dpg.configure_item("hud_c4", p1=(view_w - 10, 10), p2=(view_w - 10, 35))
    dpg.configure_item("hud_c5", p1=(10, view_h - 10), p2=(35, view_h - 10))
    dpg.configure_item("hud_c6", p1=(10, view_h - 10), p2=(10, view_h - 35))
    dpg.configure_item("hud_c7", p1=(view_w - 10, view_h - 10), p2=(view_w - 35, view_h - 10))
    dpg.configure_item("hud_c8", p1=(view_w - 10, view_h - 10), p2=(view_w - 10, view_h - 35))
    
    dpg.configure_item("time_hud_window", pos=(view_w - 350, view_h - 105))


def update_hud(M_val, speed_factor, current_fps):
    r_s = 2.0 * M_val
    r_ps = 3.0 * M_val
    # Physical reference value (Circular orbit velocity at r=5M)
    v_peak = np.clip(speed_factor * np.sqrt(1.0 / 3.0), 0.0, 0.99)
    gamma = 1.0 / np.sqrt(1.0 - v_peak * v_peak)
    g_max = np.sqrt(0.6) / (gamma * (1.0 - v_peak))

    dpg.set_value("hud_mass", f"MASS (M): {M_val:.2f} M_sol")
    dpg.set_value("hud_rs", f"EVENT HORIZON (Rs): {r_s:.2f} G/c^2")
    dpg.set_value("hud_rps", f"PHOTON SPHERE (Rps): {r_ps:.2f} G/c^2")
    dpg.set_value("hud_gmax", f"PEAK DOPPLER FACTOR: {g_max:.2f}")
    dpg.set_value("hud_fps", f"TELEMETRY FPS: {current_fps:.1f}")


def preset_callback(sender, app_data):
    if app_data == "Gargantua Fire (Default)":
        dpg.set_value("color_inner", [255, 180, 50, 255])
        dpg.set_value("color_outer", [255, 60, 0, 255])
    elif app_data == "Nebula Cyan/Magenta":
        dpg.set_value("color_inner", [255, 0, 180, 255])
        dpg.set_value("color_outer", [0, 220, 255, 255])
    elif app_data == "Aurora Green":
        dpg.set_value("color_inner", [0, 120, 255, 255])
        dpg.set_value("color_outer", [0, 255, 100, 255])
    elif app_data == "Singularity Violet":
        dpg.set_value("color_inner", [255, 255, 255, 255])
        dpg.set_value("color_outer", [120, 0, 255, 255])


# Apply futuristic cyan-dark sci-fi dashboard theme
def apply_sci_fi_theme():
    with dpg.theme() as global_theme:
        with dpg.theme_component(dpg.mvAll):
            dpg.add_theme_color(dpg.mvThemeCol_WindowBg, (12, 12, 18, 245))
            dpg.add_theme_color(dpg.mvThemeCol_ChildBg, (18, 18, 26, 245))
            dpg.add_theme_color(dpg.mvThemeCol_Text, (220, 230, 242, 255))
            dpg.add_theme_color(dpg.mvThemeCol_FrameBg, (32, 32, 48, 255))
            dpg.add_theme_color(dpg.mvThemeCol_FrameBgHovered, (48, 48, 72, 255))
            dpg.add_theme_color(dpg.mvThemeCol_FrameBgActive, (64, 64, 96, 255))
            dpg.add_theme_color(dpg.mvThemeCol_SliderGrab, (0, 229, 255, 255))
            dpg.add_theme_color(dpg.mvThemeCol_SliderGrabActive, (128, 242, 255, 255))
            dpg.add_theme_color(dpg.mvThemeCol_Button, (24, 76, 120, 200))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (36, 114, 180, 255))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (0, 229, 255, 255))
            dpg.add_theme_color(dpg.mvThemeCol_Header, (30, 50, 80, 200))
            dpg.add_theme_color(dpg.mvThemeCol_Border, (40, 40, 60, 255))
            dpg.add_theme_color(dpg.mvThemeCol_ScrollbarBg, (12, 12, 18, 255))
            dpg.add_theme_color(dpg.mvThemeCol_ScrollbarGrab, (32, 32, 48, 255))

            dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 4)
            dpg.add_theme_style(dpg.mvStyleVar_WindowRounding, 8)
            dpg.add_theme_style(dpg.mvStyleVar_ChildRounding, 6)
            dpg.add_theme_style(dpg.mvStyleVar_GrabRounding, 4)

    dpg.bind_theme(global_theme)

    with dpg.theme(tag="viewport_theme"):
        with dpg.theme_component(dpg.mvChildWindow):
            dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 0, 0)
            dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 0, 0)
    dpg.bind_item_theme("viewport_panel", "viewport_theme")


# ---------------------------------------------------------------------------
# UI Layout Construction (Dear PyGui)
# ---------------------------------------------------------------------------
with dpg.window(tag="main_window", width=1600, height=840, no_title_bar=True, no_resize=True, no_move=True, no_scrollbar=True, no_scroll_with_mouse=True):
    with dpg.group(horizontal=True):
        # Render Viewport Panel
        with dpg.child_window(tag="viewport_panel", width=1280, height=800, border=False, no_scrollbar=True, no_scroll_with_mouse=True):
            dpg.add_image("texture_tag", width=1280, height=800, tag="render_image")
            with dpg.drawlist(width=1280, height=800, pos=(0, 0), tag="hud_drawlist"):
                dpg.draw_circle(center=(640, 400), radius=15, color=(0, 229, 255, 60), thickness=1, tag="hud_circle")
                dpg.draw_line(p1=(610, 400), p2=(625, 400), color=(0, 229, 255, 120), thickness=1, tag="hud_l1")
                dpg.draw_line(p1=(655, 400), p2=(670, 400), color=(0, 229, 255, 120), thickness=1, tag="hud_l2")
                dpg.draw_line(p1=(640, 370), p2=(640, 385), color=(0, 229, 255, 120), thickness=1, tag="hud_l3")
                dpg.draw_line(p1=(640, 415), p2=(640, 430), color=(0, 229, 255, 120), thickness=1, tag="hud_l4")
                dpg.draw_line(p1=(10, 10), p2=(35, 10), color=(0, 229, 255, 140), thickness=2, tag="hud_c1")
                dpg.draw_line(p1=(10, 10), p2=(10, 35), color=(0, 229, 255, 140), thickness=2, tag="hud_c2")
                dpg.draw_line(p1=(1270, 10), p2=(1245, 10), color=(0, 229, 255, 140), thickness=2, tag="hud_c3")
                dpg.draw_line(p1=(1270, 10), p2=(1270, 35), color=(0, 229, 255, 140), thickness=2, tag="hud_c4")
                dpg.draw_line(p1=(10, 790), p2=(35, 790), color=(0, 229, 255, 140), thickness=2, tag="hud_c5")
                dpg.draw_line(p1=(10, 790), p2=(10, 765), color=(0, 229, 255, 140), thickness=2, tag="hud_c6")
                dpg.draw_line(p1=(1270, 790), p2=(1245, 790), color=(0, 229, 255, 140), thickness=2, tag="hud_c7")
                dpg.draw_line(p1=(1270, 790), p2=(1270, 765), color=(0, 229, 255, 140), thickness=2, tag="hud_c8")

        # Telemetry & Controller Panel
        with dpg.child_window(tag="control_panel", width=320, height=800, border=True):
            dpg.add_text("   SPACESHIP CONTROL DECK  ", color=(0, 229, 255))
            dpg.add_text("  SPACETIME NAVIGATION PANEL", color=(100, 130, 160))
            dpg.add_separator()
            dpg.add_text("PHYSICS VARIABLES", color=(0, 229, 255))
            
            dpg.add_text("Black Hole Mass (M):", color=(180, 200, 220))
            dpg.add_slider_float(label="", tag="bh_mass", default_value=1.0, min_value=0.2, max_value=1.5, format="%.2f M_sol")
            with dpg.tooltip(parent="bh_mass"):
                dpg.add_text("Black Hole Mass (M):\n"
                             "Adjusts the mass of the black hole in solar masses.\n"
                             "Increasing mass scales up the event horizon (Rs) and\n"
                             "photon sphere (Rps) dimensions in world space.\n\n"
                             "Note: When increasing mass, the event horizon expands instantly\n"
                             "but gas particles take time to spiral outward to new stable\n"
                             "orbits, making the disk appear temporarily smaller.")

            dpg.add_text("Disk Rotation Speed:", color=(180, 200, 220))
            dpg.add_slider_float(label="", tag="disk_speed", default_value=1.0, min_value=0.0, max_value=2.0, format="%.2f Kepler")
            with dpg.tooltip(parent="disk_speed"):
                dpg.add_text("Disk Rotation Speed:\n"
                             "Multiplier for the orbital speed of the accretion disk gas.\n"
                             "A value of 1.0 corresponds to physical Keplerian orbital speed.")

            dpg.add_text("Accretion Rate (Drag):", color=(180, 200, 220))
            dpg.add_slider_float(label="", tag="accretion_rate", default_value=0.05, min_value=0.0, max_value=0.2, format="%.3f drag")
            with dpg.tooltip(parent="accretion_rate"):
                dpg.add_text("Accretion Rate (Drag):\n"
                             "Viscosity drag coefficient. Higher drag forces particles to lose\n"
                             "angular momentum and spiral into the black hole faster.")

            dpg.add_text("OPTICAL & SPECTRAL", color=(0, 229, 255))

            dpg.add_checkbox(label="Cinematic Mode", tag="cinematic_mode", default_value=False)
            with dpg.tooltip(parent="cinematic_mode"):
                dpg.add_text("Cinematic Mode:\n"
                             "Replaces the particle disk with a smooth procedural\n"
                             "gas disk inspired by Gargantua from Interstellar.\n"
                             "Uses layered orbital FBM streaks, radial heat falloff,\n"
                             "and relativistic Doppler color shift for a cinematic look.")
            
            dpg.add_checkbox(label="Planckian Blackbody Mode", tag="physical_mode", default_value=True)
            with dpg.tooltip(parent="physical_mode"):
                dpg.add_text("Planckian Blackbody Mode:\n"
                             "Enables physical blackbody color temperature rendering.\n"
                             "Calculates local Shakura-Sunyaev disk temperature T(r) ~ r^-0.75,\n"
                             "shifted by the relativistic Doppler redshift (T_obs = g * T).\n"
                             "This creates the physical blueshift/redshift asymmetry\n"
                             "(approaching gas is hot white-blue, receding is cool red).")

            dpg.add_text("Planckian Base Temp (K):", color=(180, 200, 220))
            dpg.add_slider_float(label="", tag="disk_temp", default_value=5800.0, min_value=2000.0, max_value=20000.0, format="%.0f K")
            with dpg.tooltip(parent="disk_temp"):
                dpg.add_text("Planckian Base Temp (K):\n"
                             "Base color temperature of the inner accretion disk in Kelvin.\n"
                             "Lower values shift the spectrum to saturated red/orange, whereas\n"
                             "higher values shift it to white/blue (up to 20000 K).")

            dpg.add_text("Doppler Beaming Strength:", color=(180, 200, 220))
            dpg.add_slider_float(label="", tag="doppler", default_value=1.0, min_value=0.0, max_value=2.5, format="%.2f beaming")
            with dpg.tooltip(parent="doppler"):
                dpg.add_text("Doppler Beaming Strength:\n"
                             "Adjusts the strength of relativistic Doppler beaming.\n"
                             "Higher values amplify the brightness difference between\n"
                             "the approaching (left) and receding (right) sides.")

            dpg.add_text("Disk Glow Intensity:", color=(180, 200, 220))
            dpg.add_slider_float(label="", tag="disk_intensity", default_value=1.2, min_value=0.1, max_value=4.0, format="%.2f brightness")
            with dpg.tooltip(parent="disk_intensity"):
                dpg.add_text("Disk Glow Intensity:\n"
                             "Overall brightness multiplier for the accretion disk.")

            dpg.add_text("Gas Turbulence Scale:", color=(180, 200, 220))
            dpg.add_slider_float(label="", tag="noise_scale", default_value=4.0, min_value=0.0, max_value=8.0, format="%.1f density")
            with dpg.tooltip(parent="noise_scale"):
                dpg.add_text("Gas Turbulence Scale:\n"
                             "Controls the spatial frequency of the gas density noise.\n"
                             "Higher values create more turbulent, high-frequency density filaments.")

            dpg.add_text("ENVIRONMENT & STARFIELD", color=(0, 229, 255))
            
            dpg.add_text("Lensed Star Intensity:", color=(180, 200, 220))
            dpg.add_slider_float(label="", tag="star_intensity", default_value=1.0, min_value=0.0, max_value=3.0, format="%.2f brightness")
            with dpg.tooltip(parent="star_intensity"):
                dpg.add_text("Lensed Star Intensity:\n"
                             "Brightness multiplier for the lensed background stars.")

            dpg.add_text("Skybox Intensity:", color=(180, 200, 220))
            dpg.add_slider_float(label="", tag="skybox_intensity", default_value=1.0, min_value=0.0, max_value=2.0, format="%.2f brightness")
            with dpg.tooltip(parent="skybox_intensity"):
                dpg.add_text("Skybox Intensity:\n"
                             "Brightness multiplier for the background skybox/nebula texture.")

            dpg.add_text("Navigation Sensitivity:", color=(180, 200, 220))
            dpg.add_slider_float(label="", tag="nav_sensitivity", default_value=0.2, min_value=0.05, max_value=1.0, format="%.2f sensitivity")
            with dpg.tooltip(parent="nav_sensitivity"):
                dpg.add_text("Navigation Sensitivity:\n"
                             "Sensitivity coefficient for orbiting and zooming the spaceship camera.")

            dpg.add_text("ARTISTIC PALETTE presets (Non-Physical)", tag="artistic_title", color=(0, 229, 255))
            
            dpg.add_text("Preset Color Palette:", tag="artistic_preset_label", color=(180, 200, 220))
            dpg.add_combo(items=["Gargantua Fire (Default)", "Nebula Cyan/Magenta", "Aurora Green", "Singularity Violet"],
                          label="", tag="color_preset", default_value="Gargantua Fire (Default)", callback=preset_callback)
            with dpg.tooltip(parent="color_preset"):
                dpg.add_text("Selects a pre-configured artistic color palette.", tag="tt_preset")
            
            dpg.add_text("Inner Disk Temperature Color:", tag="artistic_inner_label", color=(180, 200, 220))
            dpg.add_color_edit(label="", tag="color_inner", default_value=[255, 180, 50, 255], no_alpha=True)
            with dpg.tooltip(parent="color_inner"):
                dpg.add_text("Inner Disk Temperature Color:\n"
                             "Custom color representing the temperature of the inner accretion disk\n"
                             "when Planckian mode is disabled.", tag="tt_inner")

            dpg.add_text("Outer Disk Temperature Color:", tag="artistic_outer_label", color=(180, 200, 220))
            dpg.add_color_edit(label="", tag="color_outer", default_value=[255, 60, 0, 255], no_alpha=True)
            with dpg.tooltip(parent="color_outer"):
                dpg.add_text("Outer Disk Temperature Color:\n"
                             "Custom color representing the temperature of the outer accretion disk\n"
                             "when Planckian mode is disabled.", tag="tt_outer")

            dpg.add_separator()
            with dpg.group(horizontal=True):
                dpg.add_button(label="Reset Disk [R]", tag="reset_btn", callback=reset_particles, width=120, height=35)
                with dpg.tooltip(parent="reset_btn"):
                    dpg.add_text("Reset Disk [R]:\n"
                                 "Resets all accretion disk particles to their initial circular orbits.")

                def recenter_camera():
                    global camera_phi, camera_theta, cam_radius
                    camera_phi, camera_theta, cam_radius = 0.12, -0.5, 13.0

                dpg.add_button(label="Center View", tag="center_btn", callback=recenter_camera, width=120, height=35)
                with dpg.tooltip(parent="center_btn"):
                    dpg.add_text("Center View:\n"
                                 "Recenters the spaceship camera to its default orbital position and radius.")

            dpg.add_separator()
            dpg.add_text("MANUAL ORBITING PROTOCOL:", color=(100, 150, 180))
            dpg.add_text("- Left Mouse Drag: Rotate Camera", color=(150, 170, 190))
            dpg.add_text("- Scroll Mouse: Distance Range (Zoom)", color=(150, 170, 190))
            dpg.add_text("- Press [SPACE] to pause/unpause", color=(150, 170, 190))
            dpg.add_text("- Press [R] to reset gas cluster", color=(150, 170, 190))

# Sci-fi Overlay HUD (Overlay over ModernGL Render)
with dpg.window(tag="hud_window", pos=(35, 35), width=480, height=270, no_title_bar=True, no_resize=True, no_move=True, no_background=True, no_scrollbar=True, no_scroll_with_mouse=True):
    dpg.add_text(">>> BLACK HOLE TELEMETRY <<<", color=(0, 229, 255))
    dpg.add_text("-----------------------------", color=(0, 120, 180))
    dpg.add_text("MASS (M): 1.00 M_sol", tag="hud_mass", color=(180, 230, 255))
    dpg.add_text("EVENT HORIZON (Rs): 2.00 G/c^2", tag="hud_rs", color=(180, 230, 255))
    dpg.add_text("PHOTON SPHERE (Rps): 3.00 G/c^2", tag="hud_rps", color=(180, 230, 255))
    dpg.add_text("PEAK DOPPLER FACTOR: 1.51", tag="hud_gmax", color=(255, 128, 0))
    dpg.add_text("PARTICLE COUNT: 80,000", tag="hud_particles", color=(128, 255, 128))
    dpg.add_text("TELEMETRY FPS: 0.0", tag="hud_fps", color=(128, 255, 128))

with dpg.window(tag="time_hud_window", pos=(910, 690), width=350, height=100, no_title_bar=True, no_resize=True, no_move=True, no_background=True, no_scrollbar=True, no_scroll_with_mouse=True):
    dpg.add_text(">>> TIME CONTROL DECK <<<", color=(0, 229, 255))
    dpg.add_text("-----------------------------", color=(0, 120, 180))
    with dpg.group(horizontal=True):
        dpg.add_button(label="<< REW", callback=time_rewind_callback, width=60)
        dpg.add_button(label="|| PAUSE", callback=time_pause_callback, width=60)
        dpg.add_button(label="> PLAY", callback=time_play_callback, width=60)
        dpg.add_button(label=">> 2x", callback=time_fwd2_callback, width=45)
        dpg.add_button(label=">>> 4x", callback=time_fwd4_callback, width=45)
    dpg.add_text("STATUS: PLAYING", tag="time_status", color=(0, 255, 128))

# Handlers for inputs
with dpg.handler_registry():
    dpg.add_mouse_drag_handler(callback=mouse_drag_callback)
    dpg.add_mouse_wheel_handler(callback=mouse_wheel_callback)
    dpg.add_key_press_handler(callback=key_press_callback)

dpg.create_viewport(title='Relativistic Black Hole Telemetry Deck - Taichi & ModernGL', width=1600, height=840, resizable=True)
dpg.setup_dearpygui()
dpg.show_viewport()
dpg.set_primary_window("main_window", True)
dpg.set_viewport_resize_callback(resize_callback)

apply_sci_fi_theme()

start_time = time.time()
last_time = time.time()
fps_counter, fps_timer, fps_val = 0, 0.0, 60.0

last_color_inner = [255, 180, 50, 255]
last_color_outer = [255, 60, 0, 255]
was_physical_mode = True

# ---------------------------------------------------------------------------
# Main Rendering Loop
# ---------------------------------------------------------------------------
while dpg.is_dearpygui_running():
    M_val = dpg.get_value("bh_mass")
    speed_factor = dpg.get_value("disk_speed")
    accretion_rate = dpg.get_value("accretion_rate")
    doppler_strength = dpg.get_value("doppler")
    disk_intensity = dpg.get_value("disk_intensity")
    noise_scale = dpg.get_value("noise_scale")
    
    # Physics settings
    physical_mode  = dpg.get_value("physical_mode")
    cinematic_mode = dpg.get_value("cinematic_mode")
    star_intensity = dpg.get_value("star_intensity")
    skybox_intensity = dpg.get_value("skybox_intensity")
    disk_temp = dpg.get_value("disk_temp")

    speed_factor_field[None] = speed_factor
    accretion_rate_field[None] = accretion_rate

    # Step simulation if playing
    if time_rate != 0.0:
        step_simulation(0.016 * time_rate, M_val)

    # Render particles into texture grid in Taichi
    disk_max_r = 18.0
    rasterize_particles(disk_max_r)
    grid_np = disk_grid.to_numpy()
    disk_tex.write(grid_np.tobytes())

    # OpenGL Render Pass
    fbo.use()
    ctx.clear(0.0, 0.0, 0.0, 1.0)

    # Bind textures
    disk_tex.use(location=0)
    skybox_tex.use(location=1)
    prog['u_skybox'].value = 1

    # Camera Coordinate Calculations
    cam_x = cam_radius * np.cos(camera_phi) * np.cos(camera_theta)
    cam_y = cam_radius * np.cos(camera_phi) * np.sin(camera_theta)
    cam_z = cam_radius * np.sin(camera_phi)
    camera_pos = np.array([cam_x, cam_y, cam_z], dtype=np.float32)

    # Compute LookAt Camera Rotation Matrix
    f_vec = -camera_pos / np.linalg.norm(camera_pos)
    r_vec = np.array([-np.sin(camera_theta), np.cos(camera_theta), 0.0], dtype=np.float32)
    r_vec = r_vec / np.linalg.norm(r_vec)
    u_vec = np.cross(r_vec, f_vec)
    u_vec = u_vec / np.linalg.norm(u_vec)

    camera_rot = np.array([r_vec, u_vec, -f_vec], dtype=np.float32)

    # Update uniforms
    prog['camera_pos'].value = tuple(camera_pos)
    prog['camera_rot'].value = tuple(camera_rot.flatten())
    prog['r_s'].value = float(2.0 * M_val)
    prog['aspect'].value = float(view_w / view_h)

    prog['disk_intensity'].value = float(disk_intensity)
    prog['u_disk_glow_intensity'].value = float(disk_intensity)
    prog['doppler_strength'].value = float(doppler_strength)
    prog['noise_scale'].value = float(noise_scale)
    prog['time'].value = float(time.time() - start_time)
    prog['disk_color_inner'].value = tuple(np.array(dpg.get_value("color_inner")[:3]) / 255.0)
    prog['disk_color_outer'].value = tuple(np.array(dpg.get_value("color_outer")[:3]) / 255.0)
    
    # Disable artistic UI controls when Physical (Planckian) Mode is enabled
    is_artistic = not physical_mode
    
    if physical_mode and not was_physical_mode:
        last_color_inner = dpg.get_value("color_inner")
        last_color_outer = dpg.get_value("color_outer")
        dpg.set_value("color_inner", [100, 100, 100, 255])
        dpg.set_value("color_outer", [100, 100, 100, 255])
    elif not physical_mode and was_physical_mode:
        dpg.set_value("color_inner", last_color_inner)
        dpg.set_value("color_outer", last_color_outer)
    was_physical_mode = physical_mode

    dpg.configure_item("color_preset", enabled=is_artistic)
    dpg.configure_item("color_inner", enabled=is_artistic)
    dpg.configure_item("color_outer", enabled=is_artistic)
    
    gray_color = (100, 100, 100)
    dpg.configure_item("artistic_title", color=(0, 229, 255) if is_artistic else gray_color)
    dpg.configure_item("artistic_preset_label", color=(180, 200, 220) if is_artistic else gray_color)
    dpg.configure_item("artistic_inner_label", color=(180, 200, 220) if is_artistic else gray_color)
    dpg.configure_item("artistic_outer_label", color=(180, 200, 220) if is_artistic else gray_color)
    
    warn_text = "\n\n[DEACTIVATED IN PLANCKIAN MODE]"
    tt_preset_base = "Selects a pre-configured artistic color palette."
    tt_inner_base = "Inner Disk Temperature Color:\nCustom color representing the temperature of the inner accretion disk\nwhen Planckian mode is disabled."
    tt_outer_base = "Outer Disk Temperature Color:\nCustom color representing the temperature of the outer accretion disk\nwhen Planckian mode is disabled."
    
    dpg.set_value("tt_preset", tt_preset_base if is_artistic else tt_preset_base + warn_text)
    dpg.set_value("tt_inner", tt_inner_base if is_artistic else tt_inner_base + warn_text)
    dpg.set_value("tt_outer", tt_outer_base if is_artistic else tt_outer_base + warn_text)
    
    # Set physics extension uniforms
    prog['u_physical_mode'].value  = bool(physical_mode)
    prog['u_cinematic_mode'].value = bool(cinematic_mode)
    prog['u_star_intensity'].value    = float(star_intensity)
    prog['u_skybox_intensity'].value  = float(skybox_intensity)
    prog['u_view_height'].value       = float(view_h)
    prog['u_temperature'].value       = float(disk_temp)

    # Render fullscreen quad
    vao.render()

    # Read back framebuffer to DPG texture
    raw_output = fbo.read(components=4, dtype='f4')
    output_np = np.frombuffer(raw_output, dtype=np.float32)
    dpg.set_value("texture_tag", output_np)

    # Compute frame stats
    now = time.time()
    dt_frame = now - last_time
    last_time = now
    fps_counter += 1
    fps_timer += dt_frame
    if fps_timer >= 0.5:
        fps_val = fps_counter / fps_timer
        fps_counter, fps_timer = 0, 0.0

    update_hud(M_val, speed_factor, fps_val)
    dpg.render_dearpygui_frame()

dpg.destroy_context()
print("Simulation closed clean.")
