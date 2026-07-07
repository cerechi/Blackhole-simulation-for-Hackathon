import dearpygui.dearpygui as dpg
import moderngl
import numpy as np
import time

dpg.create_context()

width, height = 800, 600

# Create ModernGL standalone context
ctx = moderngl.create_context(standalone=True)
fbo = ctx.framebuffer(
    color_attachments=ctx.texture((width, height), 4, dtype='f4'),
)

# A simple shader to clear screen with a color that changes over time
prog = ctx.program(
    vertex_shader="""
    #version 330
    in vec2 in_vert;
    out vec2 uv;
    void main() {
        uv = in_vert * 0.5 + 0.5;
        gl_Position = vec4(in_vert, 0.0, 1.0);
    }
    """,
    fragment_shader="""
    #version 330
    in vec2 uv;
    out vec4 fragColor;
    uniform float time;
    void main() {
        fragColor = vec4(uv.x, uv.y, sin(time) * 0.5 + 0.5, 1.0);
    }
    """
)

# Fullscreen quad
vbo = ctx.buffer(np.array([-1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1], dtype='f4').tobytes())
vao = ctx.simple_vertex_array(prog, vbo, 'in_vert')

# Dynamic texture for DPG
texture_data = np.zeros((width * height * 4,), dtype=np.float32)
with dpg.texture_registry(show=False):
    dpg.add_dynamic_texture(width=width, height=height, default_value=texture_data, tag="texture_tag")

with dpg.window(label="ModernGL Render Window", width=width, height=height):
    dpg.add_image("texture_tag")

dpg.create_viewport(title='ModernGL + Dear PyGui test', width=width+50, height=height+50)
dpg.setup_dearpygui()
dpg.show_viewport()

t0 = time.time()
while dpg.is_dearpygui_running():
    t = time.time() - t0
    
    # Render with ModernGL
    fbo.use()
    ctx.clear(0, 0, 0, 1)
    if 'time' in prog:
        prog['time'].value = t
    vao.render()
    
    # Read back to NumPy array
    raw_data = fbo.read(components=4, dtype='f4')
    img_data = np.frombuffer(raw_data, dtype=np.float32)
    
    # Update DPG texture
    dpg.set_value("texture_tag", img_data)
    
    dpg.render_dearpygui_frame()

dpg.destroy_context()
print("Success!")
