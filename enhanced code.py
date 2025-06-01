import asyncio
import platform as std_platform
import pygame
from pygame.locals import *
import sys
import math
import random
import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
import pandas as pd  # Added for CSV export

# --- PARAMETERS ---
WIDTH, HEIGHT = 1024, 768
FPS = 60

# Colors
RED = (1, 0, 0)
BLUE = (0, 0, 1)
YELLOW = (1, 1, 0)
GRAY = (0.7, 0.7, 0.7)
BLACK = (0, 0, 0)
BROWN = (0.55, 0.27, 0.07)
METAL = (0.8, 0.8, 0.9)
BRASS = (0.85, 0.65, 0.13)
ACRYLIC = (0.7, 0.9, 1.0, 0.6)
GREEN = (0, 1, 0)
CONCRETE = (0.6, 0.6, 0.6)
RACK_COLOR = (0.4, 0.4, 0.5)
WALL_COLOR = (0.5, 0.5, 0.55)
CRATE_COLOR = (0.65, 0.35, 0.15)

# Forklift parameters
FORKLIFT_WIDTH = 1.5
FORKLIFT_LENGTH = 2.5
FORKLIFT_HEIGHT = 0.5
WHEEL_RADIUS = 0.3
WHEEL_WIDTH = 0.2
WHEEL_ROLLERS = 12
SCREW_ROD_LENGTH = 2.0
SCREW_ROD_RADIUS = 0.05
STEPPER_SIZE = 0.25
FORK_WIDTH = 1.0
FORK_LENGTH = 1.8
FORK_THICKNESS = 0.1

# State
position = [0, 0, 0]
rotation = 0
fork_height = 5
max_fork_height = 95
min_fork_height = 5
wheel_steering = [0, 0, 0, 0]
screw_rotation = [0, 0]
linear_speed = 0.1
angular_speed = 2
fork_speed = 1
screw_rotation_speed = 15

# Loadcell and vibration parameters
loadcell_data = []
vibration_data_travel = []
vibration_data_lift = []
MAX_DATA_POINTS = 1000
vibration_amplitude = 0.0
load_weight = 0.0
is_vibrating = False
sample_rate = 60  # Hz, based on FPS
traveling = False
lifting = False

# Block parameters
block = {
    'position': [5, 5, 0.75],  # Elevated on shelf
    'size': [0.6, 0.4, 0.5],
    'color': BROWN,
    'is_picked': False,
    'attached_height': 0
}
place_position = [-5, -5, 0.25]
shelf_height = 0.5

# Warehouse parameters
WAREHOUSE_SIZE = 20
RACK_HEIGHT = 3
RACK_WIDTH = 2
RACK_DEPTH = 1
RACK_COUNT = 6
WALL_HEIGHT = 5

# Analysis flags
analysis_complete = False
plots_saved = False

# --- OPENGL PRIMITIVES ---
def create_cube(sx, sy, sz, color=RED):
    vertices = (
        (sx/2, -sy/2, -sz/2), (sx/2, sy/2, -sz/2), (-sx/2, sy/2, -sz/2), (-sx/2, -sy/2, -sz/2),
        (sx/2, -sy/2, sz/2), (sx/2, sy/2, sz/2), (-sx/2, -sy/2, sz/2), (-sx/2, sy/2, sz/2)
    )
    surfaces = (
        (0, 1, 2, 3), (4, 5, 7, 6), (0, 4, 6, 3), (1, 5, 7, 2), (0, 1, 5, 4), (3, 2, 7, 6)
    )
    normals = ((0, 0, -1), (0, 0, 1), (-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0))
    glColor4fv(color if len(color) == 4 else color + (1,))
    glBegin(GL_QUADS)
    for i, surface in enumerate(surfaces):
        glNormal3fv(normals[i])
        for vertex in surface:
            glVertex3fv(vertices[vertex])
    glEnd()

def create_cylinder(radius, length, sides=20, color=GRAY):
    glColor3fv(color)
    quadric = gluNewQuadric()
    gluCylinder(quadric, radius, radius, length, sides, 1)
    glPushMatrix()
    gluDisk(quadric, 0, radius, sides, 1)
    glTranslatef(0, 0, length)
    gluDisk(quadric, 0, radius, sides, 1)
    glPopMatrix()
    gluDeleteQuadric(quadric)

def create_mecanum_wheel(radius, width, rollers=8, angle=45, color=BLUE):
    glColor3fv(BLACK)
    quadric = gluNewQuadric()
    gluCylinder(quadric, radius, radius, width, 20, 1)
    glPushMatrix()
    gluDisk(quadric, 0, radius, 20, 1)
    glTranslatef(0, 0, width)
    gluDisk(quadric, 0, radius, 20, 1)
    glPopMatrix()
    for i in range(rollers):
        roll_angle = i * (360 / rollers)
        glPushMatrix()
        glRotatef(roll_angle, 0, 0, 1)
        glTranslatef(0, radius, width/2)
        glRotatef(angle, 0, 0, 1)
        glColor3fv(color)
        glRotatef(90, 1, 0, 0)
        create_cylinder(radius*0.3, width*1.2, 8, color)
        glPopMatrix()
    gluDeleteQuadric(quadric)

def create_threaded_rod(length, radius, rotation_angle=0, color=METAL):
    glColor3fv(color)
    create_cylinder(radius, length, 12, color)
    thread_spacing = length / 20
    thread_height = radius * 0.2
    glColor3fv(BLACK)
    for i in range(20):
        glPushMatrix()
        glTranslatef(0, 0, i * thread_spacing)
        glRotatef(rotation_angle + (i * 18), 0, 0, 1)
        glTranslatef(radius*0.8, 0, 0)
        create_cylinder(thread_height, thread_height*2, 8, BLACK)
        glPopMatrix()

def create_t_nut(radius, height, color=BRASS):
    create_cylinder(radius*2, height, 8, color)
    glPushMatrix()
    glTranslatef(0, 0, height/3)
    create_cylinder(radius*3, height/3, 8, color)
    glPopMatrix()
    glPushMatrix()
    glTranslatef(0, 0, -0.01)
    glColor3fv(BLACK)
    create_cylinder(radius*0.8, height+0.02, 8, BLACK)
    glPopMatrix()

def create_stepper_motor(size, color=BLACK):
    create_cube(size, size, size, color)
    glPushMatrix()
    glTranslatef(0, 0, size/2)
    glColor3fv(METAL)
    create_cylinder(size/8, size/2, 12, METAL)
    glPopMatrix()

def create_coupler(radius, length, color=METAL):
    create_cylinder(radius, length, 12, color)
    glPushMatrix()
    glTranslatef(0, 0, length/4)
    glColor3fv(BLACK)
    create_cylinder(radius*1.1, length/10, 12, BLACK)
    glPopMatrix()
    glPushMatrix()
    glTranslatef(0, 0, length*3/4)
    create_cylinder(radius*1.1, length/10, 12, BLACK)
    glPopMatrix()

def create_loadcell(width, height, depth, color=METAL):
    create_cube(width, height, depth, color)
    glPushMatrix()
    glTranslatef(0, 0, depth/2 + 0.01)
    glColor3fv(RED)
    create_cube(width*0.7, height*0.2, 0.01, RED)
    glPopMatrix()

# --- WAREHOUSE ENVIRONMENT ---
def draw_warehouse():
    # Floor
    glPushMatrix()
    glTranslatef(0, 0, -FORKLIFT_HEIGHT - 0.01)
    create_cube(WAREHOUSE_SIZE*2, WAREHOUSE_SIZE*2, 0.05, CONCRETE)
    glPopMatrix()

    # Walls
    glPushMatrix()
    glTranslatef(0, WAREHOUSE_SIZE, WALL_HEIGHT/2)
    create_cube(WAREHOUSE_SIZE*2, 0.2, WALL_HEIGHT, WALL_COLOR)
    glPopMatrix()
    glPushMatrix()
    glTranslatef(0, -WAREHOUSE_SIZE, WALL_HEIGHT/2)
    create_cube(WAREHOUSE_SIZE*2, 0.2, WALL_HEIGHT, WALL_COLOR)
    glPopMatrix()
    glPushMatrix()
    glTranslatef(WAREHOUSE_SIZE, 0, WALL_HEIGHT/2)
    create_cube(0.2, WAREHOUSE_SIZE*2, WALL_HEIGHT, WALL_COLOR)
    glPopMatrix()
    glPushMatrix()
    glTranslatef(-WAREHOUSE_SIZE, 0, WALL_HEIGHT/2)
    create_cube(0.2, WAREHOUSE_SIZE*2, WALL_HEIGHT, WALL_COLOR)
    glPopMatrix()

    # Racks
    for i in range(RACK_COUNT):
        for j in range(2):
            x = -WAREHOUSE_SIZE/2 + RACK_WIDTH*2*i + RACK_WIDTH
            y = WAREHOUSE_SIZE/2 - RACK_DEPTH*2 - j*WAREHOUSE_SIZE/2
            glPushMatrix()
            glTranslatef(x, y, RACK_HEIGHT/2)
            create_cube(RACK_WIDTH, RACK_DEPTH, RACK_HEIGHT, RACK_COLOR)
            for k in range(1, 4):
                glPushMatrix()
                glTranslatef(0, 0, k*RACK_HEIGHT/4 - RACK_HEIGHT/2)
                create_cube(RACK_WIDTH*0.9, RACK_DEPTH*0.9, 0.05, RACK_COLOR)
                if random.random() > 0.3:
                    glPushMatrix()
                    glTranslatef(random.uniform(-RACK_WIDTH/4, RACK_WIDTH/4), 0, 0.15)
                    create_cube(0.5, 0.5, 0.3, CRATE_COLOR)
                    glPopMatrix()
                glPopMatrix()
            glPopMatrix()

    # Shelf for initial block
    glPushMatrix()
    glTranslatef(5, 5, shelf_height/2)
    create_cube(1.0, 1.0, shelf_height, RACK_COLOR)
    glPopMatrix()

    # Additional crates
    crate_positions = [
        (-8, 8, 0.25), (-8, 7, 0.75), (-7, 8, 0.25),
        (8, -8, 0.25), (8, -7, 0.75), (7, -8, 0.25)
    ]
    for x, y, z in crate_positions:
        glPushMatrix()
        glTranslatef(x, y, z)
        create_cube(0.8, 0.8, 0.5, CRATE_COLOR)
        glPopMatrix()

# --- BLOCK HANDLING ---
def check_block_proximity():
    fork_pos = [
        position[0] + math.sin(math.radians(rotation)) * FORKLIFT_LENGTH/2,
        position[1] - math.cos(math.radians(rotation)) * FORKLIFT_LENGTH/2,
        fork_height / 100 * (SCREW_ROD_LENGTH - STEPPER_SIZE*1.25) + STEPPER_SIZE*1.25 + SCREW_ROD_RADIUS*4 + 0.08
    ]
    dist = math.sqrt(sum((a - b)**2 for a, b in zip(fork_pos, block['position'])))
    height_diff = abs(fork_pos[2] - block['position'][2])
    return dist < 1.0 and height_diff < 0.3

def pick_block():
    global block, load_weight
    if not block['is_picked'] and check_block_proximity():
        block['is_picked'] = True
        block['attached_height'] = fork_height
        load_weight = 5

def place_block():
    global block, load_weight
    if block['is_picked']:
        block['is_picked'] = False
        block['position'] = place_position.copy()
        load_weight = 0
        perform_vibration_analysis()

# --- VIBRATION ANALYSIS ---
def perform_vibration_analysis():
    global analysis_complete, plots_saved
    if not (vibration_data_travel or vibration_data_lift) or analysis_complete:
        return

    # Generate CSV file
    vibration_data = vibration_data_travel + vibration_data_lift
    time = np.arange(len(vibration_data)) / sample_rate
    vibration = np.array(vibration_data)
    df = pd.DataFrame({
        'Time (s)': time,
        'Vibration Amplitude (m)': vibration
    })
    df.to_csv('vibration_data.csv', index=False)
    print("CSV file 'vibration_data.csv' has been generated.")

    analysis_complete = True
    plt.figure(figsize=(10, 6))
    plt.plot(time, vibration, 'b-')
    plt.title('Vibration vs Time (Traveling and Lifting)')
    plt.xlabel('Time (s)')
    plt.ylabel('Vibration Amplitude (m)')
    plt.grid(True)
    plt.savefig('vibration_time.png')
    plt.close()

    N = len(vibration)
    yf = fft(vibration)
    xf = fftfreq(N, 1/sample_rate)[:N//2]
    plt.figure(figsize=(10, 6))
    plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]), 'b-')
    plt.title('Frequency Spectrum of Vibration')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude (m)')
    plt.grid(True)
    plt.xlim(0, sample_rate/2)
    plt.savefig('vibration_frequency.png')
    plt.close()

    plots_saved = True

# --- MAIN DRAW FUNCTION ---
def draw_forklift():
    global vibration_amplitude, is_vibrating, vibration_data_travel, vibration_data_lift
    
    glPushMatrix()
    glTranslatef(position[0], position[1], position[2])
    glRotatef(rotation, 0, 0, 1)
    create_cube(FORKLIFT_WIDTH, FORKLIFT_LENGTH, FORKLIFT_HEIGHT, BLUE)
    wheel_positions = [
        (-FORKLIFT_WIDTH/2, -FORKLIFT_LENGTH/2 + WHEEL_RADIUS, -FORKLIFT_HEIGHT/2),
        (FORKLIFT_WIDTH/2, -FORKLIFT_LENGTH/2 + WHEEL_RADIUS, -FORKLIFT_HEIGHT/2),
        (FORKLIFT_WIDTH/2, FORKLIFT_LENGTH/2 - WHEEL_RADIUS, -FORKLIFT_HEIGHT/2),
        (-FORKLIFT_WIDTH/2, FORKLIFT_LENGTH/2 - WHEEL_RADIUS, -FORKLIFT_HEIGHT/2)
    ]
    wheel_angles = [45, -45, 45, -45]
    for i, (wx, wy, wz) in enumerate(wheel_positions):
        glPushMatrix()
        glTranslatef(wx, wy, wz)
        glRotatef(wheel_steering[i], 0, 0, 1)
        if i == 0 or i == 3:
            glRotatef(90, 0, 1, 0)
        else:
            glRotatef(-90, 0, 1, 0)
        create_mecanum_wheel(WHEEL_RADIUS, WHEEL_WIDTH, WHEEL_ROLLERS, wheel_angles[i])
        glPopMatrix()
    
    glPushMatrix()
    glTranslatef(0, -FORKLIFT_LENGTH/2 + 0.1, FORKLIFT_HEIGHT/2)
    rod_distance = FORKLIFT_WIDTH * 0.3
    rod_height = SCREW_ROD_LENGTH
    
    for x_offset in [-rod_distance, rod_distance]:
        glPushMatrix()
        glTranslatef(x_offset, 0, 0)
        glColor3fv(GRAY)
        create_cylinder(SCREW_ROD_RADIUS*0.5, rod_height, 8, GRAY)
        glPopMatrix()
    
    for i, x_offset in enumerate([-rod_distance, rod_distance]):
        glPushMatrix()
        glTranslatef(x_offset, 0, -FORKLIFT_HEIGHT/2)
        create_stepper_motor(STEPPER_SIZE)
        glPushMatrix()
        glTranslatef(0, 0, STEPPER_SIZE * 0.75)
        create_coupler(STEPPER_SIZE/6, STEPPER_SIZE/2)
        glPopMatrix()
        glPushMatrix()
        glTranslatef(0, 0, STEPPER_SIZE * 1.25)
        create_threaded_rod(rod_height - STEPPER_SIZE * 1.25, SCREW_ROD_RADIUS, screw_rotation[i])
        glPopMatrix()
        glPopMatrix()
    
    normalized_height = (fork_height - min_fork_height) / (max_fork_height - min_fork_height)
    usable_rod_height = rod_height - STEPPER_SIZE * 1.25 - SCREW_ROD_RADIUS * 8
    current_height = STEPPER_SIZE * 1.25 + normalized_height * usable_rod_height
    
    fork_vibration_offset = 0
    if is_vibrating:
        fork_vibration_offset = math.sin(pygame.time.get_ticks() * 0.01) * vibration_amplitude
        if traveling and block['is_picked']:
            vibration_data_travel.append(fork_vibration_offset)
        if lifting and block['is_picked']:
            vibration_data_lift.append(fork_vibration_offset)
    
    glPushMatrix()
    glTranslatef(0, 0.1, current_height + fork_vibration_offset)
    
    for x_offset in [-rod_distance, rod_distance]:
        glPushMatrix()
        glTranslatef(x_offset, 0, 0)
        create_t_nut(SCREW_ROD_RADIUS * 2, SCREW_ROD_RADIUS * 8)
        glPopMatrix()
    
    glPushMatrix()
    glTranslatef(0, 0, SCREW_ROD_RADIUS * 4)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glColor4f(*ACRYLIC)
    create_cube(rod_distance*2+0.2, FORK_LENGTH, 0.08, ACRYLIC)
    glDisable(GL_BLEND)
    glPopMatrix()
    
    glPushMatrix()
    glTranslatef(0, FORK_LENGTH*0.3, SCREW_ROD_RADIUS * 4 + 0.08)
    create_loadcell(rod_distance*0.8, 0.3, 0.1, METAL)
    glPopMatrix()
    
    glPopMatrix()
    glPopMatrix()
    glPopMatrix()

def draw_block():
    if block['is_picked']:
        normalized_height = (fork_height - min_fork_height) / (max_fork_height - min_fork_height)
        usable_rod_height = SCREW_ROD_LENGTH - STEPPER_SIZE * 1.25 - SCREW_ROD_RADIUS * 8
        current_height = STEPPER_SIZE * 1.25 + normalized_height * usable_rod_height + SCREW_ROD_RADIUS * 4 + 0.08 + 0.1 + block['size'][2]/2
        glPushMatrix()
        glTranslatef(position[0], position[1], 0)
        glRotatef(rotation, 0, 0, 1)
        glTranslatef(0, -FORKLIFT_LENGTH/2 + 0.1 + FORK_LENGTH*0.3, current_height)
        create_cube(*block['size'], block['color'])
        glPopMatrix()
    else:
        glPushMatrix()
        glTranslatef(*block['position'])
        create_cube(*block['size'], block['color'])
        glPopMatrix()

# --- SUPPORT FUNCTIONS ---
def display_text(text, x, y, size=18):
    font = pygame.font.SysFont('Arial', size)
    text_surface = font.render(text, True, (255, 255, 255))
    text_rect = text_surface.get_rect()
    text_rect.topleft = (x, y)
    glMatrixMode(GL_PROJECTION)
    glPushMatrix()
    glLoadIdentity()
    gluOrtho2D(0, WIDTH, HEIGHT, 0)
    glMatrixMode(GL_MODELVIEW)
    glPushMatrix()
    glLoadIdentity()
    glDisable(GL_LIGHTING)
    glDisable(GL_DEPTH_TEST)
    screen = pygame.display.get_surface()
    screen.blit(text_surface, text_rect)
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_LIGHTING)
    glMatrixMode(GL_PROJECTION)
    glPopMatrix()
    glMatrixMode(GL_MODELVIEW)
    glPopMatrix()

def draw_grid():
    glDisable(GL_LIGHTING)
    glColor3f(0.5, 0.5, 0.5)
    glLineWidth(1.0)
    grid_size = WAREHOUSE_SIZE
    glBegin(GL_LINES)
    for i in range(-grid_size, grid_size + 1):
        glVertex3f(i, -grid_size, -FORKLIFT_HEIGHT)
        glVertex3f(i, grid_size, -FORKLIFT_HEIGHT)
        glVertex3f(-grid_size, i, -FORKLIFT_HEIGHT)
        glVertex3f(grid_size, i, -FORKLIFT_HEIGHT)
    glEnd()
    glEnable(GL_LIGHTING)

def draw_loadcell_graph():
    if not loadcell_data:
        return
    
    glMatrixMode(GL_PROJECTION)
    glPushMatrix()
    glLoadIdentity()
    gluOrtho2D(0, WIDTH, 0, HEIGHT)
    glMatrixMode(GL_MODELVIEW)
    glPushMatrix()
    glLoadIdentity()
    
    glDisable(GL_LIGHTING)
    glDisable(GL_DEPTH_TEST)
    
    glColor4f(0.0, 0.0, 0.0, 0.7)
    glBegin(GL_QUADS)
    glVertex2f(WIDTH - 310, HEIGHT - 210)
    glVertex2f(WIDTH - 10, HEIGHT - 210)
    glVertex2f(WIDTH - 10, HEIGHT - 10)
    glVertex2f(WIDTH - 310, HEIGHT - 10)
    glEnd()
    
    glColor3f(1.0, 1.0, 1.0)
    glBegin(GL_LINE_LOOP)
    glVertex2f(WIDTH - 300, HEIGHT - 200)
    glVertex2f(WIDTH - 20, HEIGHT - 200)
    glVertex2f(WIDTH - 20, HEIGHT - 20)
    glVertex2f(WIDTH - 300, HEIGHT - 20)
    glEnd()
    
    glColor3f(0.0, 1.0, 0.0)
    glBegin(GL_LINE_STRIP)
    for i, value in enumerate(loadcell_data[-MAX_DATA_POINTS:]):
        x = WIDTH - 300 + (280 * i / min(len(loadcell_data), MAX_DATA_POINTS))
        y = HEIGHT - 110 + value * 80
        glVertex2f(x, y)
    glEnd()
    
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_LIGHTING)
    glMatrixMode(GL_PROJECTION)
    glPopMatrix()
    glMatrixMode(GL_MODELVIEW)
    glPopMatrix()

def update_physics():
    global screw_rotation, loadcell_data, vibration_amplitude, is_vibrating, traveling, lifting
    
    keys = pygame.key.get_pressed()
    if keys[K_r] and fork_height < max_fork_height:
        screw_rotation[0] = (screw_rotation[0] + screw_rotation_speed) % 360
        screw_rotation[1] = (screw_rotation[1] + screw_rotation_speed) % 360
        lifting = True
    elif keys[K_f] and fork_height > min_fork_height:
        screw_rotation[0] = (screw_rotation[0] - screw_rotation_speed) % 360
        screw_rotation[1] = (screw_rotation[1] - screw_rotation_speed) % 360
        lifting = True
    else:
        lifting = False
    
    traveling = any(keys[k] for k in [K_w, K_s, K_a, K_d, K_q, K_e])
    
    base_value = load_weight / 10.0
    noise = random.uniform(-0.05, 0.05)
    movement_vibration = 0
    if (traveling or lifting) and block['is_picked']:
        movement_vibration = math.sin(pygame.time.get_ticks() * 0.01) * 0.2
        is_vibrating = True
        vibration_amplitude = 0.03 + (load_weight * 0.01)
    elif is_vibrating:
        vibration_amplitude *= 0.98
        if vibration_amplitude < 0.001:
            is_vibrating = False
            vibration_amplitude = 0
    
    current_value = base_value + noise + movement_vibration
    loadcell_data.append(current_value)
    
    if len(loadcell_data) > MAX_DATA_POINTS:
        loadcell_data.pop(0)
    if len(vibration_data_travel) > MAX_DATA_POINTS:
        vibration_data_travel.pop(0)
    if len(vibration_data_lift) > MAX_DATA_POINTS:
        vibration_data_lift.pop(0)

# --- MAIN LOOP ---
async def main():
    global position, rotation, fork_height, wheel_steering, load_weight
    
    pygame.init()
    pygame.display.set_mode((WIDTH, HEIGHT), DOUBLEBUF | OPENGL)
    pygame.display.set_caption('Forklift Simulation with Warehouse')
    
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
    glEnable(GL_LIGHT1)
    glLightfv(GL_LIGHT0, GL_POSITION, (5, 5, 10, 1))
    glLightfv(GL_LIGHT0, GL_AMBIENT, (0.3, 0.3, 0.3, 1))
    glLightfv(GL_LIGHT0, GL_DIFFUSE, (0.7, 0.7, 0.7, 1))
    glLightfv(GL_LIGHT1, GL_POSITION, (-5, -5, 10, 1))
    glLightfv(GL_LIGHT1, GL_AMBIENT, (0.2, 0.2, 0.2, 1))
    glLightfv(GL_LIGHT1, GL_DIFFUSE, (0.5, 0.5, 0.5, 1))
    glEnable(GL_COLOR_MATERIAL)
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
    
    gluPerspective(45, (WIDTH / HEIGHT), 0.1, 50.0)
    glTranslatef(0.0, -1.0, -10.0)
    glRotatef(20, 1, 0, 0)
    
    view_angle_x = 20
    view_angle_y = 0
    view_distance = 10
    
    fork_height = min_fork_height
    
    clock = pygame.time.Clock()
    
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 4:
                    view_distance = max(5, view_distance - 0.5)
                elif event.button == 5:
                    view_distance = min(20, view_distance + 0.5)
            if event.type == pygame.MOUSEMOTION and event.buttons[2]:
                view_angle_x += event.rel[1] * 0.5
                view_angle_y += event.rel[0] * 0.5
                view_angle_x = max(-90, min(90, view_angle_x))
            if event.type == pygame.KEYDOWN:
                if event.key == K_p:
                    pick_block()
                elif event.key == K_o:
                    place_block()
        
        keys = pygame.key.get_pressed()
        if keys[K_ESCAPE]:
            pygame.quit()
            return
        
        if keys[K_w]:
            position[0] += linear_speed * math.sin(math.radians(rotation))
            position[1] -= linear_speed * math.cos(math.radians(rotation))
            wheel_steering[:] = [0, 0, 0, 0]
        if keys[K_s]:
            position[0] -= linear_speed * math.sin(math.radians(rotation))
            position[1] += linear_speed * math.cos(math.radians(rotation))
            wheel_steering[:] = [0, 0, 0, 0]
        if keys[K_a]:
            position[0] -= linear_speed * math.cos(math.radians(rotation))
            position[1] -= linear_speed * math.sin(math.radians(rotation))
            wheel_steering[:] = [15, -15, -15, 15]
        if keys[K_d]:
            position[0] += linear_speed * math.cos(math.radians(rotation))
            position[1] += linear_speed * math.sin(math.radians(rotation))
            wheel_steering[:] = [-15, 15, 15, -15]
        if keys[K_q]:
            rotation = (rotation - angular_speed) % 360
            wheel_steering[:] = [20, 20, -20, -20]
        if keys[K_e]:
            rotation = (rotation + angular_speed) % 360
            wheel_steering[:] = [-20, -20, 20, 20]
        if keys[K_r] and fork_height < max_fork_height:
            fork_height += fork_speed
        if keys[K_f] and fork_height > min_fork_height:
            fork_height -= fork_speed
        
        fork_height = max(min_fork_height, min(fork_height, max_fork_height))
        
        update_physics()
        
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, (WIDTH / HEIGHT), 0.1, 50.0)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        
        cam_x = view_distance * math.cos(math.radians(view_angle_y)) * math.cos(math.radians(view_angle_x))
        cam_y = view_distance * math.sin(math.radians(view_angle_y)) * math.cos(math.radians(view_angle_x))
        cam_z = view_distance * math.sin(math.radians(view_angle_x))
        
        gluLookAt(cam_x, cam_y, cam_z, 0, 0, 0, 0, 0, 1)
        
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        draw_warehouse()
        draw_grid()
        draw_forklift()
        draw_block()
        draw_loadcell_graph()
        
        instructions = [
            "Controls:",
            "W/S - Forward/Backward",
            "A/D - Strafe Left/Right",
            "Q/E - Rotate Left/Right",
            "R/F - Raise/Lower Fork",
            "P - Pick Block",
            "O - Place Block",
            "Right Click + Move - Rotate Camera",
            "Mouse Wheel - Zoom In/Out",
            "ESC - Quit",
            f"Fork Height: {fork_height:.1f}%",
            f"Load Weight: {load_weight} kg",
            f"Block Picked: {block['is_picked']}",
            f"Analysis Done: {analysis_complete}",
            f"Plots Saved: {plots_saved}"
        ]
        
        for i, text in enumerate(instructions):
            display_text(text, 10, 10 + i*20)
        
        pygame.display.flip()
        clock.tick(FPS)
        await asyncio.sleep(1.0 / FPS)

if std_platform.system() == "Emscripten":
    asyncio.ensure_future(main())
else:
    if __name__ == "__main__":
        asyncio.run(main())