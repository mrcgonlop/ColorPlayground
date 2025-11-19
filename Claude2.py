"""
DaVinci Wide Gamut & Film Emulation Experimentation Notebook
============================================================
This notebook explores DaVinci Wide Gamut color space and provides tools
for understanding film emulation color transforms (DCTLs).
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from mpl_toolkits.mplot3d import Axes3D

# ============================================================================
# PART 1: DaVinci Wide Gamut Color Space Definition
# ============================================================================

class ColorSpace:
    """Color space definition with transformation matrices"""
    
    def __init__(self, name, to_xyz_matrix, from_xyz_matrix):
        self.name = name
        self.to_xyz = np.array(to_xyz_matrix)
        self.from_xyz = np.array(from_xyz_matrix)

# DaVinci Wide Gamut primaries and white point (D65)
# These are the official DWG matrices for conversion to/from XYZ
DWG_TO_XYZ = [
    [0.700622320175, 0.148774802685, 0.101058728993],
    [0.274118483067, 0.873631775379, -0.147750422359],
    [-0.098962903023, -0.137895315886, 1.325916051865]
]

XYZ_TO_DWG = [
    [1.512205958366, -0.236214995384, -0.095667056739],
    [-0.461994469166, 1.204409718513, 0.025726420060],
    [0.077355667949, 0.109971389174, 0.764165461063]
]

# Rec.709 for comparison
REC709_TO_XYZ = [
    [0.412391, 0.357584, 0.180481],
    [0.212639, 0.715169, 0.072192],
    [0.019331, 0.119195, 0.950532]
]

XYZ_TO_REC709 = [
    [3.240970, -1.537383, -0.498611],
    [-0.969244, 1.875968, 0.041555],
    [0.055630, -0.203977, 1.056972]
]

# Create color space objects
dwg = ColorSpace("DaVinci Wide Gamut", DWG_TO_XYZ, XYZ_TO_DWG)
rec709 = ColorSpace("Rec.709", REC709_TO_XYZ, XYZ_TO_REC709)

# ============================================================================
# PART 2: Color Space Conversion Functions
# ============================================================================

def rgb_to_xyz(rgb, color_space):
    """Convert RGB to CIE XYZ"""
    return np.dot(color_space.to_xyz, rgb)

def xyz_to_rgb(xyz, color_space):
    """Convert CIE XYZ to RGB"""
    return np.dot(color_space.from_xyz, xyz)

def convert_color_space(rgb, from_space, to_space):
    """Convert RGB from one color space to another via XYZ"""
    xyz = rgb_to_xyz(rgb, from_space)
    return xyz_to_rgb(xyz, to_space)

# ============================================================================
# PART 3: Film Emulation Building Blocks
# ============================================================================

def linear_to_log(linear, black_offset=0.0, log_gain=1.0):
    """
    Convert linear to logarithmic encoding (like film density)
    Similar to Cineon or DaVinci Intermediate log
    """
    return log_gain * np.log10(linear + black_offset)

def log_to_linear(log_val, black_offset=0.0, log_gain=1.0):
    """Convert log back to linear"""
    return np.power(10, log_val / log_gain) - black_offset

def rgb_to_hsv(rgb):
    """Convert RGB to HSV for hue-based operations"""
    r, g, b = rgb
    max_c = max(r, g, b)
    min_c = min(r, g, b)
    delta = max_c - min_c
    
    # Value
    v = max_c
    
    # Saturation
    s = 0 if max_c == 0 else delta / max_c
    
    # Hue
    if delta == 0:
        h = 0
    elif max_c == r:
        h = 60 * (((g - b) / delta) % 6)
    elif max_c == g:
        h = 60 * (((b - r) / delta) + 2)
    else:
        h = 60 * (((r - g) / delta) + 4)
    
    return np.array([h, s, v])

def hsv_to_rgb(hsv):
    """Convert HSV back to RGB"""
    h, s, v = hsv
    c = v * s
    x = c * (1 - abs((h / 60) % 2 - 1))
    m = v - c
    
    if h < 60:
        r, g, b = c, x, 0
    elif h < 120:
        r, g, b = x, c, 0
    elif h < 180:
        r, g, b = 0, c, x
    elif h < 240:
        r, g, b = 0, x, c
    elif h < 300:
        r, g, b = x, 0, c
    else:
        r, g, b = c, 0, x
    
    return np.array([r + m, g + m, b + m])

def apply_hue_rotation(rgb, hue_shift_degrees):
    """
    Apply hue rotation - common in film emulation
    This simulates how different film stocks shift colors
    """
    hsv = rgb_to_hsv(rgb)
    hsv[0] = (hsv[0] + hue_shift_degrees) % 360
    return hsv_to_rgb(hsv)

def apply_hue_vs_hue_curve(rgb, curve_points):
    """
    Apply hue-dependent hue shift
    curve_points: list of (input_hue, output_hue_shift) tuples
    This is how film creates signature color responses
    """
    hsv = rgb_to_hsv(rgb)
    input_hue = hsv[0]
    
    # Simple linear interpolation between curve points
    curve_points = sorted(curve_points, key=lambda x: x[0])
    
    if input_hue <= curve_points[0][0]:
        shift = curve_points[0][1]
    elif input_hue >= curve_points[-1][0]:
        shift = curve_points[-1][1]
    else:
        for i in range(len(curve_points) - 1):
            h1, s1 = curve_points[i]
            h2, s2 = curve_points[i + 1]
            if h1 <= input_hue <= h2:
                t = (input_hue - h1) / (h2 - h1)
                shift = s1 + t * (s2 - s1)
                break
    
    hsv[0] = (hsv[0] + shift) % 360
    return hsv_to_rgb(hsv)

def apply_saturation_adjustment(rgb, saturation_factor):
    """Adjust color saturation"""
    hsv = rgb_to_hsv(rgb)
    hsv[1] = np.clip(hsv[1] * saturation_factor, 0, 1)
    return hsv_to_rgb(hsv)

def film_negative_exposure(rgb, exposure_compensation=1.0):
    """Simulate film negative response to light"""
    # Film has a gentle shoulder and toe
    return np.power(rgb * exposure_compensation, 0.6)

def film_print_stock(rgb, contrast=1.2, lift=0.0):
    """Simulate film print characteristics"""
    return np.clip(np.power(rgb, 1/contrast) + lift, 0, 1)

# ============================================================================
# PART 4: Complete Film Emulation Pipeline (DCTL-style)
# ============================================================================

class FilmEmulationPipeline:
    """
    A complete film emulation pipeline that mimics what a DCTL would do
    This follows typical color grading workflow
    """
    
    def __init__(self):
        # Default parameters
        self.exposure = 1.0
        self.hue_shift = 0.0  # Global hue rotation
        self.saturation = 1.0
        self.contrast = 1.0
        self.film_type = "neutral"
        
        # Advanced: Hue vs Hue curve
        self.hue_curve = [
            (0, 0),      # Reds
            (60, 0),     # Yellows
            (120, 0),    # Greens
            (180, 0),    # Cyans
            (240, 0),    # Blues
            (300, 0),    # Magentas
            (360, 0)
        ]
    
    def set_film_preset(self, film_type):
        """Set parameters to emulate different film stocks"""
        if film_type == "fuji_velvia":
            # Velvia: saturated, warm, vibrant greens and blues
            self.saturation = 1.4
            self.hue_curve = [
                (0, 5),      # Reds slightly warmer
                (60, 10),    # Yellows/oranges warmer
                (120, -5),   # Greens cooler/more cyan
                (180, 0),    # Cyans neutral
                (240, 5),    # Blues slightly purple
                (300, 0),    # Magentas neutral
                (360, 5)
            ]
        elif film_type == "kodak_portra":
            # Portra: soft, pastel skin tones, cooler shadows
            self.saturation = 0.9
            self.hue_curve = [
                (0, -3),     # Reds slightly cooler
                (30, 5),     # Orange skin tones warmer
                (120, 0),    # Greens neutral
                (180, -5),   # Cyans slightly green
                (240, -5),   # Blues slightly cyan
                (300, 0),    # Magentas neutral
                (360, -3)
            ]
        elif film_type == "kodak_ektar":
            # Ektar: ultra fine grain, saturated, vivid
            self.saturation = 1.3
            self.hue_curve = [
                (0, 0),
                (60, 5),
                (120, 0),
                (180, -5),
                (240, 0),
                (300, 0),
                (360, 0)
            ]
    
    def process(self, rgb_dwg):
        """
        Process a DWG RGB value through the film pipeline
        
        Pipeline stages:
        1. Exposure adjustment (scene-referred)
        2. Film negative response
        3. Color space operations (hue, saturation)
        4. Film print characteristics
        5. Output tone mapping
        """
        rgb = np.array(rgb_dwg, dtype=float)
        
        # Stage 1: Exposure (simulates more/less light hitting film)
        rgb = rgb * self.exposure
        
        # Stage 2: Film negative characteristic curve
        rgb = film_negative_exposure(rgb)
        
        # Stage 3: Hue operations (this is where film "personality" comes in)
        rgb = apply_hue_vs_hue_curve(rgb, self.hue_curve)
        rgb = apply_hue_rotation(rgb, self.hue_shift)
        
        # Stage 4: Saturation
        rgb = apply_saturation_adjustment(rgb, self.saturation)
        
        # Stage 5: Film print contrast
        rgb = film_print_stock(rgb, self.contrast)
        
        # Final clipping
        rgb = np.clip(rgb, 0, 1)
        
        return rgb

# ============================================================================
# PART 5: Visualization Tools
# ============================================================================

def plot_color_gamut_2d():
    """Plot color gamut in CIE xy chromaticity diagram"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot spectral locus (simplified)
    wavelengths = np.linspace(380, 700, 100)
    # This is a simplified version - real spectral locus needs CIE data
    
    # DWG primaries in xy
    dwg_primaries_xy = [
        (0.8000, 0.3177),  # Red
        (0.1682, 0.9877),  # Green
        (0.0790, -0.1155), # Blue
        (0.8000, 0.3177)   # Close the triangle
    ]
    
    # Rec.709 primaries in xy
    rec709_primaries_xy = [
        (0.64, 0.33),   # Red
        (0.30, 0.60),   # Green
        (0.15, 0.06),   # Blue
        (0.64, 0.33)    # Close
    ]
    
    dwg_tri = Polygon(dwg_primaries_xy[:-1], fill=False, edgecolor='red', 
                      linewidth=2, label='DaVinci Wide Gamut')
    rec709_tri = Polygon(rec709_primaries_xy[:-1], fill=False, edgecolor='blue', 
                        linewidth=2, label='Rec.709')
    
    ax.add_patch(dwg_tri)
    ax.add_patch(rec709_tri)
    
    ax.set_xlim(-0.1, 0.9)
    ax.set_ylim(-0.2, 1.0)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Color Gamut Comparison (CIE xy)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    return fig

def plot_hue_rotation_demo(pipeline):
    """Visualize hue rotation effect"""
    hues = np.linspace(0, 360, 360)
    input_hues = []
    output_hues = []
    
    for h in hues:
        # Create a fully saturated color at this hue
        rgb = hsv_to_rgb([h, 1.0, 1.0])
        # Process through pipeline
        processed = pipeline.process(rgb)
        # Get output hue
        hsv_out = rgb_to_hsv(processed)
        
        input_hues.append(h)
        output_hues.append(hsv_out[0])
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(input_hues, output_hues, linewidth=2, label='Output Hue')
    ax.plot(input_hues, input_hues, '--', alpha=0.5, label='Input Hue (reference)')
    ax.set_xlabel('Input Hue (degrees)')
    ax.set_ylabel('Output Hue (degrees)')
    ax.set_title('Hue vs Hue Response (Film Emulation)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig

def plot_3d_color_cube(pipeline, resolution=8):
    """Plot RGB color cube showing transformation"""
    fig = plt.figure(figsize=(14, 6))
    
    # Generate color cube
    r = np.linspace(0, 1, resolution)
    g = np.linspace(0, 1, resolution)
    b = np.linspace(0, 1, resolution)
    
    colors_input = []
    colors_output = []
    
    for ri in r:
        for gi in g:
            for bi in b:
                rgb_in = np.array([ri, gi, bi])
                rgb_out = pipeline.process(rgb_in)
                colors_input.append(rgb_in)
                colors_output.append(rgb_out)
    
    colors_input = np.array(colors_input)
    colors_output = np.array(colors_output)
    
    # Plot input cube
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(colors_input[:, 0], colors_input[:, 1], colors_input[:, 2],
                c=colors_input, s=20, alpha=0.6)
    ax1.set_xlabel('R')
    ax1.set_ylabel('G')
    ax1.set_zlabel('B')
    ax1.set_title('Input RGB Cube (DWG)')
    
    # Plot output cube
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(colors_output[:, 0], colors_output[:, 1], colors_output[:, 2],
                c=np.clip(colors_output, 0, 1), s=20, alpha=0.6)
    ax2.set_xlabel('R')
    ax2.set_ylabel('G')
    ax2.set_zlabel('B')
    ax2.set_title('Output RGB Cube (After Film Emulation)')
    
    plt.tight_layout()
    return fig

# ============================================================================
# PART 6: N-Log to DaVinci Wide Gamut Intermediate (DWG/Log)
# ============================================================================

def nlog_to_linear(nlog_12bit):
    """
    Convert Nikon N-Log (12-bit) to linear
    N-Log parameters based on Nikon specifications
    """
    # Normalize 12-bit to 0-1 range
    nlog_norm = nlog_12bit / 4095.0
    
    # N-Log to linear conversion
    # These are approximate parameters - adjust based on Nikon specs
    cut = 0.328
    
    linear = np.where(
        nlog_norm < cut,
        (nlog_norm - 0.12) / 5.0,  # Linear segment
        np.power(10, (nlog_norm - 0.619) / 0.25)  # Log segment
    )
    
    return np.clip(linear, 0, 1)

def linear_to_dwg_log(linear):
    """
    Convert linear to DaVinci Intermediate log
    DaVinci Intermediate log curve
    """
    # DaVinci Intermediate log parameters
    A = 0.0075
    B = 7.0
    C = 0.07329248
    
    log = np.where(
        linear <= 0.00262409,
        linear * 14.98325,
        (np.log10(linear + A) - np.log10(A)) * B + C
    )
    
    return np.clip(log, 0, 1)

def nlog_to_dwg_log(nlog_12bit):
    """Complete N-Log to DWG/Log conversion"""
    linear = nlog_to_linear(nlog_12bit)
    return linear_to_dwg_log(linear)

# ============================================================================
# PART 7: Synthetic Test Image Generation
# ============================================================================

def generate_hue_saturation_chart(width=512, height=512):
    """
    Generate a chart showing all hues (horizontal) and saturations (vertical)
    at constant luminance
    """
    img = np.zeros((height, width, 3))
    
    for y in range(height):
        saturation = 1.0 - (y / height)  # Top = saturated, bottom = desaturated
        
        for x in range(width):
            hue = (x / width) * 360  # Full hue circle
            value = 0.7  # Constant brightness
            
            rgb = hsv_to_rgb([hue, saturation, value])
            img[y, x] = rgb
    
    return img

def generate_hue_luminance_chart(width=512, height=512):
    """
    Generate a chart showing all hues (horizontal) and luminance (vertical)
    at constant saturation
    """
    img = np.zeros((height, width, 3))
    
    for y in range(height):
        value = 1.0 - (y / height)  # Top = bright, bottom = dark
        
        for x in range(width):
            hue = (x / width) * 360  # Full hue circle
            saturation = 0.8  # Constant saturation
            
            rgb = hsv_to_rgb([hue, saturation, value])
            img[y, x] = rgb
    
    return img

def generate_macbeth_chart():
    """
    Generate a simplified Macbeth ColorChecker-style chart
    Good for seeing realistic color shifts
    """
    # Approximate Macbeth chart colors in linear RGB
    colors = [
        # Row 1
        [0.44, 0.31, 0.24],  # Dark skin
        [0.77, 0.57, 0.48],  # Light skin
        [0.36, 0.45, 0.60],  # Blue sky
        [0.33, 0.42, 0.27],  # Foliage
        [0.54, 0.51, 0.73],  # Blue flower
        [0.47, 0.75, 0.67],  # Bluish green
        # Row 2
        [0.93, 0.60, 0.12],  # Orange
        [0.36, 0.40, 0.70],  # Purplish blue
        [0.76, 0.37, 0.42],  # Moderate red
        [0.36, 0.22, 0.43],  # Purple
        [0.64, 0.83, 0.36],  # Yellow green
        [0.98, 0.73, 0.20],  # Orange yellow
        # Row 3
        [0.27, 0.29, 0.75],  # Blue
        [0.41, 0.71, 0.43],  # Green
        [0.75, 0.24, 0.26],  # Red
        [0.96, 0.92, 0.20],  # Yellow
        [0.82, 0.38, 0.70],  # Magenta
        [0.17, 0.62, 0.73],  # Cyan
        # Row 4 (Grayscale)
        [0.95, 0.95, 0.95],  # White
        [0.78, 0.78, 0.78],  # Gray 1
        [0.63, 0.63, 0.63],  # Gray 2
        [0.47, 0.47, 0.47],  # Gray 3
        [0.31, 0.31, 0.31],  # Gray 4
        [0.12, 0.12, 0.12],  # Black
    ]
    
    # Create 4x6 grid
    patch_size = 64
    img = np.zeros((4 * patch_size, 6 * patch_size, 3))
    
    for i, color in enumerate(colors):
        row = i // 6
        col = i % 6
        y_start = row * patch_size
        y_end = (row + 1) * patch_size
        x_start = col * patch_size
        x_end = (col + 1) * patch_size
        
        img[y_start:y_end, x_start:x_end] = color
    
    return img

# ============================================================================
# PART 8: Before/After Comparison Visualization
# ============================================================================

def visualize_transformation_comparison(pipeline, film_preset="fuji_velvia"):
    """
    Create comprehensive before/after visualization
    """
    pipeline.set_film_preset(film_preset)
    
    # Generate test images
    hue_sat_chart = generate_hue_saturation_chart(512, 512)
    hue_lum_chart = generate_hue_luminance_chart(512, 512)
    macbeth = generate_macbeth_chart()
    
    # Process images through pipeline
    hue_sat_processed = np.zeros_like(hue_sat_chart)
    for y in range(hue_sat_chart.shape[0]):
        for x in range(hue_sat_chart.shape[1]):
            hue_sat_processed[y, x] = pipeline.process(hue_sat_chart[y, x])
    
    hue_lum_processed = np.zeros_like(hue_lum_chart)
    for y in range(hue_lum_chart.shape[0]):
        for x in range(hue_lum_chart.shape[1]):
            hue_lum_processed[y, x] = pipeline.process(hue_lum_chart[y, x])
    
    macbeth_processed = np.zeros_like(macbeth)
    for y in range(macbeth.shape[0]):
        for x in range(macbeth.shape[1]):
            macbeth_processed[y, x] = pipeline.process(macbeth[y, x])
    
    # Create figure with all comparisons
    fig = plt.figure(figsize=(18, 12))
    
    # Hue-Saturation comparison
    ax1 = plt.subplot(3, 2, 1)
    ax1.imshow(np.clip(hue_sat_chart, 0, 1))
    ax1.set_title('Original: Hue (H) vs Saturation (V)\nH: 0°→360° | S: 100%→0%', 
                  fontsize=11, pad=10)
    ax1.axis('off')
    
    ax2 = plt.subplot(3, 2, 2)
    ax2.imshow(np.clip(hue_sat_processed, 0, 1))
    ax2.set_title(f'After {film_preset.replace("_", " ").title()}: Hue vs Saturation\n' + 
                  'Notice color shifts and saturation changes',
                  fontsize=11, pad=10)
    ax2.axis('off')
    
    # Hue-Luminance comparison
    ax3 = plt.subplot(3, 2, 3)
    ax3.imshow(np.clip(hue_lum_chart, 0, 1))
    ax3.set_title('Original: Hue (H) vs Luminance (V)\nH: 0°→360° | L: 100%→0%',
                  fontsize=11, pad=10)
    ax3.axis('off')
    
    ax4 = plt.subplot(3, 2, 4)
    ax4.imshow(np.clip(hue_lum_processed, 0, 1))
    ax4.set_title(f'After {film_preset.replace("_", " ").title()}: Hue vs Luminance\n' +
                  'See how different hues respond to the grade',
                  fontsize=11, pad=10)
    ax4.axis('off')
    
    # Macbeth chart comparison
    ax5 = plt.subplot(3, 2, 5)
    ax5.imshow(np.clip(macbeth, 0, 1))
    ax5.set_title('Original: Macbeth ColorChecker\nRealistic color patches',
                  fontsize=11, pad=10)
    ax5.axis('off')
    
    ax6 = plt.subplot(3, 2, 6)
    ax6.imshow(np.clip(macbeth_processed, 0, 1))
    ax6.set_title(f'After {film_preset.replace("_", " ").title()}: Macbeth ColorChecker\n' +
                  'Skin tones, primaries, and neutrals',
                  fontsize=11, pad=10)
    ax6.axis('off')
    
    plt.tight_layout()
    return fig

def plot_hue_shift_vector_field(pipeline):
    """
    Create a vector field showing how hues are shifted
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Create a polar-like visualization
    n_hues = 36
    n_sats = 5
    
    for sat_idx in range(n_sats):
        saturation = (sat_idx + 1) / n_sats
        radius = saturation
        
        input_angles = []
        output_angles = []
        
        for hue_idx in range(n_hues):
            hue = (hue_idx / n_hues) * 360
            
            # Create color at this hue and saturation
            rgb_in = hsv_to_rgb([hue, saturation, 0.7])
            rgb_out = pipeline.process(rgb_in)
            
            hsv_out = rgb_to_hsv(rgb_out)
            
            input_angle = np.radians(hue)
            output_angle = np.radians(hsv_out[0])
            
            # Plot input position
            x_in = radius * np.cos(input_angle)
            y_in = radius * np.sin(input_angle)
            
            # Plot output position
            x_out = radius * np.cos(output_angle)
            y_out = radius * np.sin(output_angle)
            
            # Draw arrow from input to output
            ax.arrow(x_in, y_in, x_out - x_in, y_out - y_in,
                    head_width=0.03, head_length=0.02, fc='red', ec='red',
                    alpha=0.5, width=0.005)
            
            # Mark input point
            ax.plot(x_in, y_in, 'o', color=np.clip(rgb_in, 0, 1), 
                   markersize=8, markeredgecolor='black', markeredgewidth=0.5)
    
    # Add hue labels
    hue_labels = ['Red\n0°', 'Yellow\n60°', 'Green\n120°', 
                  'Cyan\n180°', 'Blue\n240°', 'Magenta\n300°']
    for i, label in enumerate(hue_labels):
        angle = np.radians(i * 60)
        x = 1.2 * np.cos(angle)
        y = 1.2 * np.sin(angle)
        ax.text(x, y, label, ha='center', va='center', fontsize=10,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_xlim(-1.4, 1.4)
    ax.set_ylim(-1.4, 1.4)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_title('Hue Shift Vector Field\n' +
                 'Arrows show how each hue is shifted\n' +
                 'Concentric rings = different saturation levels',
                 fontsize=12, pad=15)
    ax.set_xlabel('Hue wheel representation')
    
    return fig

# ============================================================================
# PART 9: Example Usage and Testing
# ============================================================================

def example_workflow():
    """Demonstrate the complete workflow with N-Log to DWG"""
    
    print("=" * 70)
    print("N-Log to DaVinci Wide Gamut + Film Emulation Demo")
    print("=" * 70)
    
    # Create pipeline
    pipeline = FilmEmulationPipeline()
    
    # Test N-Log conversion
    print("\n1. Testing N-Log to DWG/Log Conversion:")
    print("-" * 70)
    test_nlog_values = [0, 1024, 2048, 3072, 4095]  # 12-bit values
    for nlog_val in test_nlog_values:
        linear = nlog_to_linear(nlog_val)
        dwg_log = linear_to_dwg_log(linear)
        print(f"N-Log 12-bit: {nlog_val:4d} → Linear: {linear:.4f} → DWG/Log: {dwg_log:.4f}")
    
    # Generate comparison visualizations for different film stocks
    print("\n2. Generating Comparison Visualizations...")
    print("-" * 70)
    
    # Fuji Velvia
    print("Creating Fuji Velvia comparison...")
    fig1 = visualize_transformation_comparison(pipeline, "fuji_velvia")
    plt.savefig('comparison_fuji_velvia.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: comparison_fuji_velvia.png")
    
    # Kodak Portra
    print("Creating Kodak Portra comparison...")
    fig2 = visualize_transformation_comparison(pipeline, "kodak_portra")
    plt.savefig('comparison_kodak_portra.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: comparison_kodak_portra.png")
    
    # Kodak Ektar
    print("Creating Kodak Ektar comparison...")
    fig3 = visualize_transformation_comparison(pipeline, "kodak_ektar")
    plt.savefig('comparison_kodak_ektar.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: comparison_kodak_ektar.png")
    
    # Hue shift vector fields
    print("\nCreating hue shift vector field for Velvia...")
    pipeline.set_film_preset("fuji_velvia")
    fig4 = plot_hue_shift_vector_field(pipeline)
    plt.savefig('hue_vector_field_velvia.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: hue_vector_field_velvia.png")
    
    print("\nCreating hue shift vector field for Portra...")
    pipeline.set_film_preset("kodak_portra")
    fig5 = plot_hue_shift_vector_field(pipeline)
    plt.savefig('hue_vector_field_portra.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: hue_vector_field_portra.png")
    
    # Original visualizations
    print("\nCreating hue rotation demo...")
    pipeline.set_film_preset("fuji_velvia")
    fig6 = plot_hue_rotation_demo(pipeline)
    plt.savefig('hue_rotation_velvia.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: hue_rotation_velvia.png")
    
    plt.show()
    
    print("\n" + "=" * 70)
    print("Demo complete!")
    print("\nKey insights from the visualizations:")
    print("- Hue vs Saturation chart: See how saturation changes affect colors")
    print("- Hue vs Luminance chart: See how brightness affects color rendering")
    print("- Macbeth chart: Real-world colors like skin tones and primaries")
    print("- Vector field: Shows exact hue shifts as arrows on a color wheel")
    print("=" * 70)

