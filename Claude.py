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
# PART 6: Example Usage and Testing
# ============================================================================

def example_workflow():
    """Demonstrate the complete workflow"""
    
    print("=" * 70)
    print("DaVinci Wide Gamut Film Emulation Demo")
    print("=" * 70)
    
    # Create pipeline
    pipeline = FilmEmulationPipeline()
    
    # Test with a few colors
    test_colors = {
        "Neutral Gray": [0.5, 0.5, 0.5],
        "Red": [1.0, 0.0, 0.0],
        "Green": [0.0, 1.0, 0.0],
        "Blue": [0.0, 0.0, 1.0],
        "Skin Tone": [0.8, 0.5, 0.4],
        "Sky Blue": [0.3, 0.6, 0.9]
    }
    
    print("\n1. Testing Neutral Pipeline:")
    print("-" * 70)
    for name, rgb in test_colors.items():
        output = pipeline.process(rgb)
        print(f"{name:15} Input: {rgb}  →  Output: {np.round(output, 3)}")
    
    print("\n2. Testing Fuji Velvia Emulation:")
    print("-" * 70)
    pipeline.set_film_preset("fuji_velvia")
    for name, rgb in test_colors.items():
        output = pipeline.process(rgb)
        print(f"{name:15} Input: {rgb}  →  Output: {np.round(output, 3)}")
    
    print("\n3. Testing Kodak Portra Emulation:")
    print("-" * 70)
    pipeline.set_film_preset("kodak_portra")
    for name, rgb in test_colors.items():
        output = pipeline.process(rgb)
        print(f"{name:15} Input: {rgb}  →  Output: {np.round(output, 3)}")
    
    # Generate visualizations
    print("\n4. Generating Visualizations...")
    print("-" * 70)
    
    # Plot gamut comparison
    fig1 = plot_color_gamut_2d()
    plt.savefig('gamut_comparison.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: gamut_comparison.png")
    
    # Plot hue rotation for Velvia
    pipeline.set_film_preset("fuji_velvia")
    fig2 = plot_hue_rotation_demo(pipeline)
    plt.savefig('hue_rotation_velvia.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: hue_rotation_velvia.png")
    
    # Plot 3D color cube transformation
    fig3 = plot_3d_color_cube(pipeline, resolution=6)
    plt.savefig('color_cube_transform.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: color_cube_transform.png")
    
    plt.show()
    
    print("\n" + "=" * 70)
    print("Demo complete! Experiment with the pipeline parameters above.")
    print("=" * 70)

# Run the example
if __name__ == "__main__":
    example_workflow()
    
    # Interactive experimentation
    print("\n\nInteractive Mode:")
    print("Try creating your own pipeline:")
    print("  pipeline = FilmEmulationPipeline()")
    print("  pipeline.saturation = 1.5")
    print("  pipeline.hue_shift = 15")
    print("  result = pipeline.process([0.8, 0.5, 0.4])  # skin tone")