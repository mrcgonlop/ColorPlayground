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

def cubic_spline_interpolate(x, points):
    """
    Cubic spline interpolation for smooth curves
    points: list of (x, y) tuples
    Returns: interpolated y value at x
    """
    points = sorted(points, key=lambda p: p[0])
    
    # Handle edge cases
    if x <= points[0][0]:
        return points[0][1]
    if x >= points[-1][0]:
        return points[-1][1]
    
    # Find the segment
    for i in range(len(points) - 1):
        x1, y1 = points[i]
        x2, y2 = points[i + 1]
        
        if x1 <= x <= x2:
            # Catmull-Rom spline for smooth interpolation
            t = (x - x1) / (x2 - x1)
            
            # Get neighboring points for tangent calculation
            x0, y0 = points[max(0, i - 1)]
            x3, y3 = points[min(len(points) - 1, i + 2)]
            
            # Calculate tangents
            m1 = (y2 - y0) / (x2 - x0) if i > 0 else (y2 - y1) / (x2 - x1)
            m2 = (y3 - y1) / (x3 - x1) if i < len(points) - 2 else (y2 - y1) / (x2 - x1)
            
            # Hermite interpolation
            t2 = t * t
            t3 = t2 * t
            h00 = 2*t3 - 3*t2 + 1
            h10 = t3 - 2*t2 + t
            h01 = -2*t3 + 3*t2
            h11 = t3 - t2
            
            return h00*y1 + h10*(x2-x1)*m1 + h01*y2 + h11*(x2-x1)*m2
    
    return points[-1][1]

def apply_hue_vs_hue_curve(rgb, curve_points):
    """
    Apply hue-dependent hue shift with smooth cubic interpolation
    curve_points: list of (input_hue, output_hue_shift) tuples
    This is how film creates signature color responses
    """
    hsv = rgb_to_hsv(rgb)
    input_hue = hsv[0]
    
    # Use cubic spline for smooth interpolation
    shift = cubic_spline_interpolate(input_hue, curve_points)
    
    hsv[0] = (hsv[0] + shift) % 360
    return hsv_to_rgb(hsv)

def apply_saturation_adjustment(rgb, saturation_factor):
    """Adjust color saturation"""
    hsv = rgb_to_hsv(rgb)
    hsv[1] = np.clip(hsv[1] * saturation_factor, 0, 1)
    return hsv_to_rgb(hsv)

# ============================================================================
# PART 3B: ADVANCED PARAMETRIC COLOR CONTROL SYSTEM
# ============================================================================

class ParametricColorTransform:
    """
    A comprehensive, parametric system for controlling color transformations
    inspired by film response curves and professional color grading tools.
    
    This system provides multi-dimensional control over:
    1. Hue → Hue (color rotation/warping)
    2. Hue → Saturation (how hue affects colorfulness)
    3. Hue → Luminance (how hue affects brightness)
    4. Saturation → Saturation (saturation compression/expansion)
    5. Luminance → Saturation (how brightness affects colorfulness)
    6. Luminance → Luminance (tone curve/contrast)
    
    Each dimension uses smooth cubic spline interpolation through control points.
    """
    
    def __init__(self):
        # HUE TRANSFORMS
        # Hue vs Hue: (input_hue, output_hue_shift)
        # This controls color "warping" - shifts specific colors
        self.hue_vs_hue = [
            (0, 0), (60, 0), (120, 0), (180, 0), (240, 0), (300, 0), (360, 0)
        ]
        
        # Hue vs Saturation: (input_hue, saturation_multiplier)
        # Makes certain hues more/less saturated (e.g., boost blue saturation)
        self.hue_vs_sat = [
            (0, 1.0), (60, 1.0), (120, 1.0), (180, 1.0), (240, 1.0), (300, 1.0), (360, 1.0)
        ]
        
        # Hue vs Luminance: (input_hue, luminance_offset)
        # Makes certain hues brighter/darker (e.g., yellows naturally brighter)
        self.hue_vs_lum = [
            (0, 0.0), (60, 0.0), (120, 0.0), (180, 0.0), (240, 0.0), (300, 0.0), (360, 0.0)
        ]
        
        # SATURATION TRANSFORMS
        # Saturation vs Saturation: (input_sat, output_sat_multiplier)
        # Non-linear saturation response (compression in highlights, etc.)
        self.sat_vs_sat = [
            (0.0, 1.0), (0.25, 1.0), (0.5, 1.0), (0.75, 1.0), (1.0, 1.0)
        ]
        
        # Luminance vs Saturation: (input_lum, saturation_multiplier)
        # Controls how brightness affects color (desaturate shadows/highlights)
        self.lum_vs_sat = [
            (0.0, 1.0), (0.25, 1.0), (0.5, 1.0), (0.75, 1.0), (1.0, 1.0)
        ]
        
        # LUMINANCE TRANSFORMS
        # Luminance vs Luminance: (input_lum, output_lum)
        # Traditional tone curve / contrast control
        self.lum_vs_lum = [
            (0.0, 0.0), (0.25, 0.25), (0.5, 0.5), (0.75, 0.75), (1.0, 1.0)
        ]
        
        # CHANNEL CROSSTALK (Advanced)
        # Simulates how film dyes interact - R affects G, B affects R, etc.
        # Format: {'target_channel': [(other_channel, influence_curve)]}
        self.enable_crosstalk = False
        self.crosstalk_strength = 0.1
        
        # GLOBAL MODIFIERS
        self.global_hue_shift = 0.0  # Degrees
        self.global_saturation = 1.0  # Multiplier
        self.global_contrast = 1.0    # Pivot around 0.5
        self.global_exposure = 1.0    # Multiplier
        
    def apply_hue_transforms(self, hsv):
        """Apply all hue-based transformations"""
        h, s, v = hsv
        
        # Hue vs Hue (color warping)
        hue_shift = cubic_spline_interpolate(h, self.hue_vs_hue)
        h_shifted = (h + hue_shift + self.global_hue_shift) % 360
        
        # Hue vs Saturation (hue-dependent saturation)
        sat_mult = cubic_spline_interpolate(h, self.hue_vs_sat)
        s_adjusted = s * sat_mult
        
        # Hue vs Luminance (hue-dependent brightness)
        lum_offset = cubic_spline_interpolate(h, self.hue_vs_lum)
        v_adjusted = v + lum_offset
        
        return np.array([h_shifted, s_adjusted, v_adjusted])
    
    def apply_saturation_transforms(self, hsv):
        """Apply saturation-based transformations"""
        h, s, v = hsv
        
        # Saturation vs Saturation (non-linear saturation response)
        sat_mult = cubic_spline_interpolate(s, self.sat_vs_sat)
        s_adjusted = np.clip(s * sat_mult * self.global_saturation, 0, 1)
        
        return np.array([h, s_adjusted, v])
    
    def apply_luminance_transforms(self, hsv):
        """Apply luminance-based transformations"""
        h, s, v = hsv
        
        # Luminance vs Saturation (brightness-dependent saturation)
        sat_mult = cubic_spline_interpolate(v, self.lum_vs_sat)
        s_adjusted = s * sat_mult
        
        # Luminance vs Luminance (tone curve)
        v_mapped = cubic_spline_interpolate(v, self.lum_vs_lum)
        
        # Apply global contrast around midpoint
        v_contrasted = 0.5 + (v_mapped - 0.5) * self.global_contrast
        
        # Apply global exposure
        v_final = np.clip(v_contrasted * self.global_exposure, 0, 1)
        
        return np.array([h, s_adjusted, v_final])
    
    def apply_channel_crosstalk(self, rgb):
        """
        Simulate film dye crosstalk - how one color channel affects others
        This is what creates the "film look" - dyes aren't perfectly independent
        """
        if not self.enable_crosstalk:
            return rgb
        
        r, g, b = rgb
        
        # Example crosstalk matrix (can be parameterized further)
        # Red dye absorbs some green and blue
        # Green dye absorbs some red and blue
        # Blue dye absorbs some red and green
        crosstalk_matrix = np.array([
            [1.0, -self.crosstalk_strength * 0.1, -self.crosstalk_strength * 0.05],
            [-self.crosstalk_strength * 0.05, 1.0, -self.crosstalk_strength * 0.1],
            [-self.crosstalk_strength * 0.08, -self.crosstalk_strength * 0.08, 1.0]
        ])
        
        rgb_crosstalk = np.dot(crosstalk_matrix, rgb)
        return np.clip(rgb_crosstalk, 0, 1)
    
    def transform(self, rgb):
        """
        Apply the complete parametric transformation
        
        Order of operations:
        1. RGB → HSV conversion
        2. Hue-based transforms (warping, hue-dependent sat/lum)
        3. Saturation transforms (non-linear saturation)
        4. Luminance transforms (tone curve, lum-dependent saturation)
        5. HSV → RGB conversion
        6. Channel crosstalk (optional)
        7. Final clipping
        """
        # Convert to HSV
        hsv = rgb_to_hsv(rgb)
        
        # Apply transformations in order
        hsv = self.apply_hue_transforms(hsv)
        hsv = self.apply_saturation_transforms(hsv)
        hsv = self.apply_luminance_transforms(hsv)
        
        # Ensure HSV is valid
        hsv[0] = hsv[0] % 360  # Hue wraps
        hsv[1] = np.clip(hsv[1], 0, 1)  # Saturation clamps
        hsv[2] = np.clip(hsv[2], 0, 1)  # Value clamps
        
        # Convert back to RGB
        rgb_out = hsv_to_rgb(hsv)
        
        # Apply channel crosstalk
        rgb_out = self.apply_channel_crosstalk(rgb_out)
        
        return np.clip(rgb_out, 0, 1)
    
    def load_preset(self, preset_name):
        """Load a preset film emulation"""
        if preset_name == "fuji_velvia_parametric":
            # Ultra-saturated, vibrant greens and blues
            self.hue_vs_hue = [
                (0, 5), (30, 8), (60, 10), (90, 5), (120, -5),
                (150, -3), (180, 0), (210, 3), (240, 5), 
                (270, 3), (300, 0), (330, 3), (360, 5)
            ]
            self.hue_vs_sat = [
                (0, 1.2), (60, 1.4), (120, 1.5), (180, 1.3),
                (240, 1.4), (300, 1.1), (360, 1.2)
            ]
            self.hue_vs_lum = [
                (0, 0.0), (60, 0.05), (120, 0.0), (180, -0.02),
                (240, -0.03), (300, 0.0), (360, 0.0)
            ]
            self.sat_vs_sat = [
                (0.0, 1.0), (0.3, 1.3), (0.6, 1.4), (0.9, 1.2), (1.0, 1.0)
            ]
            self.lum_vs_sat = [
                (0.0, 0.7), (0.2, 1.0), (0.5, 1.2), (0.8, 1.1), (1.0, 0.9)
            ]
            self.lum_vs_lum = [
                (0.0, 0.0), (0.1, 0.15), (0.3, 0.35), (0.5, 0.5),
                (0.7, 0.68), (0.9, 0.88), (1.0, 1.0)
            ]
            self.global_saturation = 1.2
            self.global_contrast = 1.15
            
        elif preset_name == "kodak_portra_parametric":
            # Soft, flattering skin tones, muted colors
            self.hue_vs_hue = [
                (0, -3), (15, 0), (30, 5), (45, 8), (60, 5),
                (120, 0), (180, -5), (240, -5), (300, 0), (360, -3)
            ]
            self.hue_vs_sat = [
                (0, 0.9), (30, 1.0), (60, 0.95), (120, 0.85),
                (180, 0.9), (240, 0.85), (300, 0.9), (360, 0.9)
            ]
            self.hue_vs_lum = [
                (0, 0.0), (30, 0.03), (60, 0.02), (120, 0.0),
                (180, 0.0), (240, 0.0), (300, 0.0), (360, 0.0)
            ]
            self.sat_vs_sat = [
                (0.0, 1.0), (0.3, 0.95), (0.6, 0.9), (0.9, 0.85), (1.0, 0.8)
            ]
            self.lum_vs_sat = [
                (0.0, 0.6), (0.2, 0.85), (0.5, 1.0), (0.8, 0.95), (1.0, 0.8)
            ]
            self.lum_vs_lum = [
                (0.0, 0.02), (0.1, 0.12), (0.3, 0.32), (0.5, 0.5),
                (0.7, 0.7), (0.9, 0.92), (1.0, 0.98)
            ]
            self.global_saturation = 0.9
            self.global_contrast = 0.95
            self.enable_crosstalk = True
            self.crosstalk_strength = 0.15
            
        elif preset_name == "cinestill_800t_parametric":
            # Tungsten-balanced, halation, cool shadows
            self.hue_vs_hue = [
                (0, 10), (30, 15), (60, 10), (120, -5),
                (180, -10), (240, -8), (300, 5), (360, 10)
            ]
            self.hue_vs_sat = [
                (0, 1.2), (60, 1.1), (120, 0.95), (180, 1.1),
                (240, 1.3), (300, 1.15), (360, 1.2)
            ]
            self.hue_vs_lum = [
                (0, 0.05), (60, 0.03), (120, 0.0), (180, -0.05),
                (240, -0.03), (300, 0.0), (360, 0.05)
            ]
            self.sat_vs_sat = [
                (0.0, 1.0), (0.25, 1.1), (0.5, 1.2), (0.75, 1.15), (1.0, 1.0)
            ]
            self.lum_vs_sat = [
                (0.0, 0.8), (0.2, 1.0), (0.5, 1.1), (0.8, 1.2), (1.0, 1.3)
            ]
            self.lum_vs_lum = [
                (0.0, 0.0), (0.15, 0.18), (0.35, 0.4), (0.5, 0.52),
                (0.7, 0.73), (0.85, 0.9), (1.0, 1.0)
            ]
            self.global_saturation = 1.05
            self.global_contrast = 1.1
            
        elif preset_name == "agfa_vista_parametric":
            # Warm, nostalgic, green shift
            self.hue_vs_hue = [
                (0, 8), (60, 12), (120, 5), (180, -5),
                (240, -3), (300, 5), (360, 8)
            ]
            self.hue_vs_sat = [
                (0, 1.1), (60, 1.25), (120, 1.2), (180, 1.0),
                (240, 0.95), (300, 1.05), (360, 1.1)
            ]
            self.sat_vs_sat = [
                (0.0, 1.0), (0.4, 1.15), (0.7, 1.1), (1.0, 0.95)
            ]
            self.lum_vs_sat = [
                (0.0, 0.75), (0.3, 1.05), (0.6, 1.1), (1.0, 0.9)
            ]
            self.global_saturation = 1.1
            self.enable_crosstalk = True
            
    def visualize_all_curves(self):
        """Generate a comprehensive visualization of all parametric curves"""
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        
        # Hue vs Hue
        ax = axes[0, 0]
        hues = np.linspace(0, 360, 360)
        shifts = [cubic_spline_interpolate(h, self.hue_vs_hue) for h in hues]
        ax.plot(hues, shifts, 'b-', linewidth=2)
        for h, s in self.hue_vs_hue:
            ax.plot(h, s, 'ro', markersize=8)
        ax.axhline(0, color='k', linestyle='--', alpha=0.3)
        ax.set_xlabel('Input Hue (degrees)')
        ax.set_ylabel('Hue Shift (degrees)')
        ax.set_title('Hue vs Hue Transform')
        ax.grid(True, alpha=0.3)
        
        # Hue vs Saturation
        ax = axes[0, 1]
        sat_mults = [cubic_spline_interpolate(h, self.hue_vs_sat) for h in hues]
        ax.plot(hues, sat_mults, 'g-', linewidth=2)
        for h, s in self.hue_vs_sat:
            ax.plot(h, s, 'ro', markersize=8)
        ax.axhline(1.0, color='k', linestyle='--', alpha=0.3)
        ax.set_xlabel('Input Hue (degrees)')
        ax.set_ylabel('Saturation Multiplier')
        ax.set_title('Hue vs Saturation Transform')
        ax.grid(True, alpha=0.3)
        
        # Hue vs Luminance
        ax = axes[1, 0]
        lum_offsets = [cubic_spline_interpolate(h, self.hue_vs_lum) for h in hues]
        ax.plot(hues, lum_offsets, 'm-', linewidth=2)
        for h, l in self.hue_vs_lum:
            ax.plot(h, l, 'ro', markersize=8)
        ax.axhline(0, color='k', linestyle='--', alpha=0.3)
        ax.set_xlabel('Input Hue (degrees)')
        ax.set_ylabel('Luminance Offset')
        ax.set_title('Hue vs Luminance Transform')
        ax.grid(True, alpha=0.3)
        
        # Saturation vs Saturation
        ax = axes[1, 1]
        sats = np.linspace(0, 1, 100)
        sat_out = [cubic_spline_interpolate(s, self.sat_vs_sat) * s for s in sats]
        ax.plot(sats, sat_out, 'c-', linewidth=2)
        for s_in, s_mult in self.sat_vs_sat:
            ax.plot(s_in, s_in * s_mult, 'ro', markersize=8)
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Unity')
        ax.set_xlabel('Input Saturation')
        ax.set_ylabel('Output Saturation')
        ax.set_title('Saturation vs Saturation Transform')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Luminance vs Saturation
        ax = axes[2, 0]
        lums = np.linspace(0, 1, 100)
        sat_mults_lum = [cubic_spline_interpolate(l, self.lum_vs_sat) for l in lums]
        ax.plot(lums, sat_mults_lum, 'y-', linewidth=2)
        for l, s in self.lum_vs_sat:
            ax.plot(l, s, 'ro', markersize=8)
        ax.axhline(1.0, color='k', linestyle='--', alpha=0.3)
        ax.set_xlabel('Input Luminance')
        ax.set_ylabel('Saturation Multiplier')
        ax.set_title('Luminance vs Saturation Transform')
        ax.grid(True, alpha=0.3)
        
        # Luminance vs Luminance (Tone Curve)
        ax = axes[2, 1]
        lum_out = [cubic_spline_interpolate(l, self.lum_vs_lum) for l in lums]
        ax.plot(lums, lum_out, 'r-', linewidth=2)
        for l_in, l_out in self.lum_vs_lum:
            ax.plot(l_in, l_out, 'ro', markersize=8)
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Unity')
        ax.set_xlabel('Input Luminance')
        ax.set_ylabel('Output Luminance')
        ax.set_title('Luminance vs Luminance (Tone Curve)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig

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

def visualize_parametric_transformation(transform, film_name="Custom"):
    """
    Visualize the effects of a parametric transform on test charts
    """
    # Generate test images
    hue_sat_chart = generate_hue_saturation_chart(512, 512)
    hue_lum_chart = generate_hue_luminance_chart(512, 512)
    macbeth = generate_macbeth_chart()
    
    # Process images through parametric transform
    hue_sat_processed = np.zeros_like(hue_sat_chart)
    for y in range(hue_sat_chart.shape[0]):
        for x in range(hue_sat_chart.shape[1]):
            hue_sat_processed[y, x] = transform.transform(hue_sat_chart[y, x])
    
    hue_lum_processed = np.zeros_like(hue_lum_chart)
    for y in range(hue_lum_chart.shape[0]):
        for x in range(hue_lum_chart.shape[1]):
            hue_lum_processed[y, x] = transform.transform(hue_lum_chart[y, x])
    
    macbeth_processed = np.zeros_like(macbeth)
    for y in range(macbeth.shape[0]):
        for x in range(macbeth.shape[1]):
            macbeth_processed[y, x] = transform.transform(macbeth[y, x])
    
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
    ax2.set_title(f'After {film_name}: Hue vs Saturation\n' + 
                  'Notice hue-dependent saturation changes',
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
    ax4.set_title(f'After {film_name}: Hue vs Luminance\n' +
                  'See hue warping and luminance-dependent saturation',
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
    ax6.set_title(f'After {film_name}: Macbeth ColorChecker\n' +
                  'Complete parametric transformation',
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
    """Demonstrate the complete parametric workflow"""
    
    print("=" * 80)
    print("PARAMETRIC COLOR TRANSFORM SYSTEM - Advanced Film Emulation")
    print("=" * 80)
    
    print("\n" + "=" * 80)
    print("CONCEPTUAL OVERVIEW")
    print("=" * 80)
    print("""
This system provides a comprehensive parametric approach to color transformation
inspired by film emulation and professional color grading tools.

KEY CONCEPTS:

1. MULTI-DIMENSIONAL CONTROL
   - Hue → Hue: Color warping/rotation (reds shift orange, blues shift cyan)
   - Hue → Saturation: Some hues more saturated (vibrant greens)
   - Hue → Luminance: Some hues brighter/darker (yellows naturally lighter)
   - Saturation → Saturation: Non-linear saturation response
   - Luminance → Saturation: Desaturate shadows/highlights
   - Luminance → Luminance: Traditional tone curve/contrast

2. CUBIC SPLINE INTERPOLATION
   - Control points define key positions
   - Smooth curves between points (no harsh transitions)
   - Catmull-Rom splines ensure C1 continuity

3. CHANNEL CROSSTALK (Advanced)
   - Simulates how film dyes interact
   - Red dye isn't pure red - absorbs some green/blue
   - Creates the "organic" film look

4. ORDER OF OPERATIONS
   The sequence matters! Film emulation typically:
   a) Convert RGB → HSV (separate hue/sat/luminance)
   b) Apply hue warping first (changes color identity)
   c) Apply hue-dependent saturation/luminance
   d) Apply saturation transforms
   e) Apply luminance-dependent saturation
   f) Apply tone curve (luminance mapping)
   g) Convert back to RGB
   h) Apply channel crosstalk (optional)

This approach gives colorists surgical control over specific color regions
while maintaining smooth, film-like transitions.
    """)
    
    # Create parametric transform
    print("\n" + "=" * 80)
    print("DEMONSTRATION: Parametric Transform Presets")
    print("=" * 80)
    
    transform = ParametricColorTransform()
    
    # Test colors
    test_colors = {
        "Neutral Gray": [0.5, 0.5, 0.5],
        "Red": [1.0, 0.0, 0.0],
        "Orange": [1.0, 0.5, 0.0],
        "Yellow": [1.0, 1.0, 0.0],
        "Green": [0.0, 1.0, 0.0],
        "Cyan": [0.0, 1.0, 1.0],
        "Blue": [0.0, 0.0, 1.0],
        "Magenta": [1.0, 0.0, 1.0],
        "Skin Tone": [0.8, 0.5, 0.4],
        "Sky Blue": [0.3, 0.6, 0.9]
    }
    
    # Test different presets
    presets = [
        ("fuji_velvia_parametric", "Fuji Velvia"),
        ("kodak_portra_parametric", "Kodak Portra"),
        ("cinestill_800t_parametric", "Cinestill 800T"),
        ("agfa_vista_parametric", "Agfa Vista")
    ]
    
    for preset_id, preset_name in presets:
        print(f"\n{preset_name}:")
        print("-" * 80)
        transform.load_preset(preset_id)
        
        for color_name, rgb in test_colors.items():
            output = transform.transform(np.array(rgb))
            hsv_in = rgb_to_hsv(np.array(rgb))
            hsv_out = rgb_to_hsv(output)
            print(f"{color_name:15} | H:{hsv_in[0]:6.1f}°→{hsv_out[0]:6.1f}° | "
                  f"S:{hsv_in[1]:.2f}→{hsv_out[1]:.2f} | V:{hsv_in[2]:.2f}→{hsv_out[2]:.2f}")
    
    # Generate visualizations
    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATIONS")
    print("=" * 80)
    
    # 1. Curve visualizations for each preset
    for preset_id, preset_name in presets:
        print(f"\nGenerating curve visualization for {preset_name}...")
        transform.load_preset(preset_id)
        fig = transform.visualize_all_curves()
        filename = f"parametric_curves_{preset_id}.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {filename}")
        plt.close()
    
    # 2. Before/after comparisons
    for preset_id, preset_name in presets:
        print(f"\nGenerating before/after comparison for {preset_name}...")
        transform.load_preset(preset_id)
        fig = visualize_parametric_transformation(transform, preset_name)
        filename = f"parametric_comparison_{preset_id}.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {filename}")
        plt.close()
    
    # 3. Custom transform example
    print("\n" + "=" * 80)
    print("CUSTOM TRANSFORM EXAMPLE")
    print("=" * 80)
    print("Creating a custom 'Teal & Orange' look...")
    
    transform = ParametricColorTransform()
    
    # Teal & Orange blockbuster look
    transform.hue_vs_hue = [
        (0, 15),    # Push reds toward orange
        (30, 20),   # Strong orange shift
        (60, 10),   # Yellows slightly orange
        (120, 0),   # Greens neutral
        (180, -15), # Cyans toward teal
        (200, -20), # Strong teal shift
        (240, -10), # Blues slightly teal
        (300, 5),   # Magentas toward red
        (360, 15)
    ]
    
    transform.hue_vs_sat = [
        (0, 1.3),   # Saturated oranges
        (30, 1.4),  # Very saturated oranges
        (60, 1.2),  # Yellows moderately saturated
        (120, 0.9), # Desaturated greens
        (180, 1.4), # Very saturated teals
        (200, 1.5), # Maximum teal saturation
        (240, 1.2), # Saturated blues
        (300, 1.0), # Neutral magentas
        (360, 1.3)
    ]
    
    transform.lum_vs_sat = [
        (0.0, 0.7),  # Desaturate deep shadows
        (0.3, 1.1),  # Boost mid-tone saturation
        (0.6, 1.2),  # High saturation in upper mids
        (0.9, 0.9),  # Reduce saturation in highlights
        (1.0, 0.7)   # Desaturate peak whites
    ]
    
    transform.lum_vs_lum = [
        (0.0, 0.05),  # Lifted blacks
        (0.2, 0.22),  # Compressed shadows
        (0.5, 0.5),   # Midpoint
        (0.8, 0.82),  # Compressed highlights
        (1.0, 0.95)   # Rolled-off whites
    ]
    
    transform.global_saturation = 1.2
    transform.global_contrast = 1.15
    
    print("Custom transform created!")
    print("Visualizing curves...")
    fig = transform.visualize_all_curves()
    plt.savefig('parametric_curves_teal_orange.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: parametric_curves_teal_orange.png")
    plt.close()
    
    print("Generating before/after comparison...")
    fig = visualize_parametric_transformation(transform, "Teal & Orange")
    plt.savefig('parametric_comparison_teal_orange.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: parametric_comparison_teal_orange.png")
    plt.close()
    
    print("\n" + "=" * 80)
    print("WORKFLOW COMPLETE!")
    print("=" * 80)
    print("""
All visualizations generated! Files include:
- parametric_curves_*.png: Shows all 6 control curves
- parametric_comparison_*.png: Before/after on test charts

HOW TO USE THIS SYSTEM:

1. START WITH A PRESET
   transform = ParametricColorTransform()
   transform.load_preset('fuji_velvia_parametric')

2. CUSTOMIZE CURVES
   # Adjust control points (hue, shift_amount)
   transform.hue_vs_hue = [(0, 5), (60, 10), (120, 0), ...]
   
3. VISUALIZE
   fig = transform.visualize_all_curves()
   plt.show()

4. TEST ON IMAGES
   result = transform.transform(rgb_pixel)

5. ITERATE
   Adjust control points, re-visualize, repeat

TIPS FOR COLORISTS:
- Start with fewer control points, add detail as needed
- Watch the curve visualization - smooth is usually better
- Test on Macbeth chart - skin tones are critical
- Hue vs Hue is the most powerful control
- Lum vs Sat creates depth (desaturate shadows/highlights)
- Channel crosstalk adds organic film feel

EXPORTING TO DCTL:
This parametric system can be converted to DCTL by:
1. Baking control points into lookup tables (LUTs)
2. Implementing spline interpolation in DCTL code
3. Or using a hybrid approach with 3D LUTs for complex curves
    """)
