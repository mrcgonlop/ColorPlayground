# Film Emulation DCTL - User Guide

## Overview

This DCTL (DaVinci Color Transform Language) provides professional parametric film emulation for DaVinci Resolve. It offers precise control over color transformations using 8 parametric curves, per-channel adjustments, and multiple working color spaces.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Working Color Spaces](#working-color-spaces)
3. [Film Presets](#film-presets)
4. [Parametric Curves System](#parametric-curves-system)
5. [Advanced Features](#advanced-features)
6. [Exposed Constants](#exposed-constants)
7. [Troubleshooting](#troubleshooting)
8. [Best Practices](#best-practices)

---

## Quick Start

### Installation

1. Copy `base7.dctl` to your DaVinci Resolve LUT folder:
   - **Windows**: `C:\ProgramData\Blackmagic Design\DaVinci Resolve\Support\LUT\`
   - **macOS**: `/Library/Application Support/Blackmagic Design/DaVinci Resolve/LUT/`
   - **Linux**: `/opt/resolve/LUT/`

2. In DaVinci Resolve Color page:
   - Right-click on a node
   - Select **DCTL** → **base7**

### Basic Usage

1. **Select a Preset**: Choose from 6 built-in film presets or start with **Neutral**
2. **Choose Working Color Space**: HSV (recommended), LCH (for perceptual accuracy), or YCbCr
3. **Adjust Global Controls**: Exposure, Saturation, Contrast, Hue Shift
4. **Fine-tune**: Use parametric curves in the web app to export custom presets

---

## Working Color Spaces

### HSV (Hue, Saturation, Value)
- **Best for**: General color grading, intuitive control
- **Pros**: Fast, predictable, easy to understand
- **Cons**: Not perceptually uniform
- **Recommended for**: Most use cases, real-time grading

### LCH (Lightness, Chroma, Hue)
- **Best for**: Perceptually accurate color manipulation
- **Pros**: Perceptually uniform, excellent for skin tones
- **Cons**: Slower, can produce out-of-gamut colors (especially blues)
- **Recommended for**: Final grading, skin tone work, subtle adjustments
- **Special Controls**:
  - **LCH Chroma Scale** (64-256): Adjusts the chroma normalization factor
    - Default: 128
    - Higher values: More saturated colors
    - Lower values: More muted colors
  - **LCH Max Chroma** (1.0-4.0): Controls out-of-gamut clipping
    - Default: 2.0
    - Higher values: Allow more vivid blues and cyans (may clip)
    - Lower values: More conservative, safer for delivery

**LCH Blue Handling**: This DCTL now properly handles blue colors in LCH space. If you see blue clipping or artifacts:
1. Increase **LCH Max Chroma** to 3.0 or higher
2. Adjust **LCH Chroma Scale** to 96 or 160 for different chroma ranges
3. Consider switching to HSV if extreme blues are critical

### YCbCr (Luma, Chroma Blue, Chroma Red)
- **Best for**: Broadcast-safe grading
- **Pros**: Separates luminance from chrominance, BT.709 coefficients
- **Cons**: Less intuitive hue control
- **Recommended for**: Broadcast delivery, maintaining luma levels

---

## Film Presets

### 1. Neutral
- **Character**: Transparent, no color shift
- **Use**: Starting point, technical grading

### 2. Fuji Velvia
- **Character**: Punchy, high saturation, vivid greens/blues
- **Use**: Landscapes, vibrant scenes
- **Signature**: +15% saturation in greens, slight green shift

### 3. Kodak Portra
- **Character**: Soft, warm skin tones, reduced reds
- **Use**: Portraits, weddings, fashion
- **Signature**: -3° red rotation toward orange, gentle highlight roll-off

### 4. Cinestill 800T
- **Character**: Cyan/teal highlights, warm shadows
- **Use**: Night scenes, neon lights, urban
- **Signature**: +12° cyan shift, halation-like glow

### 5. Teal & Orange
- **Character**: Hollywood blockbuster look
- **Use**: Action, drama, commercial work
- **Signature**: +15° red toward orange, -15° cyan enhancement

### 6. Agfa Vista
- **Character**: Slightly desaturated, cool shadows
- **Use**: Documentary, vintage look
- **Signature**: -8° yellow shift, reduced saturation rolloff

---

## Parametric Curves System

The DCTL uses **8 parametric curves** to transform colors. These are defined in the web app and exported as DCTL presets.

### Core Curves

#### 1. Hue → Hue (Color Rotation)
- **Input**: 0-360° (hue)
- **Output**: -30 to +30° (hue shift)
- **Effect**: Rotates specific hues toward different colors
- **Example**: Shift reds toward orange (0° → +10°)

#### 2. Hue → Saturation
- **Input**: 0-360° (hue)
- **Output**: 0-2x (saturation multiplier)
- **Effect**: Makes specific colors more or less saturated
- **Example**: Boost green saturation (120° → 1.3x)

#### 3. Hue → Luminance
- **Input**: 0-360° (hue)
- **Output**: -0.2 to +0.2 (luminance offset)
- **Effect**: Brightens or darkens specific hues
- **Example**: Darken blues (240° → -0.05)

#### 4. Luminance → Hue (NEW)
- **Input**: 0-1 (luminance)
- **Output**: -30 to +30° (hue shift)
- **Effect**: Different hue shifts in shadows vs highlights
- **Example**: Warm shadows (0.0 → +5°), cool highlights (1.0 → -3°)
- **Film Behavior**: Accurately simulates how film stocks shift colors differently at different exposure levels

#### 5. Saturation → Saturation
- **Input**: 0-1 (input saturation)
- **Output**: 0-1 (output saturation)
- **Effect**: S-curves create film-like saturation rolloff
- **Example**: Compress high saturation (0.8 → 0.75)

#### 6. Saturation → Luminance
- **Input**: 0-1 (saturation)
- **Output**: -0.2 to +0.2 (luminance offset)
- **Effect**: Film density - saturated colors appear darker
- **Example**: Darken high saturation (1.0 → -0.05)

#### 7. Luminance → Saturation
- **Input**: 0-1 (luminance)
- **Output**: 0-2x (saturation multiplier)
- **Effect**: **Critical!** Desaturates shadows/highlights
- **Example**: Typical film response:
  - Shadows (0.0 → 0.6x)
  - Midtones (0.5 → 1.1x)
  - Highlights (1.0 → 0.7x)

#### 8. Luminance → Luminance (Tone Curve)
- **Input**: 0-1 (input luminance)
- **Output**: 0-1 (output luminance)
- **Effect**: Traditional tone curve, S-curves add contrast
- **Example**: Classic S-curve for film contrast

### Per-Channel Curves

- **Red Curve**: Simulates cyan dye layer
- **Green Curve**: Simulates magenta dye layer
- **Blue Curve**: Simulates yellow dye layer

Usually kept neutral unless simulating specific film crossover.

---

## Advanced Features

### Hue Loops (Saturation-based Hue Rotation)

Simulates how high-saturation colors shift hue in film (especially near-gamut colors).

**Parameters**:
- **Enable Hue Loops**: Turn on/off
- **Per-Hue Loop Strengths** (Red, Yellow, Green, Cyan, Blue, Magenta): -30 to +30°
- **Global Saturation Threshold**: 0.0-1.0 (default 0.7)
  - Only colors above this saturation are affected
- **Enable Per-Hue Thresholds**: Advanced control per hue region
  - Allows different saturation thresholds for each hue

**How it works**: When saturation exceeds the threshold, the hue rotates proportionally to the excess saturation.

**Example Use**:
- Set **Blue Loop** to +10° to shift saturated blues toward cyan
- Set **Global Threshold** to 0.8 to only affect very saturated colors

### Per-Hue Saturation Compression

Simulates how different hues compress saturation differently at high chroma.

**Parameters**:
- **Enable Per-Hue Saturation Compression**: Turn on/off
- **Per-Hue Compression** (Red, Yellow, Green, Cyan, Blue, Magenta): 0.5-1.5
  - < 1.0: Compresses high saturation
  - = 1.0: No effect
  - > 1.0: Expands saturation

**Example Use**:
- Set **Blue Compression** to 0.85 to roll off saturated blues
- Set **Red Compression** to 1.1 to allow more saturated reds

### Luminance-Dependent Hue Shift

The new **Luminance → Hue** curve allows different hue shifts at different brightness levels.

**Film-Accurate Behavior**:
- Film stocks often shift warm in shadows (due to base fog)
- Film may shift cool in highlights (due to dye saturation)
- This curve replicates that behavior

**Example**:
```
Luminance → Hue curve:
0.0 (shadows) → +5° (warmer)
0.5 (midtones) → 0° (neutral)
1.0 (highlights) → -3° (cooler)
```

---

## Exposed Constants

### Middle Grey Reference
- **Parameter**: `middle_grey`
- **Range**: 0.01 - 0.50
- **Default**: 0.18
- **Purpose**: Sets the middle grey reference point for contrast and tone mapping
- **Usage**:
  - 0.18: Standard 18% grey (Ansel Adams Zone V)
  - 0.12: Darker mid-point for moody looks
  - 0.25: Brighter mid-point for high-key scenes

### LCH Chroma Scale
- **Parameter**: `lch_chroma_scale`
- **Range**: 64.0 - 256.0
- **Default**: 128.0
- **Purpose**: Normalizes chroma values in LCH space
- **Usage**:
  - 128: Standard normalization
  - 64: More aggressive chroma compression (muted colors)
  - 256: Less compression (more vivid colors)

### LCH Max Chroma
- **Parameter**: `lch_max_chroma`
- **Range**: 1.0 - 4.0
- **Default**: 2.0
- **Purpose**: Controls out-of-gamut clipping in LCH mode
- **Usage**:
  - 1.0: Conservative, clips to gamut (safe for delivery)
  - 2.0: Moderate, allows some out-of-gamut (recommended)
  - 4.0: Aggressive, maximum chroma range (may clip blues/cyans)

**Note**: Increase this value if you see blue clipping or banding in LCH mode.

---

## Troubleshooting

### Blue Colors Look Weird in LCH Mode

**Problem**: Blues appear clipped, banded, or shift unexpectedly

**Solution**:
1. Increase **LCH Max Chroma** to 3.0 or 4.0
2. Adjust **LCH Chroma Scale**:
   - Try 96 for less vivid blues
   - Try 160 for more vivid blues
3. If still problematic, switch to **HSV** mode

**Why this happens**: Blue colors in CIE Lab/LCH space can exceed the typical RGB gamut. The updated DCTL now handles this better with soft clipping.

### Colors Look Muted

**Problem**: Overall saturation is lower than expected

**Solution**:
1. Check **Global Saturation** slider (should be 1.0 for neutral)
2. In **LCH mode**: Increase **LCH Chroma Scale** to 160 or 192
3. Check **Saturation → Saturation** curve has neutral S-curve
4. Verify **Luminance → Saturation** curve isn't over-desaturating

### Highlight Desaturation Too Strong

**Problem**: Highlights are completely desaturated (white)

**Solution**:
1. Reduce **Highlight Desaturation** slider (default 0.7)
2. Adjust **Luminance → Saturation** curve endpoint:
   - Change (1.0 → 0.7) to (1.0 → 0.85) for more color in highlights

### Shadow Desaturation Too Strong

**Problem**: Shadows are grey/desaturated

**Solution**:
1. Reduce **Shadow Desaturation** slider (default 0.3)
2. Adjust **Luminance → Saturation** curve start point:
   - Change (0.0 → 0.6) to (0.0 → 0.8) for more color in shadows

### Hue Loops Creating Artifacts

**Problem**: Banding or strange hue shifts at high saturation

**Solution**:
1. Reduce **Hue Loop** strengths (try ±5° instead of ±30°)
2. Increase **Global Saturation Threshold** to 0.85 (affects fewer colors)
3. Disable **Per-Hue Thresholds** for simpler behavior

### Out-of-Gamut Colors After Grading

**Problem**: Colors clip when exporting to Rec.709

**Solution**:
1. In **LCH mode**: Reduce **LCH Max Chroma** to 1.5 or 1.0
2. Add a **Gamut Mapping** node after this DCTL
3. Use **Soft Clip** plugin in post-DCTL chain
4. Switch to **YCbCr** mode for broadcast-safe grading

---

## Best Practices

### General Workflow

1. **Start with a Preset**: Choose a film stock that matches your desired look
2. **Set Working Color Space**:
   - HSV for speed and predictability
   - LCH for final grading and skin tones
3. **Global Adjustments First**: Exposure, Saturation, Contrast before fine-tuning
4. **Use the Web App**: Design custom curves in the web app, export as DCTL preset
5. **Test on Scopes**: Monitor waveform and vectorscope for clipping

### LCH Mode Workflow

1. Set **LCH Max Chroma** to 2.0 (default)
2. Grade normally
3. If blues clip: Increase to 3.0-4.0
4. If colors too vivid: Reduce **LCH Chroma Scale** to 96
5. Always check scopes - LCH can produce out-of-gamut values

### Hue Loop Usage

1. Start with loops **disabled**
2. Enable only if you need extreme saturation hue shifts
3. Use subtle values (±5° to ±10°) unless going for an extreme look
4. Set **Global Threshold** high (0.8-0.9) to affect only peak saturation

### Creating Custom Presets

1. Use the **web app** ([ClaudeMonsterInteractive_v5.html](webapp/ClaudeMonsterInteractive_v5.html))
2. Load a preset as starting point
3. Adjust curves visually with real-time preview
4. Export as DCTL using the **Export DCTL Preset** button
5. Paste the exported curves into `base7.dctl` under `CUSTOM` preset
6. Set **Preset** dropdown to **Custom** in Resolve

### Maintaining Compatibility

- The web app and DCTL share the same curve system
- Always export from web app v5 (includes **lumVsHue** curve)
- Older web app versions (v1-v4) don't have **lumVsHue** - add it manually:
  ```
  __CONSTANT__ float lumVsHue_CUSTOM[][2] = {
      {0.0f, 0.0f}, {1.0f, 0.0f}
  };
  ```

### Performance Optimization

- **HSV** is fastest - use for real-time grading
- **LCH** is slower - use for final color
- Disable **Hue Loops** and **Per-Hue Compression** if not needed
- Fewer curve nodes = faster processing

---

## Input/Output Setup

### Input Gamut
Supports all major camera gamuts:
- DaVinci Wide Gamut
- Rec.709, Rec.2020
- ACES AP0, AP1
- ARRI Wide Gamut 3/4
- RED Wide Gamut
- Sony S-Gamut3/Cine
- Panasonic V-Gamut
- Blackmagic Wide Gamut
- And more...

### Input Transfer Function
Supports log and linear:
- Linear
- DaVinci Intermediate
- ACEScct
- ARRI LogC3/LogC4
- RED Log3G10
- Sony S-Log3
- Panasonic V-Log
- And more...

### Display Encoding
Supports all common outputs:
- Rec.1886 (Rec.709 display)
- sRGB
- Display P3
- DCI P3
- Rec.2100 PQ/HLG
- Dolby Vision PQ

---

## Technical Notes

### Color Space Conversions

- **RGB → XYZ**: Uses correct matrix for input gamut (not hardcoded Rec.709)
- **XYZ → Lab**: D65 white point, CIE Lab standards
- **Lab → LCH**: Polar coordinates (Chroma, Hue) from Cartesian (a*, b*)
- **LCH Chroma Normalization**: Divides by `lch_chroma_scale` (default 128)

### Curve Interpolation

All curves use **Catmull-Rom cubic spline interpolation**:
- Smooth, continuous curves
- Passes through all control points
- Natural-looking transitions
- No oscillation artifacts

### Hue Loop Mathematics

Gaussian window functions weight each hue region:
- **Primaries** (R, G, B): 60° width
- **Secondaries** (Y, C, M): 45° width
- Smooth falloff prevents banding
- Normalized weights sum to 1.0

---

## FAQ

**Q: Can I use this in Rec.709 timeline?**
A: Yes! Set **Input Gamut** to match your footage and **Display Encoding** to Rec.1886.

**Q: Does this work in ACES?**
A: Yes! Set **Input Gamut** to ACES AP0 or AP1.

**Q: Why are my blues clipping in LCH mode?**
A: Increase **LCH Max Chroma** to 3.0 or 4.0. See [Troubleshooting](#blue-colors-look-weird-in-lch-mode).

**Q: Can I animate parameters?**
A: No, DCTL parameters cannot be keyframed in Resolve. Use multiple nodes if animation is needed.

**Q: How do I create my own film emulation?**
A: Use the web app to design curves visually, then export as DCTL preset. See [Creating Custom Presets](#creating-custom-presets).

**Q: What's the difference between v6 and v7?**
A: v7 adds:
- Luminance → Hue curve (NEW)
- Per-hue saturation compression
- Advanced hue loop controls
- Exposed constants (middle grey, LCH parameters)
- Fixed blue handling in LCH mode

---

## Credits

**Film Emulation DCTL v7**
- Developed with Claude (Anthropic)
- Web App: [ClaudeMonsterInteractive_v5.html](webapp/ClaudeMonsterInteractive_v5.html)
- DCTL: [base7.dctl](Modular/base7.dctl)

**References**:
- CIE Lab color space: ISO/CIE 11664-4:2019
- Catmull-Rom splines: E. Catmull and R. Rom, 1974
- Film emulation curves: Empirical measurements and visual matching

---

## Version History

### v7 (Current)
- Added Luminance → Hue curve
- Fixed LCH blue color handling
- Added exposed constants (middle grey, LCH parameters)
- Added per-hue saturation compression
- Added advanced hue loop controls

### v6
- Added hue loops feature
- Fixed LCH color space conversions
- Improved YCbCr coefficients (BT.709)

### v5
- Added web app v5 with undo/redo
- Added image upload functionality
- Improved UI layout

---

**For support, issues, or contributions**: See the main repository README.
