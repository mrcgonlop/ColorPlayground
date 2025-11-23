What We Currently Have:

Hue → Hue - Color warping
Hue → Saturation - Hue-dependent colorfulness
Luminance → Saturation - Brightness-dependent colorfulness
Luminance → Luminance - Tone curve

Critical Missing Elements:
1. Hue → Luminance (Missing!)
This controls how certain hues become brighter/darker. Film does this naturally:

Yellows are perceptually brighter than blues at the same RGB value
Magentas might get lifted while cyans get crushed
This is crucial for film emulation and I omitted it from the interactive version

2. Saturation → Saturation (Missing!)
Non-linear saturation response:

Highly saturated colors might compress (film can't hold extreme saturation)
Low saturation might expand (lift muted colors)
Creates the "film saturation rolloff" in highlights

3. Channel Crosstalk (Missing - and yes, it should be graphable!)
This is essential for authentic film look. It should be modeled as a 3x3 matrix or as 9 curves:

R → R, R → G, R → B
G → R, G → G, G → B
B → R, B → G, B → B

Film dyes aren't pure:

Cyan dye absorbs red but also some green/blue
Magenta dye absorbs green but bleeds into red/blue
Yellow dye absorbs blue but affects red/green

This creates the "organic" film look where colors influence each other.
Additional Interactions for Complete Control:
4. Saturation → Luminance
How colorfulness affects brightness:

Highly saturated colors might get darker (density)
Desaturated areas might lift (like faded film)

5. Chroma-Dependent Curves in Multiple Color Spaces
Our HSV approach has limitations:

Perceptual non-uniformity - HSV doesn't match human vision
Better to also model in:

LCH (perceptually uniform)
YCbCr (separates luma from chroma)
LMS (cone response - closer to film dyes)


6. Spectral Sensitivity Curves
The ultimate control - model the actual wavelength response of film dyes:

3 curves for R, G, B sensitivity across 380-700nm wavelengths
This is how film actually works
Allows modeling of film's response to real light spectra

7. Grain & Halation (Spatial Effects)
Not color transforms per se, but essential for film:

Grain - luminance and chroma-dependent noise
Halation - glow around bright areas (Cinestill's signature)
These are convolution operations, not point operations

Can Our Current Curves Approximate All Film Behavior?
Short answer: About 70-80% with the 4 curves, 95%+ if we add the missing ones.
What We CAN Do:
✅ Color warping (hue shifts)
✅ Selective saturation boosts
✅ Desaturate shadows/highlights
✅ Tone curves (contrast, lifted blacks)
✅ Most "color grading" film looks
What We CANNOT Do Well:
❌ Channel crosstalk (organic color blending)
❌ Hue-dependent luminance shifts (yellows brighter than blues)
❌ Saturation compression at extremes
❌ Accurate emulation of specific film stocks measured spectrally
❌ Perceptually uniform adjustments
The Complete Parametric System Should Include:
ESSENTIAL (must-have):
1. Hue → Hue
2. Hue → Saturation  
3. Hue → Luminance ⚠️ MISSING
4. Saturation → Saturation ⚠️ MISSING
5. Luminance → Saturation
6. Luminance → Luminance
7. Channel Crosstalk Matrix (9 values or curves) ⚠️ MISSING

ADVANCED (nice-to-have):
8. Saturation → Luminance
9. Per-channel tone curves (separate R, G, B curves)
10. LCH/Lab space controls (perceptually uniform)
11. Chroma vs Luma interactions

ULTIMATE (research-grade):
12. Spectral sensitivity curves (3×wavelength response)
13. Spatial effects (grain, halation, bloom)
Practical Answer for Your DCTL:
For 90% of film emulation work, you need:

The 4 curves we have
+ Hue → Luminance
+ Saturation → Saturation
+ A 3×3 crosstalk matrix

That gives you 7 parametric controls (6 curves + 1 matrix) which can approximate nearly any film stock's color behavior when properly tuned.
Channel crosstalk is the secret sauce - it's what makes film look "analog" vs "digital." Without it, colors feel too clean and separated.
Would you like me to create an enhanced version with all 6 curves + the crosstalk matrix as an interactive control?



Purpose
A pre-DRT look development tool that works in DaVinci Wide Gamut Log space. Designed to be placed before OpenDRT in the node tree, allowing scene-referred color grading that complements the display rendering transform.
Input/Output
Input: DWG Log (DaVinci Intermediate)
Output: DWG Log (passes to OpenDRT for display rendering)
Features
1. Film Stock Presets
a few iconic filmstocks
2. Global Density
Darkens saturated colors to simulate how film dye blocks more light. Single slider with enable toggle.
1. RGB Density Curves
Per-channel density adjustments (Red, Green, Blue) to simulate different film dye layer responses. Works in log space.
1. Channel Crosstalk
6 sliders controlling how each channel bleeds into the others (R→G, R→B, G→R, G→B, B→R, B→G). Simulates film dye interaction.
1. Overall Hue Rotation
3-axis hue rotation (R, G, B) using JP_2499-style matrix. Affects the entire image uniformly.
1. Saturation-to-Hue Rotation
Rotates hue based on saturation level:
Global: Strength, Peak, Shape controls
Per-zone CMY: Cyan, Magenta, Yellow independent sliders
Per-zone RGB: Red, Green, Blue independent sliders
1. Luminance-to-Hue Rotation
Rotates hue based on luminance:
Global: Shadow shift, Highlight shift, Midpoint, Smoothness
Per-zone CMY: Cyan, Magenta, Yellow independent sliders
Per-zone RGB: Red, Green, Blue independent sliders
1. Luminance-to-Luminance (Contrast)
Scene-referred contrast curve:
Contrast (pivot-based)
Toe Lift
Shoulder Softness
Luminance Pivot
1. Purity (Saturation)
Per-channel saturation control using inset matrix (R, G, B sliders).
1.  Advanced Options
Moment space saturation toggle (alternative saturation calculation)
Plan for Rewrite from Scratch
Phase 1: Foundation
Study OpenDRT's working color space approach and matrix definitions
Study JP_2499's Linear Rec.709 workflow and energy correction
Research DaVinci Intermediate log encoding (OETF/EOTF)
Define direct DWG↔Rec.709 matrices with exact inverses
Phase 2: Core Pipeline
Implement log-to-linear and linear-to-log conversions
Implement color space conversion (DWG → working space → DWG)
Verify lossless passthrough with no effects enabled
Implement preset system using OpenDRT's if/else pattern
Phase 3: Film Emulation Features
Implement global density using saturation mask
Implement RGB density curves working in log space
Implement channel crosstalk matrix
Implement JP_2499-style hue rotation matrix
Phase 4: Hue Operations
Research proper hue angle calculation in opponent space
Implement Sat→Hue with Gaussian falloff zones
Implement Lum→Hue with smooth shadow/highlight blending
Ensure hue operations only trigger when enabled (avoid roundtrip when bypassed)
Phase 5: Luminance & Saturation
Implement purity control using inset matrix
Add mix sliders for all sections
Phase 6: Testing & Refinement
Test each preset for pleasing film-like characteristics
Verify no artifacts at color boundaries
Verify smooth transitions across all parametric curves
Test integration with OpenDRT downstream