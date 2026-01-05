# Procedural Audio API Guide

## Overview

The procedural audio system has been refactored to be modular and composable, making it easy to create complex sounds by combining a base frequency generator with various audio effects.

## Quick Start

### Basic Motor Sound

```python
from procedural_audio import FrequencyGenerator, HarmonicsEffect
from easing_curves import ease_in_out_cubic

# Create a base frequency generator
gen = FrequencyGenerator(
    base_freq=60,                          # Base frequency in Hz
    duration=1.0,                          # Duration in seconds
    volume=0.5,                            # Base volume (0.0-1.0)
    volume_fade_in_duration=0.3,           # Fade in duration
    volume_fade_out_duration=0.3,          # Fade out duration
    volume_interp_func=ease_in_out_cubic,  # Easing function
    pitch_fade_in_duration=0.2,            # Pitch fade in
    pitch_fade_out_duration=0.5,           # Pitch fade out
    pitch_fade_range=(0.7, 1.0),          # Pitch range (70% to 100%)
)

# Generate with effects
sound = gen.apply_effects(
    HarmonicsEffect([(2, 0.3), (3, 0.15)]),  # Add 2nd and 3rd harmonics
)
```

## Core Components

### FrequencyGenerator

The base generator that creates a sine wave with controllable pitch and volume envelopes.

**Parameters:**
- `base_freq`: Base frequency in Hz
- `duration`: Duration in seconds
- `sample_rate`: Sample rate (default: 44100 Hz)
- `volume`: Base volume 0.0-1.0
- `volume_fade_in_duration`: Volume fade in time
- `volume_fade_out_duration`: Volume fade out time
- `volume_fade_in_interp_func`: Easing function for volume fade in
- `volume_fade_out_interp_func`: Easing function for volume fade out (optional, uses fade_in if None)
- `pitch_fade_in_duration`: Pitch fade in time
- `pitch_fade_out_duration`: Pitch fade out time
- `pitch_fade_in_interp_func`: Easing function for pitch fade in
- `pitch_fade_out_interp_func`: Easing function for pitch fade out (optional, uses fade_in if None)
- `pitch_fade_range`: Tuple of (min, max) pitch multipliers

**Methods:**
- `generate()`: Generate base audio without effects
- `apply_effects(*effects)`: Generate and apply a chain of effects

### Available Effects

#### WobbleEffect
Adds frequency wobble/vibrato for character.

```python
WobbleEffect(
    wobbles=[(5, 0.8), (3, 1.5)],  # List of (amplitude, frequency) tuples
    mix=0.2                         # How much to mix (0.0-1.0)
)
```

#### HarmonicsEffect
Adds harmonic overtones to the signal.

```python
HarmonicsEffect([
    (2, 0.3),   # 2nd harmonic at 30% amplitude
    (3, 0.15),  # 3rd harmonic at 15% amplitude
    (4, 0.08)   # 4th harmonic at 8% amplitude
])
```

#### DistortionEffect
Adds soft clipping distortion.

```python
DistortionEffect(
    gain=1.5,       # Gain before clipping
    threshold=0.6   # Clipping threshold
)
```

#### EchoEffect
Adds echo/delay.

```python
EchoEffect(
    delay_time=0.15,  # Delay in seconds
    decay=0.3,        # Decay factor per echo
    repeats=2         # Number of echoes
)
```

#### WahWahEffect
Sweeping filter effect.

```python
WahWahEffect(
    min_freq=300,    # Minimum filter frequency
    max_freq=1500,   # Maximum filter frequency
    sweep_rate=0.5   # Sweep rate in Hz
)
```

#### TremoloEffect
Amplitude modulation.

```python
TremoloEffect(
    rate=8.0,   # Modulation rate in Hz
    depth=0.3   # Modulation depth (0.0-1.0)
)
```

## Debug Tools

The `audio_debug` module provides visualization and analysis tools.

### create_audio_debug_plot()

Visualize waveform and spectrogram for debugging.

```python
from procedural_audio import FrequencyGenerator, HarmonicsEffect
from audio_debug import create_audio_debug_plot

# Generate some audio
gen = FrequencyGenerator(base_freq=100, duration=1.0)
sound = gen.apply_effects(HarmonicsEffect([(2, 0.3)]))

# Visualize it
create_audio_debug_plot(
    sound,
    sample_rate=44100,
    title="My Audio",
    save_path="analysis.png"  # Or None to display interactively
)
```

**Requirements:** `pip install matplotlib scipy`

The plot shows:
- **Waveform**: Time-domain visualization with amplitude stats
- **Spectrogram**: Frequency content over time (STFT)

## Complete Examples

### Motor Sound with Effects

```python
gen = FrequencyGenerator(
    base_freq=60,
    duration=2.0,
    volume=0.5,
    volume_fade_in_duration=0.3,
    volume_fade_out_duration=0.3,
    volume_fade_in_interp_func=ease_in_cubic,
    pitch_fade_in_duration=0.2,
    pitch_fade_out_duration=0.5,
    pitch_fade_in_interp_func=ease_in_out_cubic,
    pitch_fade_range=(0.7, 1.0),
)

sound = gen.apply_effects(
    WobbleEffect([(5, 0.8), (3, 1.5)], mix=0.2),
    HarmonicsEffect([(2, 0.3), (3, 0.15), (4, 0.08)]),
    EchoEffect(delay_time=0.15, decay=0.3, repeats=2),
)

export_audio(sound, "motor.wav")
```

### Using Different Fade In/Out Curves

You can specify different easing curves for fade-in and fade-out:

```python
from easing_curves import ease_in_elastic, ease_out_bounce, ease_in_out_cubic

gen = FrequencyGenerator(
    base_freq=100,
    duration=2.0,
    volume=0.5,
    # Sharp fade in, smooth fade out for volume
    volume_fade_in_duration=0.1,
    volume_fade_out_duration=0.5,
    volume_fade_in_interp_func=ease_in_cubic,
    volume_fade_out_interp_func=ease_out_bounce,
    # Different curves for pitch
    pitch_fade_in_duration=0.3,
    pitch_fade_out_duration=0.8,
    pitch_fade_in_interp_func=ease_in_elastic,
    pitch_fade_out_interp_func=ease_in_out_cubic,
    pitch_fade_range=(0.5, 1.0),
)
```

### Using Custom Velocity/Speed Envelopes

```python
import numpy as np

# Get velocity data from your robot (or simulate it)
# This could be from joint encoders, motion capture, etc.
joint_velocity = np.array([...])  # Your velocity data

# Normalize to 0-1 range
max_vel = np.max(np.abs(joint_velocity))
velocity_normalized = np.abs(joint_velocity) / max_vel if max_vel > 0 else joint_velocity

# Use velocity to drive both volume and pitch
gen = FrequencyGenerator(
    base_freq=80,
    duration=2.0,
    volume=0.5,
    # Volume follows velocity directly
    volume_envelope=velocity_normalized,
    # Pitch also follows velocity (mapped to 60-100% of base frequency)
    pitch_envelope=velocity_normalized,
    pitch_fade_range=(0.6, 1.0),
)

sound = gen.apply_effects(
    HarmonicsEffect([(2, 0.3), (3, 0.15)]),
    WobbleEffect([(3, 1.2)], mix=0.15),
)
```

**Notes:**
- Custom envelopes override fade parameters (fade_in_duration, fade_out_duration, etc.)
- Envelopes should be in 0-1 range for best results
- If the envelope length doesn't match the audio length, it will be automatically resampled
- You can provide custom envelopes for volume, pitch, or both

### Distorted Servo Sound

```python
gen = FrequencyGenerator(
    base_freq=800,
    duration=1.5,
    volume=0.4,
    pitch_fade_range=(0.5, 1.0),
)

sound = gen.apply_effects(
    WobbleEffect([(50, 0.4)], mix=0.15),
    HarmonicsEffect([(2, 0.2)]),
    DistortionEffect(gain=1.5, threshold=0.6),
    TremoloEffect(rate=8.0, depth=0.3),
)

export_audio(sound, "servo.wav")
```

### Creating Custom Effects

You can create custom effects by extending `AudioEffect`:

```python
from procedural_audio import AudioEffect
import numpy as np

class MyCustomEffect(AudioEffect):
    def __init__(self, param1, param2):
        self.param1 = param1
        self.param2 = param2

    def apply(self, audio, t, frequency, sample_rate):
        # Modify audio here
        modified = audio * self.param1
        # ... your processing ...
        return modified

# Use it
sound = gen.apply_effects(
    MyCustomEffect(param1=1.5, param2=0.3),
    HarmonicsEffect([(2, 0.3)]),
)
```

## Tips

1. **Start simple**: Begin with just `FrequencyGenerator` and `HarmonicsEffect`
2. **Layer effects**: Effects are applied in order, so experiment with different chains
3. **Use create_audio_debug_plot()**: Visualize your sounds to understand what each effect does
4. **Pitch ranges**: Use `pitch_fade_range=(0.7, 1.0)` for natural motor start/stop sounds
5. **Volume envelopes**: Always use fade in/out to avoid clicks
6. **Wobble subtly**: Keep wobble `mix` parameter low (0.1-0.3) for realism

## Running Examples

Run the module directly to generate all examples:

```bash
cd animation-database-service
python src/procedural_audio.py
```

This generates:
- Legacy rotation test sounds
- Example sounds with various effects
- Debug visualization example

See `example_debug_plot.py` for a standalone usage example.

