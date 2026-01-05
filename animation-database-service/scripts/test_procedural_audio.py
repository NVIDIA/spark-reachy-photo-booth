# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path

# Add src directory to path for imports. Hacky, but not very relevant,
# since it's only for testing audio files while creating them.
src_dir = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_dir))

from audio import (  # noqa: E402  # pyright: ignore[reportMissingImports]
    audio_debug,
    audio_utility,
)
from procedural import (  # noqa: E402  # pyright: ignore[reportMissingImports]
    procedural_audio,
)

if __name__ == "__main__":
    tests_dir = Path(__file__).parent / "audio_tests"
    tests_dir.mkdir(exist_ok=True)

    test_sounds = [
        (
            "output_body_angle",
            lambda: procedural_audio.generate_body_angle_sound(duration=3.0),
        ),
        (
            "output_head_yaw",
            lambda: procedural_audio.generate_head_yaw_sound(duration=3.0),
        ),
        (
            "output_head_pitch",
            lambda: procedural_audio.generate_head_pitch_sound(duration=3.0),
        ),
        (
            "output_head_roll",
            lambda: procedural_audio.generate_head_roll_sound(duration=3.0),
        ),
        (
            "output_l_antenna_angle",
            lambda: procedural_audio.generate_l_antenna_angle_sound(duration=3.0),
        ),
        (
            "output_r_antenna_angle",
            lambda: procedural_audio.generate_r_antenna_angle_sound(duration=3.0),
        ),
    ]

    generated_files = []
    audio = None
    for _, (filename, func) in enumerate(test_sounds, 1):
        try:
            audio = func()
            filepath = str(tests_dir / f"{filename}.wav")
            audio_utility.export_audio(audio, filepath)
            generated_files.append(filepath)
            print(f"  ✓ {filepath}")
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            import traceback

            traceback.print_exc()

    print(f"=== Generated {len(generated_files)}/{len(test_sounds)} test sounds! ===")

    # Play last sound
    if generated_files:
        last_path = generated_files[-1]
        print(f"\nPlaying: {last_path}")
        try:
            audio_utility.play_audio(last_path)
        except PermissionError as e:
            print(f"⚠ Playback failed (permission error): {e}")
            print("💡 Try: Close audio players and run again")
            print(f"   or play manually: {last_path}")
        except Exception as e:
            print(f"⚠ Playback failed: {e}")
            print(f"💡 You can manually play: {last_path}")

    # # Visualize last sound
    if generated_files:
        audio_path = generated_files[-1]

        visualization_path = str(tests_dir / "audio_viz.png")
        audio_debug.create_audio_debug_plot(
            audio,
            title=audio_path,
            save_path=visualization_path,
        )
        print(f"✓ Visualization saved: {visualization_path}")

    print("\n🤖 Complete! All functions working correctly.")
