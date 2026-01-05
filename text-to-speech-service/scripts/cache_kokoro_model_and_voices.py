# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import time

import torch
from kokoro import KPipeline

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Available device: {device} ({device.type})")

pipeline = KPipeline(lang_code="a", device=device.type)
text = "Caching Kokoro model and voices"
# Time the pipeline execution
start_time = time.time()
intermediate_time_a = time.time()

for voice_name in [
    "af_alloy",
    "af_aoede",
    "af_bella",
    "af_heart",
    "af_jessica",
    "af_kore",
    "af_nicole",
    "af_nova",
    "af_river",
    "af_sarah",
    "af_sky",
    "am_adam",
    "am_echo",
    "am_eric",
    "am_fenrir",
    "am_liam",
    "am_michael",
    "am_onyx",
    "am_puck",
    "am_santa",
    "bf_alice",
    "bf_emma",
    "bf_isabella",
    "bf_lily",
    "bm_daniel",
    "bm_fable",
    "bm_george",
    "bm_lewis",
    "ef_dora",
    "em_alex",
    "em_santa",
    "ff_siwis",
    "hf_alpha",
    "hf_beta",
    "hm_omega",
    "hm_psi",
    "if_sara",
    "im_nicola",
    "jf_alpha",
    "jf_gongitsune",
    "jf_nezumi",
    "jf_tebukuro",
    "jm_kumo",
    "pf_dora",
    "pm_alex",
    "pm_santa",
    "zf_xiaobei",
    "zf_xiaoni",
    "zf_xiaoxiao",
    "zf_xiaoyi",
    "zm_yunjian",
    "zm_yunxi",
    "zm_yunxia",
    "zm_yunyang",
]:
    print(f"Processing voice: {voice_name}")

    generator = pipeline(text, voice=voice_name)

    intermediate_time_b = time.time()
    intermediate_time = intermediate_time_b - intermediate_time_a
    intermediate_time_a = intermediate_time_b
    print(
        "Intermediate (voice pipeline setup) execution time: "
        f"{intermediate_time:.2f} seconds"
    )

    for i, (gs, ps, audio) in enumerate(generator):
        intermediate_time_b = time.time()
        intermediate_time = intermediate_time_b - intermediate_time_a
        intermediate_time_a = intermediate_time_b
        print(
            "Intermediate (voice generation) execution time: "
            f"{intermediate_time:.2f} seconds"
        )
        print(f"Chunk {i}: gs={gs}, ps={ps}, audio={audio}")

end_time = time.time()
total_time = end_time - start_time
print(f"\nTotal execution time: {total_time:.2f} seconds\n")
