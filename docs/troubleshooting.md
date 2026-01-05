| Symptom | Cause | Fix |
|---------|-------|-----|
| No audio from robot (low volume) | Reachy speaker volume set too low by default | Increase Reachy speaker volume to maximum |
| No audio from robot (device conflict) | Another application capturing Reachy speaker | Check `animation-compositor` logs for "Error querying device (-1)", verify Reachy speaker is not set as system default in Ubuntu sound settings, ensure no other apps are capturing the speaker, then restart the demo |

If you have any issues with Reachy that are not covered by this guide, please read [Hugging Face's official troubleshooting guide](https://huggingface.co/docs/reachy_mini/troubleshooting).

> [!NOTE] 
> DGX Spark uses a Unified Memory Architecture (UMA), which enables dynamic memory sharing between the GPU and CPU. 
> With many applications still updating to take advantage of UMA, you may encounter memory issues even when within 
> the memory capacity of DGX Spark. If that happens, manually flush the buffer cache with:
```bash
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
```


For latest known issues, please review the [DGX Spark User Guide](https://docs.nvidia.com/dgx/dgx-spark/known-issues.html).

