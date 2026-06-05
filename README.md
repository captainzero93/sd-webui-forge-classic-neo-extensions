# SD WebUI Forge Classic Neo Extensions

[![ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/captainzero)

This repository contains missing extensions from Webui Forge - Neo that have been updated.
https://github.com/Haoming02/sd-webui-forge-classic/tree/neo

## Available Extensions

- **FreeU (v2)** - FreeU extension for Forge Neo
- **Perturbed Attention** - Perturbed attention guidance for Forge Neo  
- **SAG (Self-Attention Guidance)** - Self-attention guidance for Forge Neo
- **UDDetailer Setup Instructions** - μ Detection Detailer install instructions with fix
## Installation

1. Clone or download this repository
2. Open / extract depending on which method you used. Extract zip of the repo if that is how you decided to do it. ( Inside the zip / directory `in the folder named sd-webui-forge-classic-neo-extensions-main` take the folders in there and copy them to your `sd-webui-forge-neo/extensions/` directory. 
3. ## Directory Structure

**\sd-webui-forge-neo\extensions**

```
├───sd_forge_freeu_neo
│   └───scripts
├───sd_forge_perturbed_attention_neo
│   └───scripts
├───sd_forge_sag_neo
│   └───scripts
```
4. Restart Forge Neo 

## UDDetailer Setup (ADetailer Alternative)

To use UDDetailer for auto face/hand inpainting (updated ADetailer working in Neo):

### Step 1: Install UDDetailer

Install from URL in Forge Neo's extension manager:
```
https://github.com/ussoewwin/ADetailer_without_mediapipe
```


On Windows from Open CMD in the sd-webui-forge-neo folder and then paste these

venv\Scripts\python.exe -m pip uninstall -y numpy insightface onnxruntime onnxruntime-gpu

venv\Scripts\python.exe -m pip install numpy==1.26.4

venv\Scripts\python.exe -m pip install onnxruntime-gpu==1.20.2

venv\Scripts\python.exe -m pip install insightface==0.7.3


Then restart Forge Neo.

This will fix the depends errors loading the plugin using the recommended version of Py



The extension(s) should now be working!

## Contributing

Feel free to submit issues or pull requests if you encounter any problems or have improvements.

## License

Please refer to individual extension licenses.
