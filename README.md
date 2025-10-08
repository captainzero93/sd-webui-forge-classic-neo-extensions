# SD WebUI Forge Classic Neo Extensions

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
https://github.com/wkpark/uddetailer
```

Then restart Forge Neo.

### Step 2: Add Requirements

Without this step, the second tab where you download the hands/alternate models won't work.

1. Navigate to: `sd-webui-forge-neo\extensions\uddetailer`
2. Create or edit `requirements.txt`
3. Add the following lines:

```txt
ultralytics>=8.0.0
```
may also be worth adding 
```opencv-python-headless``` in the line below.

I also reccomend mediapipe_face_full, otherwise it seems to not active sometimes

4. Restart Forge Neo

The extension(s) should now be working!

## Contributing

Feel free to submit issues or pull requests if you encounter any problems or have improvements.

## License

Please refer to individual extension licenses.
