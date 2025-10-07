# SD WebUI Forge Classic Neo Extensions

This repository contains missing extensions from Webui Forge - Neo that have been updated.
https://github.com/Haoming02/sd-webui-forge-classic/tree/neo

## Available Extensions

- **FreeU (v2)** - FreeU extension for Forge Neo
- **Perturbed Attention** - Perturbed attention guidance for Forge Neo  
- **SAG (Self-Attention Guidance)** - Self-attention guidance for Forge Neo

## Installation

1. Clone or download this repository
2. Copy the extracted folders ( inside the zip / pull `in the folder named sd-webui-forge-classic-neo-extensions-main` take the folders in there and copy them to your `sd-webui-forge-neo/extensions/` directory.
3. Restart Forge Neo 

## UDDetailer Setup (ADetailer Alternative)

To use UDDetailer for auto face/hand inpainting (updated ADetailer working in Neo):

### Step 1: Install UDDetailer

Install from URL in Forge Neo's extension manager:
```
https://github.com/Uminosachi/sd-webui-uddetailer
```

Then restart Forge Neo.

### Step 2: Add Requirements

Without this step, the second tab where you download the hands model won't work.

1. Navigate to: `sd-webui-forge-neo\extensions\uddetailer`
2. Create or edit `requirements.txt`
3. Add the following lines:

```txt
ultralytics>=8.0.0
opencv-python-headless
```

4. Restart Forge Neo

The extension should now be working, please report any issues
## Contributing

Feel free to submit issues or pull requests if you encounter any problems or have improvements.

## License

Please refer to individual extension licenses.
