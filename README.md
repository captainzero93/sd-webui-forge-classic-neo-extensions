# sd-webui-forge-classic-neo-extensions

sd-webui-forge-classic/tree/neo working missing extentions from Forge so far:

-- freeu (v2) for Forge Neo
-- sd_forge_perturbed_attention for Forge Neo
-- sd_forge_sag for FOrge Neo 

To fix / use uddetailer / auto face / hand inpainting ( basically adetailer updated and working in neo)
install from URL in neo, restart,
add requirments.txt to sd-webui-forge-neo\extensions\uddetailer 

in the file place:

ultralytics>=8.0.0
opencv-python-headless

restart neo
- Without the steps above the second tab where you download the hands model wont work.
