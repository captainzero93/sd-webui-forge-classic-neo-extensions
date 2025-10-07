import torch
import gradio as gr

from modules import scripts, shared
from modules.script_callbacks import on_cfg_denoiser, remove_current_script_callbacks
from modules.ui_components import InputAccordion


def Fourier_filter(x, threshold, scale):
    # FFT
    x_freq = torch.fft.fftn(x.float(), dim=(-2, -1))
    x_freq = torch.fft.fftshift(x_freq, dim=(-2, -1))

    B, C, H, W = x_freq.shape
    mask = torch.ones((B, C, H, W), device=x.device)

    crow, ccol = H // 2, W // 2
    mask[..., crow - threshold:crow + threshold, ccol - threshold:ccol + threshold] = scale
    x_freq = x_freq * mask

    # IFFT
    x_freq = torch.fft.ifftshift(x_freq, dim=(-2, -1))
    x_filtered = torch.fft.ifftn(x_freq, dim=(-2, -1)).real

    return x_filtered.to(x.dtype)


def patch_freeu_v2(unet_patcher, b1, b2, s1, s2):
    model_channels = unet_patcher.model.diffusion_model.config.get("model_channels")

    scale_dict = {model_channels * 4: (b1, s1), model_channels * 2: (b2, s2)}
    on_cpu_devices = {}

    def output_block_patch(h, hsp, transformer_options):
        process = FreeUForForge.doFreeU

        if process:
            scale = scale_dict.get(h.shape[1], None)
            if scale is not None:
                hidden_mean = h.mean(1).unsqueeze(1)
                B = hidden_mean.shape[0]
                hidden_max, _ = torch.max(hidden_mean.view(B, -1), dim=-1, keepdim=True)
                hidden_min, _ = torch.min(hidden_mean.view(B, -1), dim=-1, keepdim=True)
                hidden_mean = (hidden_mean - hidden_min.unsqueeze(2).unsqueeze(3)) / (hidden_max - hidden_min).unsqueeze(2).unsqueeze(3)

                h[:, :h.shape[1] // 2] = h[:, :h.shape[1] // 2] * ((scale[0] - 1) * hidden_mean + 1)

                if hsp.device not in on_cpu_devices:
                    try:
                        hsp = Fourier_filter(hsp, threshold=1, scale=scale[1])
                    except:
                        print("Device", hsp.device, "does not support the torch.fft functions used in the FreeU node, switching to CPU.")
                        on_cpu_devices[hsp.device] = True
                        hsp = Fourier_filter(hsp.cpu(), threshold=1, scale=scale[1]).to(hsp.device)
                else:
                    hsp = Fourier_filter(hsp.cpu(), threshold=1, scale=scale[1]).to(hsp.device)

        return h, hsp

    m = unet_patcher.clone()
    m.set_model_output_block_patch(output_block_patch)
    return m


def is_flux_model(model):
    """Check if the model is a Flux model"""
    try:
        if hasattr(model, 'model') and hasattr(model.model, 'diffusion_model'):
            config = getattr(model.model.diffusion_model, 'config', {})
            if isinstance(config, dict):
                model_type = config.get('model_type', '').lower()
                if 'flux' in model_type:
                    return True
        
        if hasattr(shared, 'sd_model'):
            if hasattr(shared.sd_model, 'is_webui_legacy_model'):
                return not shared.sd_model.is_webui_legacy_model()
            model_name = getattr(shared.sd_model, 'sd_checkpoint_info', None)
            if model_name:
                name_str = str(model_name).lower()
                if 'flux' in name_str or 'kontext' in name_str or 'krea' in name_str:
                    return True
    except:
        pass
    
    return False


class FreeUForForge(scripts.Script):
    sorting_priority = 12
    
    doFreeU = True
    
    presets_builtin = [
        #   name, b1, b2, s1, s2, start step, end step
        ('Forge default', 1.01, 1.02, 0.99, 0.95, 0.0, 1.0),
        ('SD 1.4', 1.3, 1.4, 0.9, 0.2, 0.0, 1.0),
        ('SD 1.5', 1.5, 1.6, 0.9, 0.2, 0.0, 1.0),
        ('SD 2.1', 1.4, 1.6, 0.9, 0.2, 0.0, 1.0),
        ('SDXL', 1.3, 1.4, 0.9, 0.2, 0.0, 1.0),
    ]
    try:
        import freeu_presets
        presets = presets_builtin + freeu_presets.presets_custom
    except:
        presets = presets_builtin

    def title(self):
        return "FreeU Integrated (SD 1.x, SD 2.x, SDXL)"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, *args, **kwargs):
        
        with InputAccordion(False, label=self.title(),
                          elem_id="extensions-freeu",
                          elem_classes=["extensions-freeu"]) as freeu_enabled:
            with gr.Row():
                freeu_b1 = gr.Slider(label='B1', minimum=0, maximum=2, step=0.01, value=1.01)
                freeu_b2 = gr.Slider(label='B2', minimum=0, maximum=2, step=0.01, value=1.02)
            with gr.Row():
                freeu_s1 = gr.Slider(label='S1', minimum=0, maximum=4, step=0.01, value=0.99)
                freeu_s2 = gr.Slider(label='S2', minimum=0, maximum=4, step=0.01, value=0.95)
            with gr.Row():
                freeu_start = gr.Slider(label='Start step', minimum=0.0, maximum=1.0, step=0.01, value=0.0)
                freeu_end   = gr.Slider(label='End step', minimum=0.0, maximum=1.0, step=0.01, value=1.0)
            with gr.Row():
                freeu_preset = gr.Dropdown(label='', choices=[x[0] for x in FreeUForForge.presets], value='(presets)', type='index', scale=0, allow_custom_value=True)

        def setParams (preset):
            if preset < len(FreeUForForge.presets):
                return  FreeUForForge.presets[preset][1], FreeUForForge.presets[preset][2], \
                        FreeUForForge.presets[preset][3], FreeUForForge.presets[preset][4], \
                        FreeUForForge.presets[preset][5], FreeUForForge.presets[preset][6], '(presets)'
            else:
                return 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, '(presets)'

        freeu_preset.input( fn=setParams,
                            inputs=[freeu_preset],
                            outputs=[freeu_b1, freeu_b2, freeu_s1, freeu_s2, freeu_start, freeu_end, freeu_preset], show_progress=False)

        self.infotext_fields = [
            (freeu_enabled, lambda d: d.get("freeu_enabled", False)),
            (freeu_b1,      "freeu_b1"),
            (freeu_b2,      "freeu_b2"),
            (freeu_s1,      "freeu_s1"),
            (freeu_s2,      "freeu_s2"),
            (freeu_start,   "freeu_start"),
            (freeu_end,     "freeu_end"),
        ]

        return freeu_enabled, freeu_b1, freeu_b2, freeu_s1, freeu_s2, freeu_start, freeu_end

    def denoiser_callback(self, params):
        thisStep = params.sampling_step / (params.total_sampling_steps - 1)
        
        if thisStep >= FreeUForForge.freeu_start and thisStep <= FreeUForForge.freeu_end:
            FreeUForForge.doFreeU = True
        else:
            FreeUForForge.doFreeU = False

    def process_before_every_sampling(self, p, *script_args, **kwargs):
        freeu_enabled, freeu_b1, freeu_b2, freeu_s1, freeu_s2, freeu_start, freeu_end = script_args

        if not freeu_enabled:
            return

        unet = p.sd_model.forge_objects.unet

        # Check for Flux and other incompatible models
        try:
            if is_flux_model(unet):
                gr.Info("FreeU is not compatible with Flux models")
                print("FreeU is not compatible with Flux models")
                return
            
            if hasattr(shared.sd_model, 'is_webui_legacy_model'):
                if not shared.sd_model.is_webui_legacy_model():
                    gr.Info("FreeU only supports SD 1.x, SD 2.x, and SDXL models")
                    print("FreeU only supports SD 1.x, SD 2.x, and SDXL models")
                    return
        except Exception as e:
            print(f"Warning: Could not determine model compatibility for FreeU: {e}")

        # Test if patchable
        model_channels = unet.model.diffusion_model.config.get("model_channels")
        if model_channels is None:
            gr.Info("FreeU is not supported for this model!")
            print("FreeU is not supported for this model!")
            return

        FreeUForForge.freeu_start = freeu_start
        FreeUForForge.freeu_end   = freeu_end
        on_cfg_denoiser(self.denoiser_callback)

        unet = patch_freeu_v2(unet, freeu_b1, freeu_b2, freeu_s1, freeu_s2)

        p.sd_model.forge_objects.unet = unet

        p.extra_generation_params.update(dict(
            freeu_enabled   = freeu_enabled,
            freeu_b1        = freeu_b1,
            freeu_b2        = freeu_b2,
            freeu_s1        = freeu_s1,
            freeu_s2        = freeu_s2,
            freeu_start     = freeu_start,
            freeu_end       = freeu_end,
        ))

        return

    def postprocess(self, params, processed, *args):
        remove_current_script_callbacks()
        return