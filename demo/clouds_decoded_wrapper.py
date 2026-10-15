from pathlib import Path

import matplotlib.pyplot as plt
from clouds_decoded.assets import download_asset
from clouds_decoded.data import Sentinel2Scene
from clouds_decoded.modules.albedo_estimator.config import AlbedoEstimatorConfig
from clouds_decoded.modules.albedo_estimator.processor import AlbedoEstimator
from clouds_decoded.modules.cloud_height_emulator import (
    CloudHeightEmulatorConfig,
    CloudHeightEmulatorProcessor,
)
from clouds_decoded.modules.cloud_mask.config import CloudMaskConfig
from clouds_decoded.modules.cloud_mask.processor import CloudMaskProcessor
from clouds_decoded.modules.refl2prop.config import ShadingRefl2PropConfig
from clouds_decoded.modules.refl2prop.processor import CloudPropertyInverter
from clouds_decoded.modules.refocus.config import RefocusConfig
from clouds_decoded.modules.refocus.processor import RefocusProcessor
from clouds_decoded.visualisation import Visualiser


def _run_cloud_mask(output_path: Path, sentinel_scene, device: str | None = None):
    mask_config = CloudMaskConfig(method="senseiv2", batch_size=4, output_dir=str(output_path))
    mask_config.device = device
    mask_processor = CloudMaskProcessor(mask_config)
    mask_result = mask_processor.process(sentinel_scene)
    mask_result.write(str(output_path / "cloud_mask.tif"))
    return mask_result

def _run_cloud_height(mask_result, sentinel_scene, output_path: Path, device: str | None = None):
    height_config = CloudHeightEmulatorConfig(device=device, output_dir=str(output_path))
    height_config.device = device
    height_processor = CloudHeightEmulatorProcessor(height_config)
    binary_mask = mask_result.to_binary(positive_classes=[1,2], threshold=0.2)
    height_result = height_processor.process(sentinel_scene, cloud_mask=binary_mask)
    height_result.write(str(output_path / "cloud_height.tif"))
    return height_result


def _run_albedo(output_path: Path, mask_result, sentinel_scene):
    albedo_config = AlbedoEstimatorConfig(method="idw", output_dir=str(output_path))
    albedo_processor = AlbedoEstimator(albedo_config)

    albedo_mask = mask_result.to_binary(positive_classes=[1,2,3])
    albedo_result = albedo_processor.process(sentinel_scene, cloud_mask=albedo_mask)
    albedo_result.write(str(output_path / "albedo.tif"))
    return albedo_result

def _run_refocus(scene, height_data, output_path: Path):
    refocus_config = RefocusConfig(output_resolution=None, output_dir=str(output_path))
    refocus_processor = RefocusProcessor(refocus_config)

    refocussed_scene = refocus_processor.process(scene, height_data=height_data)
    return refocussed_scene

def _cloud_props(output_path: Path, refocused_scene, height_result, albedo_result):
    props_config = ShadingRefl2PropConfig(output_dir=str(output_path))
    props_processor = CloudPropertyInverter(props_config)

    # Cloud properties use the refocused scene, height map, and albedo
    props_result = props_processor.process(
        refocused_scene,
        height_data=height_result,
        albedo_data=albedo_result,
    )
    props_result.write(str(output_path / "properties.tif"))
    print(f"Properties shape: {props_result.data.shape}")
    print(f"Bands: {props_result.metadata.band_names}")

def run_clouds_decoded(output_path: Path, scene_path: Path, device: str | None = None):
    # Download model weights
    download_asset("cloud_mask")
    download_asset("emulator")
    download_asset("refl2prop")
    scene = Sentinel2Scene()
    scene.read(scene_path)

    mask_result = _run_cloud_mask(output_path, scene, device)
    height_result = _run_cloud_height(mask_result, scene, output_path, device)
    albedo_result = _run_albedo(output_path=output_path, mask_result=mask_result, sentinel_scene=scene)
    refocused_scene = _run_refocus(scene=scene, height_data=height_result, output_path=output_path)

    _cloud_props(output_path=output_path, refocused_scene=refocused_scene, height_result=height_result, albedo_result=albedo_result)

def clouds_decoded_visualise(output_path: Path, scene_path: Path):
    vis = Visualiser.from_directory(str(output_path), scene_path=str(scene_path))
    vis.overview(ncols=3, figsize=(20, 16))
    plt.savefig(output_path / "final.png")
