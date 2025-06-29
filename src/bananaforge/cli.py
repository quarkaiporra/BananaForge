"""Command-line interface for BananaForge."""

import click
import torch
import logging
from pathlib import Path
from typing import Optional, List
import json
import sys
import cv2
import numpy as np
import os

from .core.optimizer import LayerOptimizer, OptimizationConfig
from .image.processor import ImageProcessor
from .image.heightmap import HeightMapGenerator
from .materials.database import MaterialDatabase, DefaultMaterials
from .materials.matcher import ColorMatcher
from .output.exporter import ModelExporter
from .utils.config import Config, ConfigManager
from .utils.logging import setup_logging
from .utils.color import hex_to_rgb
from .utils.visualization import plot_optimization_results
import rich.console
import rich.traceback
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

# Rich console setup
console = Console()
rich.traceback.install(console=console)

# Version import
try:
    from . import __version__
except ImportError:
    __version__ = "unknown"


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
@click.option("--quiet", "-q", is_flag=True, help="Suppress output")
@click.option(
    "--config", type=click.Path(exists=True), help="Path to configuration file"
)
@click.pass_context
def cli(ctx, verbose: bool, quiet: bool, config):
    """BananaForge: AI-powered multi-layer 3D printing optimization."""
    ctx.ensure_object(dict)

    # Setup logging
    setup_logging(verbose=verbose, quiet=quiet)

    # Load configuration
    ctx.obj["config_manager"] = ConfigManager(config)
    ctx.obj["config"] = ctx.obj["config_manager"].get_config()

    # Store context
    ctx.obj["verbose"] = verbose
    ctx.obj["quiet"] = quiet


@cli.command()
@click.argument("input_image", type=click.Path(exists=True))
@click.option(
    "--materials",
    type=click.Path(exists=True),
    help="Material database file (CSV or JSON)",
)
@click.option(
    "--output", "-o", type=click.Path(), default="./output", help="Output directory"
)
@click.option(
    "--max-materials", type=int, default=4, help="Maximum number of materials to use"
)
@click.option("--max-layers", type=int, default=15, help="Maximum number of layers")
@click.option("--layer-height", type=float, default=0.08, help="Layer height in mm")
@click.option("--initial-layer-height", type=float, default=0.16, help="Initial layer height in mm")
@click.option("--nozzle-diameter", type=float, default=0.4, help="Nozzle diameter in mm")
@click.option(
    "--physical-size",
    type=float,
    default=180.0,
    help="Physical size of longest dimension in mm",
)
@click.option(
    "--iterations", type=int, default=6000, help="Number of optimization iterations"
)
@click.option(
    "--learning-rate", type=float, default=0.01, help="Learning rate for optimization"
)
@click.option(
    "--device",
    type=click.Choice(["cpu", "cuda", "mps"]),
    default="cuda",
    help="Device for computation",
)
@click.option(
    "--export-format",
    multiple=True,
    type=click.Choice(
        ["stl", "instructions", "hueforge", "prusa", "bambu", "cost_report"]
    ),
    default=["stl", "instructions", "cost_report"],
    help="Export formats to generate",
)
@click.option(
    "--project-name", default="forgebambu_model", help="Name for the generated project"
)
@click.option(
    "--resolution", type=int, default=512, help="Processing resolution (pixels)"
)
@click.option("--preview", is_flag=True, help="Generate preview visualization")
@click.option("--num-init-rounds", type=int, default=8, help="Number of rounds for heightmap initialization")
@click.option("--num-init-cluster-layers", type=int, default=-1, help="Number of layers to cluster the image into")
@click.pass_context
def convert(
    ctx,
    input_image,
    materials,
    output,
    max_materials,
    max_layers,
    layer_height,
    initial_layer_height,
    nozzle_diameter,
    physical_size,
    iterations,
    learning_rate,
    device,
    export_format,
    project_name,
    resolution,
    preview,
    num_init_rounds,
    num_init_cluster_layers,
):
    """Convert an image to a multi-layer 3D model."""

    try:
        logger = logging.getLogger(__name__)
        logger.info(f"Starting conversion of {input_image}")

        # Initialize components
        image_processor = ImageProcessor(device)

        # Load material database
        if materials:
            material_db = MaterialDatabase()
            if materials.endswith(".csv"):
                material_db.load_from_csv(materials)
            elif materials.endswith(".json"):
                material_db.load_from_json(materials)
            else:
                raise click.ClickException("Material file must be CSV or JSON")
        else:
            click.echo("No material file specified, using default Bambu Lab PLA set")
            material_db = DefaultMaterials.create_bambu_basic_pla()

        logger.info(f"Loaded {len(material_db)} materials")

        # Load and preprocess image
        click.echo("Loading and preprocessing image...")

        # Calculate resolution based on physical size and nozzle diameter
        nozzle_diameter = 0.4  # Default nozzle diameter
        target_stl_resolution = int(round(physical_size * 2 / nozzle_diameter))
        
        # Apply processing reduction factor to avoid memory issues
        # Use larger reduction factor for very high resolutions
        if target_stl_resolution > 2000:
            processing_reduction_factor = 4  # Quarter resolution for very large targets
        elif target_stl_resolution > 1500:
            processing_reduction_factor = 3  # Third resolution for large targets
        else:
            processing_reduction_factor = 2  # Half resolution for normal targets
            
        computed_output_size = target_stl_resolution // processing_reduction_factor
        
        click.echo(f"Target STL resolution: {target_stl_resolution} pixels")
        click.echo(f"Processing resolution: {computed_output_size} pixels (reduced by factor of {processing_reduction_factor})")

        # Use image resizing that maintains aspect ratio
        # First load the image to get its dimensions
        from PIL import Image as PILImage

        pil_img = PILImage.open(input_image)
        orig_w, orig_h = pil_img.size

        # Scaling: scale based on larger dimension
        if orig_w >= orig_h:
            scale = computed_output_size / orig_w
        else:
            scale = computed_output_size / orig_h

        # Compute new dimensions maintaining aspect ratio
        new_w = int(round(orig_w * scale))
        new_h = int(round(orig_h * scale))

        click.echo(f"Scaling: {orig_w}x{orig_h} -> {new_w}x{new_h}")
        # Note: target_size is (height, width) for ImageProcessor, which converts correctly to PIL (width, height)
        image = image_processor.load_image(
            input_image,
            target_size=(new_h, new_w), 
            maintain_aspect=False,  # Already calculated exact size
        )
        # Skip preprocess_for_optimization to preserve aspect ratio from load_image
        processed_image = image  # Use the loaded image directly
        
        # Debug: Print tensor dimensions to verify they match expected dimensions
        click.echo(f"Loaded image tensor shape: {processed_image.shape}")
        click.echo(f"Expected dimensions: height={new_h}, width={new_w}")

        # Update resolution for downstream components
        resolution = computed_output_size

        # Match materials to image colors
        click.echo("Matching materials to image colors...")
        color_matcher = ColorMatcher(material_db, device)
        selected_materials, selected_colors, color_mapping = (
            color_matcher.optimize_material_selection(processed_image, max_materials)
        )

        if not selected_materials:
            raise click.ClickException("No suitable materials found for image")

        logger.info(f"Selected {len(selected_materials)} materials")

        # Initialize Height Map Generator
        click.echo("Initializing height map generator...")
        cfg = Config(
            max_layers=max_layers,
            layer_height=layer_height,
            num_init_rounds=num_init_rounds,
            num_init_cluster_layers=num_init_cluster_layers if num_init_cluster_layers != -1 else max_layers // 2,
            random_seed=ctx.obj['config'].get('random_seed', 0) if isinstance(ctx.obj['config'], dict) else getattr(ctx.obj['config'], 'random_seed', 0),
            background_color=ctx.obj['config'].get('background_color', '#000000') if isinstance(ctx.obj['config'], dict) else getattr(ctx.obj['config'], 'background_color', '#000000')
        )
        
        heightmap_generator = HeightMapGenerator(cfg, device)

        # Convert image tensor to numpy array for heightmap generation
        image_np = processed_image.squeeze(0).permute(1, 2, 0).cpu().numpy()
        image_np = (image_np * 255).astype(np.uint8)

        background_tuple = hex_to_rgb(cfg.background_color)
        material_colors_np = selected_colors.cpu().numpy()

        initial_height_logits_np, initial_global_logits_np, _ = heightmap_generator.generate(
            image_np, background_tuple, material_colors_np
        )

        initial_height_logits = torch.from_numpy(initial_height_logits_np)
        initial_global_logits = torch.from_numpy(initial_global_logits_np)

        # Setup optimization
        click.echo("Setting up optimization...")
        config = OptimizationConfig(
            iterations=iterations,
            learning_rate=learning_rate,
            layer_height=layer_height,
            max_layers=max_layers,
            device=device,
            early_stopping_patience=max(
                iterations, 1000
            ),  # At least as many as iterations
        )

        optimizer = LayerOptimizer(
            image_size=(new_h, new_w),  # Use exact computed dimensions, not square
            num_materials=len(selected_materials),
            config=config,
            target_image=processed_image,
            initial_height_logits=initial_height_logits,
            initial_global_logits=initial_global_logits,
        )

        # Progress callback
        def progress_callback(step, loss_dict, pred_image, height_map):
            if step % 100 == 0:
                total_loss = loss_dict["total"].item()
                click.echo(f" Step {step}/{iterations}, Loss: {total_loss:.4f}")

        # Run optimization
        click.echo("Starting optimization...")
        with click.progressbar(length=iterations, label="Optimizing") as bar:

            def progress_wrapper(step, loss_dict, pred_image, height_map):
                bar.update(10)  # Update by 10 since callback is called every 10 steps
                progress_callback(step, loss_dict, pred_image, height_map)

            loss_history = optimizer.optimize(
                target_image=processed_image,
                material_colors=selected_colors,
                callback=progress_wrapper,
            )

        # Get final results
        final_image, final_height_map, final_material_assignments = (
            optimizer.get_final_results(selected_colors)
        )

        # Create output directory
        output_path = Path(output)
        output_path.mkdir(parents=True, exist_ok=True)

        # Export results
        click.echo("Exporting results...")
        exporter = ModelExporter(
            layer_height=layer_height,
            initial_layer_height=initial_layer_height,
            physical_size=physical_size,
        )

        generated_files = exporter.export_complete_model(
            height_map=final_height_map,
            material_assignments=final_material_assignments,
            material_database=material_db,
            material_ids=selected_materials,
            output_dir=output_path,
            project_name=project_name,
            export_formats=list(export_format),
        )

        if "stl" in generated_files:
            click.echo(f"STL model saved to {generated_files['stl']}")

        if "instructions_txt" in generated_files:
            click.echo(f"Print instructions saved to {generated_files['instructions_txt']}")

        if "cost_report" in generated_files:
            with open(generated_files["cost_report"]) as f:
                report = f.read()
            click.echo("Cost Report:")
            click.echo(report)

        if "hueforge" in generated_files:
            click.echo(f"HueForge project saved to {generated_files['hueforge']}")

        if "prusa" in generated_files:
            click.echo(f"3MF file saved to {generated_files['prusa']}")

        if "bambu" in generated_files:
            click.echo(f"3MF file saved to {generated_files['bambu']}")

        if preview:
            from .utils.visualization import Visualizer

            vis = Visualizer()
            vis.display_image_comparison(
                processed_image.squeeze(0).permute(1, 2, 0).cpu(),
                final_image.squeeze(0).permute(1, 2, 0).cpu(),
                save_path=output_path / f"{project_name}_comparison.png",
            )
            vis.display_height_map(
                final_height_map.squeeze().cpu().numpy(),
                save_path=output_path / f"{project_name}_heightmap.png",
            )

        click.echo("Conversion complete!")
        logger.info(f"Successfully converted {input_image}")

    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
        # Optionally re-raise for debugging or return a specific error code
        # raise e
        # For a cleaner CLI experience, just show the error message
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option(
    "--format", type=click.Choice(["csv", "json"]), default="csv", help="Output format"
)
@click.option(
    "--output", "-o", type=click.Path(), required=True, help="Output file path"
)
@click.option("--brand", multiple=True, help="Filter by brand")
@click.option("--max-materials", type=int, help="Maximum number of materials")
@click.option(
    "--color-diversity", is_flag=True, default=True, help="Optimize for color diversity"
)
def export_materials(format, output, brand, max_materials, color_diversity):
    """Export materials from the database to a file."""
    try:
        from .materials.manager import MaterialManager
        
        logger = logging.getLogger(__name__)

        manager = MaterialManager()
        manager.load_default_materials() # Default for now
        
        logger.info(f"Exporting materials with filters: brands={brands}, max_materials={max_materials}")

        manager.export_materials(
            output_path=output,
            format=format,
            brands=brands,
            max_materials=max_materials,
        )

        click.echo(f"Successfully exported materials to {output}")
        logger.info(f"Exported materials to {output}")

    except Exception as e:
        logger.error(f"Error during material export: {e}", exc_info=True)
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument("input_image", type=click.Path(exists=True))
@click.option("--materials", type=click.Path(exists=True))
@click.option("--max-materials", type=int, default=4)
@click.option(
    "--method",
    type=click.Choice(["perceptual", "euclidean", "lab"]),
    default="perceptual",
)
@click.option("--output", "-o", type=click.Path())
def analyze_colors(input_image, materials, max_materials, method, output):
    """Analyze the color palette of an image and match it to materials."""
    try:
        from .materials.manager import MaterialManager
        logger = logging.getLogger(__name__)

        # Initialize components
        image_processor = ImageProcessor()
        manager = MaterialManager()

        # Load materials
        if materials:
            manager.load_materials_from_file(materials)
        else:
            click.echo("No material file specified, using default Bambu Lab PLA set")
            manager.load_default_materials()

        # Load image
        image = image_processor.load_image(input_image)

        # Analyze colors
        click.echo("Analyzing image colors...")
        analysis = manager.analyze_color_coverage(image, max_materials)

        # Display results
        click.echo("\n--- Color Coverage Analysis ---")
        click.echo(f"  Coverage Score: {analysis['coverage_score']:.2f}")
        click.echo(f"  Accuracy Score: {analysis['accuracy_score']:.2f}")
        click.echo(f"  Combined Score: {analysis['combined_score']:.2f}")
        click.echo(f"  Selected Materials ({analysis['num_materials']}):")
        for mat_id in analysis['material_ids']:
            mat_info = manager.get_material_info(mat_id)
            click.echo(f"    - {mat_info.name} ({mat_info.brand}) - {mat_info.color_hex}")

        if output:
            with open(output, "w") as f:
                json.dump(analysis, f, indent=2)
            click.echo(f"\nAnalysis saved to {output}")

    except Exception as e:
        logger.error(f"Error during color analysis: {e}", exc_info=True)
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument("stl_file", type=click.Path(exists=True))
def validate_stl(stl_file):
    """Validate an STL file for basic printability.

    Checks for issues like being watertight, manifold, and having correct winding.
    """
    try:
        import trimesh

        logger = logging.getLogger(__name__)
        click.echo(f"Validating STL file: {stl_file}")

        mesh = trimesh.load(stl_file)

        # Perform checks
        is_watertight = mesh.is_watertight
        is_manifold = all(mesh.is_manifold)
        is_windable = mesh.is_winding_consistent

        click.echo("\n--- STL Validation Report ---")
        click.echo(f"  File: {stl_file}")
        click.echo(f"  Watertight: {'OK' if is_watertight else 'FAIL'}")
        click.echo(f"  Manifold: {'OK' if is_manifold else 'FAIL'}")
        click.echo(f"  Consistent Winding: {'OK' if is_windable else 'FAIL'}")

        if not all([is_watertight, is_manifold, is_windable]):
            click.echo("\nWarning: STL file has issues that may affect printability.")
            sys.exit(1)
        else:
            click.echo("\nSTL file appears to be valid.")

    except Exception as e:
        logger.error(f"Error during STL validation: {e}", exc_info=True)
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    default="./forgebambu_config.json",
    help="Output configuration file path",
)
def init_config(output):
    """Initialize a default configuration file."""
    try:
        logger = logging.getLogger(__name__)

        # Create a default config manager
        manager = ConfigManager()
        manager.save_config(output)

        click.echo(f"Default configuration file created at: {output}")
        logger.info(f"Initialized config file at {output}")

    except Exception as e:
        logger.error(f"Error initializing config: {e}", exc_info=True)
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
def version():
    """Display BananaForge version."""
    click.echo(f"BananaForge Version: {__version__}")


def main():
    """Main entry point for the CLI."""
    cli(obj={})


if __name__ == "__main__":
    main()
