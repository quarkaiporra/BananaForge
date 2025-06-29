#!/usr/bin/env python3
"""Basic usage example for BananaForge."""

import torch
from pathlib import Path

from bananaforge.core.optimizer import LayerOptimizer, OptimizationConfig
from bananaforge.image.processor import ImageProcessor
from bananaforge.materials.database import MaterialDatabase, DefaultMaterials
from bananaforge.output.exporter import ModelExporter
from bananaforge.materials.matcher import ColorMatcher


def main():
    """Run basic BananaForge conversion example."""
    # Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_image_path = "sample_image.jpg"  # Replace with your image path
    output_dir = "./output"
    
    print(f"🚀 Starting BananaForge conversion using {device}")
    
    # 1. Initialize components
    print("📁 Initializing components...")
    image_processor = ImageProcessor(device)
    material_db = DefaultMaterials.create_bambu_basic_pla()
    
    print(f"📦 Loaded {len(material_db)} materials")
    
    # 2. Load and process image
    print("🖼️  Loading image...")
    try:
        image = image_processor.load_image(input_image_path, target_size=(512, 512))
        processed_image = image_processor.preprocess_for_optimization(image)
        print(f"✅ Image loaded: {image.shape}")
    except FileNotFoundError:
        print(f"❌ Image not found: {input_image_path}")
        print("Please provide a valid image path or create a sample image")
        return
    
    # 3. Match materials to image colors
    print("🎨 Matching materials to image colors...")
    color_matcher = ColorMatcher(material_db, device)
    selected_materials, selected_colors, color_mapping = color_matcher.optimize_material_selection(
        processed_image, max_materials=6
    )
    
    print(f"🎯 Selected {len(selected_materials)} materials:")
    for i, material_id in enumerate(selected_materials):
        material = material_db.get_material(material_id)
        if material:
            print(f"  {i+1}. {material.name} ({material.color_hex})")
    
    # 4. Setup optimization
    print("⚙️  Setting up optimization...")
    config = OptimizationConfig(
        iterations=500,  # Reduced for demo
        learning_rate=0.015,
        layer_height=0.08,
        max_layers=30,  # Reduced for demo
        device=device,
        early_stopping_patience=50
    )
    
    optimizer = LayerOptimizer(
        image_size=(512, 512),
        num_materials=len(selected_materials),
        config=config
    )
    
    # 5. Run optimization
    print("🔄 Starting optimization...")
    
    def progress_callback(step, loss_dict, pred_image, height_map):
        if step % 50 == 0:
            total_loss = loss_dict['total'].item()
            print(f"  Step {step:3d}/500, Loss: {total_loss:.4f}")
    
    loss_history = optimizer.optimize(
        target_image=processed_image,
        material_colors=selected_colors,
        callback=progress_callback
    )
    
    print("✅ Optimization completed!")
    
    # 6. Get final results
    print("📊 Extracting results...")
    final_image, height_map, material_assignments = optimizer.get_final_results(selected_colors)
    
    print(f"   Max height: {height_map.max().item() * config.layer_height:.2f}mm")
    print(f"   Layers used: {material_assignments.shape[0]}")
    
    # 7. Export results
    print("💾 Exporting results...")
    exporter = ModelExporter(
        layer_height=config.layer_height,
        physical_size=180.0
    )
    
    output_path = Path(output_dir)
    generated_files = exporter.export_complete_model(
        height_map=height_map,
        material_assignments=material_assignments,
        material_database=material_db,
        material_ids=selected_materials,
        output_dir=output_path,
        project_name="demo_model",
        export_formats=["stl", "instructions", "cost_report"]
    )
    
    # 8. Show results
    print("\n🎉 Conversion completed successfully!")
    print(f"📁 Output directory: {output_path}")
    print("📄 Generated files:")
    for file_type, file_path in generated_files.items():
        print(f"   {file_type}: {Path(file_path).name}")
    
    print(f"\n📈 Final optimization loss: {loss_history['total'][-1]:.4f}")
    
    # 9. Estimate costs
    print("\n💰 Cost estimation:")
    estimates = exporter.estimate_print_time(
        height_map, material_assignments, material_db, selected_materials
    )
    
    print(f"   Print time: {estimates['print_time_hours']:.1f} hours")
    print(f"   Material cost: ${estimates['material_cost_usd']:.2f}")
    print(f"   Material swaps: {estimates['num_swaps']}")
    
    print("\n✨ Ready for 3D printing!")


if __name__ == "__main__":
    main()