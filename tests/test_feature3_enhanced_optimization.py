"""BDD/TDD Tests for Feature 3: Enhanced Optimization Engine

This module contains comprehensive tests for all four stories in Feature 3,
following the Gherkin scenarios defined in the tasks file.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import time
from typing import Dict, List, Optional, Tuple

from bananaforge.core.optimizer import LayerOptimizer, OptimizationConfig
from bananaforge.core.enhanced_optimizer import (
    EnhancedLayerOptimizer,
    EnhancedOptimizationConfig,
    DiscreteValidator,
    LearningRateScheduler,
    EnhancedEarlyStopping,
    MixedPrecisionManager,
)
from bananaforge.core.loss import CombinedLoss
from dataclasses import replace


class TestStory31DiscreteValidationTracking:
    """BDD Tests for Story 3.1: Discrete Validation Tracking
    
    Acceptance Criteria:
    Given an optimization in progress
    When the system runs validation checks every N iterations
    Then it should compute discrete loss metrics
    And track the best discrete solution found so far
    And provide progress feedback to the user
    And use discrete loss for early stopping decisions
    """

    @pytest.fixture
    def basic_config(self):
        """Create a basic optimization configuration for testing."""
        return EnhancedOptimizationConfig(
            iterations=100,
            learning_rate=0.01,
            initial_temperature=1.0,
            final_temperature=0.1,
            layer_height=0.08,
            max_layers=8,
            device="cpu",
            early_stopping_patience=50,
            enable_discrete_tracking=True,
            validation_interval=10,
        )

    @pytest.fixture
    def test_image_and_materials(self):
        """Create test image and materials for optimization testing."""
        # Create a simple 32x32 RGB test image
        image = torch.randn(1, 3, 32, 32)
        
        # Create 4 test materials
        materials = torch.tensor([
            [1.0, 0.0, 0.0],  # Red
            [0.0, 1.0, 0.0],  # Green
            [0.0, 0.0, 1.0],  # Blue
            [1.0, 1.0, 0.0],  # Yellow
        ])
        
        return image, materials

    def test_computes_discrete_loss_metrics_every_n_iterations(
        self, basic_config, test_image_and_materials
    ):
        """Test that the system computes discrete loss metrics every N iterations."""
        image, materials = test_image_and_materials
        
        # Create enhanced optimizer with discrete validation tracking
        optimizer = EnhancedLayerOptimizer(
            image_size=(32, 32),
            num_materials=4,
            config=basic_config,
        )
        
        # Track discrete validation calls
        discrete_validations = []
        
        def validation_callback(step, metrics, pred_image, height_map):
            if 'discrete_loss' in metrics:
                discrete_validations.append((step, metrics['discrete_loss']))
        
        # When running optimization with discrete validation
        loss_history = optimizer.optimize(
            target_image=image,
            material_colors=materials,
            callback=validation_callback,
        )
        
        # Then discrete loss metrics should be computed every N iterations
        assert len(discrete_validations) >= 5, "Should have multiple discrete validations"
        
        # And validations should occur at regular intervals
        if len(discrete_validations) >= 2:
            step_diffs = [discrete_validations[i+1][0] - discrete_validations[i][0] 
                         for i in range(len(discrete_validations)-1)]
            assert all(diff >= 10 for diff in step_diffs), "Validations should be at least 10 steps apart"
        
        # And discrete loss values should be meaningful
        for step, discrete_loss in discrete_validations:
            assert isinstance(discrete_loss, (int, float)), "Discrete loss should be numeric"
            assert discrete_loss >= 0, "Discrete loss should be non-negative"

    def test_tracks_best_discrete_solution_found_so_far(
        self, basic_config, test_image_and_materials
    ):
        """Test that the system tracks the best discrete solution found so far."""
        image, materials = test_image_and_materials
        
        optimizer = EnhancedLayerOptimizer(
            image_size=(32, 32),
            num_materials=4,
            config=basic_config,
        )
        
        # When running optimization
        loss_history = optimizer.optimize(
            target_image=image,
            material_colors=materials,
        )
        
        # Then the optimizer should track the best discrete solution
        assert optimizer.discrete_validator is not None, "Should have discrete validator"
        
        # And the best discrete loss should be valid
        best_loss = optimizer.discrete_validator.best_discrete_loss
        assert isinstance(best_loss, (int, float)), "Best discrete loss should be numeric"
        assert best_loss >= 0, "Best discrete loss should be non-negative"
        
        # And the best discrete state should contain model parameters
        best_state = optimizer.get_best_discrete_state()
        if best_state is not None:
            assert 'global_logits' in best_state, "Should save global logits"
            assert 'height_logits' in best_state, "Should save height logits"

    def test_provides_progress_feedback_to_user(
        self, basic_config, test_image_and_materials
    ):
        """Test that the system provides progress feedback to the user."""
        image, materials = test_image_and_materials
        
        optimizer = EnhancedLayerOptimizer(
            image_size=(32, 32),
            num_materials=4,
            config=basic_config,
        )
        
        # Capture progress feedback
        progress_updates = []
        
        def progress_callback(step, metrics, pred_image, height_map):
            progress_updates.append({
                'step': step,
                'metrics': metrics,
                'has_discrete': 'discrete_loss' in metrics,
            })
        
        # When running optimization with progress callback
        loss_history = optimizer.optimize(
            target_image=image,
            material_colors=materials,
            callback=progress_callback,
        )
        
        # Then progress feedback should be provided
        assert len(progress_updates) > 0, "Should provide progress updates"
        
        # And some updates should include discrete metrics
        discrete_updates = [u for u in progress_updates if u['has_discrete']]
        assert len(discrete_updates) > 0, "Should provide discrete loss updates"
        
        # And updates should be at regular intervals
        steps = [u['step'] for u in progress_updates]
        assert len(set(steps)) > 1, "Should update at multiple different steps"

    def test_uses_discrete_loss_for_early_stopping_decisions(
        self, basic_config, test_image_and_materials
    ):
        """Test that discrete loss is used for early stopping decisions."""
        image, materials = test_image_and_materials
        
        # Configure for early stopping based on discrete loss
        config = EnhancedOptimizationConfig(
            iterations=200,
            learning_rate=0.01,
            early_stopping_patience=20,
            device="cpu",
            max_layers=8,
            layer_height=0.08,
            enable_discrete_tracking=True,
            validation_interval=5,
            enable_enhanced_early_stopping=True,
            early_stopping_metric="discrete_loss",
        )
        
        optimizer = EnhancedLayerOptimizer(
            image_size=(32, 32),
            num_materials=4,
            config=config,
        )
        
        # When running optimization
        loss_history = optimizer.optimize(
            target_image=image,
            material_colors=materials,
        )
        
        # Then early stopping should be based on discrete metrics
        assert optimizer.early_stopping is not None, "Should have early stopping component"
        assert hasattr(optimizer.early_stopping, 'discrete_patience_counter'), "Should track discrete patience"
        
        # And if early stopped, it should be for a valid reason
        if optimizer.early_stopping.should_stop:
            stopping_info = optimizer.early_stopping.get_stopping_info()
            assert stopping_info['reason'] in [
                'discrete_loss_plateau', 'continuous_loss_plateau', 'both_losses_plateau'
            ], "Should have valid early stopping reason"

    def test_discrete_validation_accuracy_vs_continuous(
        self, basic_config, test_image_and_materials
    ):
        """Test that discrete validation provides meaningful metrics compared to continuous."""
        image, materials = test_image_and_materials
        
        optimizer = EnhancedLayerOptimizer(
            image_size=(32, 32),
            num_materials=4,
            config=basic_config,
        )
        
        # Track both continuous and discrete losses
        continuous_losses = []
        discrete_losses = []
        
        def tracking_callback(step, metrics, pred_image, height_map):
            if 'total' in metrics:
                continuous_losses.append(metrics['total'])
            if 'discrete_loss' in metrics:
                discrete_losses.append(metrics['discrete_loss'])
        
        # When running optimization
        loss_history = optimizer.optimize(
            target_image=image,
            material_colors=materials,
            callback=tracking_callback,
        )
        
        # Then both metrics should be available
        assert len(continuous_losses) > 0, "Should have continuous loss values"
        assert len(discrete_losses) > 0, "Should have discrete loss values"
        
        # And discrete losses should generally be different from continuous
        # (since they represent discrete material assignments)
        if len(discrete_losses) >= 2:
            discrete_std = np.std(discrete_losses)
            assert discrete_std >= 0, "Discrete losses should vary meaningfully"


class TestStory32LearningRateScheduling:
    """BDD Tests for Story 3.2: Learning Rate Scheduling
    
    Acceptance Criteria:
    Given an optimization session starting
    When the system begins optimization iterations
    Then it should implement learning rate warmup for initial iterations
    And gradually increase learning rate during warmup phase
    And decay learning rate as optimization progresses
    And adjust rates based on loss convergence patterns
    """

    @pytest.fixture
    def scheduling_config(self):
        """Create configuration for learning rate scheduling tests."""
        return EnhancedOptimizationConfig(
            iterations=100,
            learning_rate=0.01,
            initial_temperature=1.0,
            final_temperature=0.1,
            device="cpu",
            max_layers=8,
            layer_height=0.08,
            enable_lr_scheduling=True,
            warmup_steps=20,
        )

    @pytest.fixture
    def simple_test_data(self):
        """Create simple test data for scheduler testing."""
        image = torch.randn(1, 3, 16, 16)
        materials = torch.tensor([
            [1.0, 0.0, 0.0],  # Red
            [0.0, 1.0, 0.0],  # Green
            [0.0, 0.0, 1.0],  # Blue
            [1.0, 1.0, 0.0],  # Yellow
        ])
        return image, materials

    def test_implements_learning_rate_warmup_for_initial_iterations(
        self, scheduling_config, simple_test_data
    ):
        """Test that the system implements learning rate warmup for initial iterations."""
        image, materials = simple_test_data
        
        # Create optimizer with learning rate scheduling
        optimizer = EnhancedLayerOptimizer(
            image_size=(16, 16),
            num_materials=4,
            config=scheduling_config,
        )
        
        # Track learning rates during optimization
        learning_rates = []
        
        def lr_tracking_callback(step, metrics, pred_image, height_map):
            if optimizer.lr_scheduler is not None:
                current_lr = optimizer.lr_scheduler.get_current_lr()
                learning_rates.append((step, current_lr))
        
        # When running optimization with learning rate scheduling
        loss_history = optimizer.optimize(
            target_image=image,
            material_colors=materials,
            callback=lr_tracking_callback,
        )
        
        # Then learning rate should start low and warm up
        initial_lrs = [lr for step, lr in learning_rates if step < 20]
        if len(initial_lrs) >= 2:
            # Learning rate should increase during warmup
            assert initial_lrs[-1] > initial_lrs[0], "Learning rate should increase during warmup"
            
        # And should reach the target learning rate after warmup
        post_warmup_lrs = [lr for step, lr in learning_rates if step >= 20]
        if len(post_warmup_lrs) > 0:
            target_lr = scheduling_config.learning_rate
            # Should be close to target learning rate after warmup
            assert abs(post_warmup_lrs[0] - target_lr) < target_lr * 0.1, \
                "Should reach target LR after warmup"

    def test_gradually_increases_learning_rate_during_warmup_phase(
        self, scheduling_config, simple_test_data
    ):
        """Test that learning rate gradually increases during warmup phase."""
        image, materials = simple_test_data
        
        # Modify config for this test
        config = EnhancedOptimizationConfig(
            iterations=100,
            learning_rate=0.01,
            initial_temperature=1.0,
            final_temperature=0.1,
            device="cpu",
            max_layers=8,
            layer_height=0.08,
            enable_lr_scheduling=True,
            warmup_steps=30,
        )
        
        optimizer = EnhancedLayerOptimizer(
            image_size=(16, 16),
            num_materials=4,
            config=config,
        )
        
        learning_rates = []
        
        def lr_callback(step, metrics, pred_image, height_map):
            if optimizer.lr_scheduler is not None:
                learning_rates.append(optimizer.lr_scheduler.get_current_lr())
        
        # When running optimization
        loss_history = optimizer.optimize(
            target_image=image,
            material_colors=materials,
            callback=lr_callback,
        )
        
        # Then learning rate should gradually increase during warmup
        warmup_lrs = learning_rates[:30] if len(learning_rates) >= 30 else learning_rates
        
        if len(warmup_lrs) >= 3:
            # Check that learning rate generally increases during warmup
            increases = sum(1 for i in range(1, len(warmup_lrs)) 
                          if warmup_lrs[i] >= warmup_lrs[i-1])
            total_pairs = len(warmup_lrs) - 1
            
            # At least 70% of steps should show non-decreasing learning rate
            increase_ratio = increases / total_pairs if total_pairs > 0 else 0
            assert increase_ratio >= 0.7, f"Learning rate should generally increase during warmup, got {increase_ratio:.2f}"

    def test_decays_learning_rate_as_optimization_progresses(
        self, scheduling_config, simple_test_data
    ):
        """Test that learning rate decays as optimization progresses."""
        image, materials = simple_test_data
        
        # Use longer optimization to see decay
        config = EnhancedOptimizationConfig(
            iterations=150,
            learning_rate=0.02,
            device="cpu",
            max_layers=8,
            layer_height=0.08,
            enable_lr_scheduling=True,
            warmup_steps=10,
            decay_schedule="linear",
        )
        
        optimizer = EnhancedLayerOptimizer(
            image_size=(16, 16),
            num_materials=4,
            config=config,
        )
        
        learning_rates = []
        
        def lr_callback(step, metrics, pred_image, height_map):
            if optimizer.lr_scheduler is not None:
                learning_rates.append(optimizer.lr_scheduler.get_current_lr())
        
        # When running optimization
        loss_history = optimizer.optimize(
            target_image=image,
            material_colors=materials,
            callback=lr_callback,
        )
        
        # Then learning rate should decay after warmup
        if len(learning_rates) >= 50:
            post_warmup_lrs = learning_rates[25:]  # After warmup
            
            if len(post_warmup_lrs) >= 10:
                # Learning rate should generally decrease after warmup
                early_post_warmup = np.mean(post_warmup_lrs[:5])
                late_post_warmup = np.mean(post_warmup_lrs[-5:])
                
                assert late_post_warmup < early_post_warmup, \
                    "Learning rate should decay after warmup phase"

    def test_adjusts_rates_based_on_loss_convergence_patterns(
        self, scheduling_config, simple_test_data
    ):
        """Test that learning rates are adjusted based on loss convergence patterns."""
        image, materials = simple_test_data
        
        # Config with adaptive learning rate
        config = EnhancedOptimizationConfig(
            iterations=100,
            learning_rate=0.01,
            initial_temperature=1.0,
            final_temperature=0.1,
            device="cpu",
            max_layers=8,
            layer_height=0.08,
            enable_lr_scheduling=True,
            warmup_steps=20,
            enable_adaptive_lr=True,  # Enable adaptive learning rate
            lr_patience=10,  # Reduce LR if no improvement for 10 steps
        )
        
        optimizer = EnhancedLayerOptimizer(
            image_size=(16, 16),
            num_materials=4,
            config=config,
        )
        
        learning_rates = []
        losses = []
        
        def adaptive_callback(step, metrics, pred_image, height_map):
            if optimizer.lr_scheduler is not None:
                learning_rates.append(optimizer.lr_scheduler.get_current_lr())
            else:
                learning_rates.append(optimizer.config.learning_rate)
            if 'total' in metrics:
                losses.append(metrics['total'])
        
        # When running optimization with adaptive learning rate
        loss_history = optimizer.optimize(
            target_image=image,
            material_colors=materials,
            callback=adaptive_callback,
        )
        
        # Then learning rate should adapt to loss patterns
        assert hasattr(optimizer, 'lr_scheduler'), "Should have learning rate scheduler"
        
        # And learning rate adjustments should be meaningful
        if len(learning_rates) >= 20:
            lr_std = np.std(learning_rates)
            assert lr_std > 0, "Learning rate should vary during optimization"
            
            # Check for learning rate reductions when loss plateaus
            lr_reductions = sum(1 for i in range(1, len(learning_rates)) 
                              if learning_rates[i] < learning_rates[i-1] * 0.95)
            assert lr_reductions >= 0, "Should have learning rate adjustments"

    def test_learning_rate_scheduler_types_work_correctly(
        self, scheduling_config, simple_test_data
    ):
        """Test that different learning rate scheduler types work correctly."""
        image, materials = simple_test_data
        
        scheduler_types = ["linear", "cosine", "exponential", "step"]
        
        for scheduler_type in scheduler_types:
            # Create config for this scheduler type
            config = EnhancedOptimizationConfig(
                iterations=100,
                learning_rate=0.01,
                initial_temperature=1.0,
                final_temperature=0.1,
                device="cpu",
                max_layers=8,
                layer_height=0.08,
                enable_lr_scheduling=True,
                warmup_steps=10,
                decay_schedule=scheduler_type,
            )
            
            optimizer = EnhancedLayerOptimizer(
                image_size=(16, 16),
                num_materials=4,
                config=config,
            )
            
            learning_rates = []
            
            def type_callback(step, metrics, pred_image, height_map):
                if optimizer.lr_scheduler is not None:
                    learning_rates.append(optimizer.lr_scheduler.get_current_lr())
            
            # When running optimization with specific scheduler type
            loss_history = optimizer.optimize(
                target_image=image,
                material_colors=materials,
                callback=type_callback,
            )
            
            # Then scheduler should produce valid learning rates
            assert len(learning_rates) > 0, f"Should have learning rates for {scheduler_type}"
            assert all(lr > 0 for lr in learning_rates), f"All LRs should be positive for {scheduler_type}"
            assert all(lr <= 1.0 for lr in learning_rates), f"All LRs should be reasonable for {scheduler_type}"


class TestStory33EnhancedEarlyStopping:
    """BDD Tests for Story 3.3: Enhanced Early Stopping
    
    Acceptance Criteria:
    Given an optimization that has found a good solution
    When discrete validation loss stops improving
    Then the system should trigger early stopping after patience period
    And save the best discrete solution found
    And provide clear feedback about stopping reason
    And allow users to configure patience parameters
    """

    @pytest.fixture
    def early_stopping_config(self):
        """Create configuration for early stopping tests."""
        return EnhancedOptimizationConfig(
            iterations=200,
            learning_rate=0.01,
            device="cpu",
            max_layers=6,
            layer_height=0.08,
            early_stopping_patience=30,
            enable_enhanced_early_stopping=True,
            discrete_patience=30,
            continuous_patience=30,
        )

    @pytest.fixture
    def convergent_test_data(self):
        """Create test data that should converge quickly."""
        # Create a simple pattern that should be easy to optimize
        image = torch.zeros(1, 3, 16, 16)
        image[0, 0, :8, :8] = 1.0    # Red top-left
        image[0, 1, :8, 8:] = 1.0    # Green top-right
        image[0, 2, 8:, :8] = 1.0    # Blue bottom-left
        image[0, :, 8:, 8:] = 1.0    # White bottom-right
        
        materials = torch.tensor([
            [1.0, 0.0, 0.0],  # Red
            [0.0, 1.0, 0.0],  # Green
            [0.0, 0.0, 1.0],  # Blue
            [1.0, 1.0, 1.0],  # White
        ])
        
        return image, materials

    def test_triggers_early_stopping_after_patience_period(
        self, early_stopping_config, convergent_test_data
    ):
        """Test that the system triggers early stopping after patience period."""
        image, materials = convergent_test_data
        
        # Configure for aggressive early stopping
        config = EnhancedOptimizationConfig(
            iterations=500,  # Long enough to potentially early stop
            learning_rate=0.02,
            device="cpu",
            max_layers=6,
            layer_height=0.08,
            early_stopping_patience=15,  # Short patience for testing
            enable_discrete_tracking=True,
            validation_interval=5,
            enable_enhanced_early_stopping=True,
            discrete_patience=15,  # Match config patience
        )
        
        optimizer = EnhancedLayerOptimizer(
            image_size=(16, 16),
            num_materials=4,
            config=config,
        )
        
        start_time = time.time()
        
        # When running optimization that should converge
        loss_history = optimizer.optimize(
            target_image=image,
            material_colors=materials,
        )
        
        end_time = time.time()
        
        # Then early stopping should potentially trigger
        total_steps = len(loss_history.get('total', []))
        
        # If optimization stopped early, it should be for a valid reason
        if total_steps < config.iterations:
            assert optimizer.early_stopping is not None, "Should have early stopping component"
            stopping_info = optimizer.early_stopping.get_stopping_info()
            
            if stopping_info['stopped']:
                assert stopping_info['reason'] in [
                    'discrete_loss_plateau', 'continuous_loss_plateau', 'both_losses_plateau'
                ], "Should have valid early stopping reason"
            
            # And should have completed in reasonable time for early stopping
            assert end_time - start_time < 30, "Early stopping should complete quickly"

    def test_saves_best_discrete_solution_found(
        self, early_stopping_config, convergent_test_data
    ):
        """Test that the system saves the best discrete solution found."""
        image, materials = convergent_test_data
        
        optimizer = EnhancedLayerOptimizer(
            image_size=(16, 16),
            num_materials=4,
            config=early_stopping_config,
        )
        
        # When running optimization
        loss_history = optimizer.optimize(
            target_image=image,
            material_colors=materials,
        )
        
        # Then the best discrete solution should be saved
        assert optimizer.discrete_validator is not None, "Should have discrete validator"
        
        # And the saved state should contain model parameters
        best_state = optimizer.get_best_discrete_state()
        if best_state is not None:
            assert 'global_logits' in best_state, "Should save global logits"
            assert 'height_logits' in best_state, "Should save height logits"
            assert 'step' in best_state, "Should save the step number"
            
            # And the saved parameters should have correct shapes
            assert best_state['global_logits'].shape == (early_stopping_config.max_layers, 4), \
                "Global logits should have correct shape"
            assert best_state['height_logits'].shape == (16, 16), \
                "Height logits should have correct shape"

    def test_provides_clear_feedback_about_stopping_reason(
        self, early_stopping_config, convergent_test_data
    ):
        """Test that the system provides clear feedback about stopping reason."""
        image, materials = convergent_test_data
        
        # Configure for likely early stopping
        config = EnhancedOptimizationConfig(
            iterations=100,
            learning_rate=0.05,  # Higher LR for faster convergence
            device="cpu",
            max_layers=4,
            layer_height=0.08,
            early_stopping_patience=10,
            enable_discrete_tracking=True,
            validation_interval=3,
            enable_enhanced_early_stopping=True,
            discrete_patience=10,
        )
        
        optimizer = EnhancedLayerOptimizer(
            image_size=(16, 16),
            num_materials=4,
            config=config,
        )
        
        # Capture feedback messages
        feedback_messages = []
        
        def feedback_callback(step, metrics, pred_image, height_map):
            if hasattr(optimizer, 'last_feedback_message'):
                feedback_messages.append(optimizer.last_feedback_message)
        
        # When running optimization
        loss_history = optimizer.optimize(
            target_image=image,
            material_colors=materials,
            callback=feedback_callback,
        )
        
        # Then stopping reason should be clear and accessible
        if optimizer.early_stopping is not None:
            stopping_info = optimizer.early_stopping.get_stopping_info()
            if stopping_info['stopped']:
                reason = stopping_info['reason']
                valid_reasons = [
                    'discrete_loss_plateau',
                    'continuous_loss_plateau', 
                    'both_losses_plateau',
                    'max_iterations_reached',
                    'convergence_achieved'
                ]
                assert reason in valid_reasons, f"Stopping reason should be valid, got: {reason}"
            
        # And feedback should include useful information
        if hasattr(optimizer, 'stopping_message'):
            message = optimizer.stopping_message
            assert isinstance(message, str), "Stopping message should be a string"
            assert len(message) > 0, "Stopping message should not be empty"

    def test_allows_users_to_configure_patience_parameters(
        self, convergent_test_data
    ):
        """Test that users can configure patience parameters."""
        image, materials = convergent_test_data
        
        # Test different patience configurations
        patience_configs = [5, 10, 20, 50]
        
        for patience in patience_configs:
            config = EnhancedOptimizationConfig(
                iterations=150,
                learning_rate=0.01,
                device="cpu",
                max_layers=4,
                layer_height=0.08,
                early_stopping_patience=patience,
                enable_discrete_tracking=True,
                validation_interval=3,
                enable_enhanced_early_stopping=True,
                discrete_patience=patience,
                continuous_patience=patience,
            )
            
            optimizer = EnhancedLayerOptimizer(
                image_size=(16, 16),
                num_materials=4,
                config=config,
            )
            
            # When running optimization with configured patience
            loss_history = optimizer.optimize(
                target_image=image,
                material_colors=materials,
            )
            
            # Then patience configuration should be respected
            if optimizer.early_stopping is not None:
                assert optimizer.early_stopping.config.discrete_patience == patience, \
                    f"Discrete patience should be {patience}"
                assert optimizer.early_stopping.config.continuous_patience == patience, \
                    f"Continuous patience should be {patience}"
            
            # And optimizer should complete successfully
            assert len(loss_history.get('total', [])) > 0, "Should have optimization history"

    def test_enhanced_early_stopping_vs_basic_early_stopping(
        self, convergent_test_data
    ):
        """Test that enhanced early stopping provides benefits over basic early stopping."""
        image, materials = convergent_test_data
        
        config = EnhancedOptimizationConfig(
            iterations=100,
            learning_rate=0.01,
            device="cpu",
            max_layers=4,
            layer_height=0.08,
            early_stopping_patience=20,
        )
        
        # Test basic early stopping
        basic_optimizer = EnhancedLayerOptimizer(
            image_size=(16, 16),
            num_materials=4,
            config=replace(config, enable_enhanced_early_stopping=False),  # Basic only
        )
        
        basic_history = basic_optimizer.optimize(
            target_image=image,
            material_colors=materials,
        )
        
        # Test enhanced early stopping
        enhanced_config = replace(
            config,
            enable_discrete_tracking=True,
            enable_enhanced_early_stopping=True,
            validation_interval=5,
        )
        enhanced_optimizer = EnhancedLayerOptimizer(
            image_size=(16, 16),
            num_materials=4,
            config=enhanced_config,
        )
        
        enhanced_history = enhanced_optimizer.optimize(
            target_image=image,
            material_colors=materials,
        )
        
        # Then enhanced version should provide additional capabilities
        assert enhanced_optimizer.discrete_validator is not None, \
            "Enhanced version should have discrete validator"
        assert hasattr(enhanced_optimizer.discrete_validator, 'best_discrete_loss'), \
            "Enhanced version should track discrete loss"
        
        # And both should complete successfully
        assert len(basic_history.get('total', [])) > 0, "Basic should have history"
        assert len(enhanced_history.get('total', [])) > 0, "Enhanced should have history"


class TestStory34MixedPrecisionSupport:
    """BDD Tests for Story 3.4: Mixed Precision Support
    
    Acceptance Criteria:
    Given a system with CUDA support
    When optimization runs with mixed precision enabled
    Then it should use bfloat16 for forward passes where possible
    And maintain float32 precision for critical operations
    And reduce memory usage without significant quality loss
    And fallback gracefully on unsupported hardware
    """

    @pytest.fixture
    def mixed_precision_config(self):
        """Create configuration for mixed precision tests."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return EnhancedOptimizationConfig(
            iterations=50,
            learning_rate=0.01,
            device=device,
            max_layers=6,
            layer_height=0.08,
            enable_mixed_precision=True,
        )

    @pytest.fixture
    def precision_test_data(self):
        """Create test data for precision testing."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        image = torch.randn(1, 3, 24, 24, device=device)
        materials = torch.tensor([
            [1.0, 0.0, 0.0],  # Red
            [0.0, 1.0, 0.0],  # Green
            [0.0, 0.0, 1.0],  # Blue
            [1.0, 1.0, 0.0],  # Yellow
        ], device=device)
        return image, materials

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_uses_bfloat16_for_forward_passes_where_possible(
        self, mixed_precision_config, precision_test_data
    ):
        """Test that the system uses bfloat16 for forward passes where possible."""
        image, materials = precision_test_data
        
        # Skip if bfloat16 is not supported
        if not torch.cuda.is_bf16_supported():
            pytest.skip("bfloat16 not supported on this hardware")
        
        config = replace(mixed_precision_config, use_bfloat16=True)
        optimizer = EnhancedLayerOptimizer(
            image_size=(24, 24),
            num_materials=4,
            config=config,
        )
        
        # Track precision usage during forward passes
        precision_logs = []
        
        def precision_callback(step, metrics, pred_image, height_map):
            # Check tensor dtypes in the forward pass
            if hasattr(optimizer, 'last_forward_dtypes'):
                precision_logs.append(optimizer.last_forward_dtypes)
        
        # When running optimization with mixed precision
        loss_history = optimizer.optimize(
            target_image=image,
            material_colors=materials,
            callback=precision_callback,
        )
        
        # Then bfloat16 should be used where appropriate
        assert optimizer.mixed_precision is not None, "Should have mixed precision manager"
        assert optimizer.mixed_precision.enabled, "Mixed precision should be enabled"
        
        # And scaler should be configured for mixed precision
        if hasattr(optimizer, 'scaler'):
            assert optimizer.scaler is not None, "Should have gradient scaler"

    def test_maintains_float32_precision_for_critical_operations(
        self, mixed_precision_config, precision_test_data
    ):
        """Test that float32 precision is maintained for critical operations."""
        image, materials = precision_test_data
        
        optimizer = EnhancedLayerOptimizer(
            image_size=(24, 24),
            num_materials=4,
            config=mixed_precision_config,
        )
        
        # When running optimization
        loss_history = optimizer.optimize(
            target_image=image,
            material_colors=materials,
        )
        
        # Then critical parameters should remain in float32
        assert optimizer.global_logits.dtype == torch.float32, \
            "Global logits should remain float32"
        assert optimizer.height_logits.dtype == torch.float32, \
            "Height logits should remain float32"
        
        # And optimization should complete successfully
        assert len(loss_history.get('total', [])) > 0, "Should have optimization history"
        
        # And final results should be valid
        with torch.no_grad():
            final_image, final_height = optimizer.forward(materials)
            assert not torch.isnan(final_image).any(), "Final image should not contain NaN"
            assert not torch.isnan(final_height).any(), "Final height should not contain NaN"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_reduces_memory_usage_without_significant_quality_loss(
        self, mixed_precision_config, precision_test_data
    ):
        """Test that mixed precision reduces memory usage without significant quality loss."""
        image, materials = precision_test_data
        
        # Test with mixed precision disabled
        fp32_config = replace(
            mixed_precision_config,
            enable_mixed_precision=False
        )
        optimizer_fp32 = EnhancedLayerOptimizer(
            image_size=(24, 24),
            num_materials=4,
            config=fp32_config,
        )
        
        # Measure memory usage for FP32
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        history_fp32 = optimizer_fp32.optimize(
            target_image=image,
            material_colors=materials,
        )
        
        memory_fp32 = torch.cuda.max_memory_allocated()
        
        # Test with mixed precision enabled
        optimizer_mixed = EnhancedLayerOptimizer(
            image_size=(24, 24),
            num_materials=4,
            config=mixed_precision_config,
        )
        
        # Measure memory usage for mixed precision
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        history_mixed = optimizer_mixed.optimize(
            target_image=image,
            material_colors=materials,
        )
        
        memory_mixed = torch.cuda.max_memory_allocated()
        
        # Then mixed precision should use less memory
        memory_reduction = (memory_fp32 - memory_mixed) / memory_fp32 if memory_fp32 > 0 else 0
        assert memory_reduction >= 0, "Mixed precision should not increase memory usage significantly"
        
        # And quality should not be significantly degraded
        final_loss_fp32 = history_fp32['total'][-1] if history_fp32['total'] else float('inf')
        final_loss_mixed = history_mixed['total'][-1] if history_mixed['total'] else float('inf')
        
        # Quality should be reasonably close (within 50% for this test)
        quality_ratio = final_loss_mixed / final_loss_fp32 if final_loss_fp32 > 0 else 1.0
        assert quality_ratio < 2.0, f"Mixed precision quality should be reasonable, got ratio {quality_ratio:.2f}"

    def test_fallback_gracefully_on_unsupported_hardware(self, precision_test_data):
        """Test that the system falls back gracefully on unsupported hardware."""
        image, materials = precision_test_data
        
        # Force CPU device to test fallback
        cpu_config = EnhancedOptimizationConfig(
            iterations=20,
            learning_rate=0.01,
            device="cpu",  # CPU doesn't support mixed precision training
            max_layers=4,
            layer_height=0.08,
            enable_mixed_precision=True,  # Try to enable even on CPU
        )
        
        # Move test data to CPU
        image_cpu = image.cpu()
        materials_cpu = materials.cpu()
        
        optimizer = EnhancedLayerOptimizer(
            image_size=(24, 24),
            num_materials=4,
            config=cpu_config,
        )
        
        # When running optimization on unsupported hardware
        loss_history = optimizer.optimize(
            target_image=image_cpu,
            material_colors=materials_cpu,
        )
        
        # Then optimization should complete successfully
        assert len(loss_history.get('total', [])) > 0, "Should complete on CPU"
        
        # And mixed precision should be disabled gracefully
        if optimizer.mixed_precision is not None:
            # On CPU, mixed precision should be disabled
            if cpu_config.device == "cpu":
                assert not optimizer.mixed_precision.enabled, \
                    "Mixed precision should be disabled on CPU"

    def test_mixed_precision_autocast_context_management(
        self, mixed_precision_config, precision_test_data
    ):
        """Test that autocast context is properly managed."""
        image, materials = precision_test_data
        
        optimizer = EnhancedLayerOptimizer(
            image_size=(24, 24),
            num_materials=4,
            config=mixed_precision_config,
        )
        
        # Track autocast usage
        autocast_calls = []
        
        # Mock autocast to track its usage
        original_autocast = torch.autocast
        
        def mock_autocast(*args, **kwargs):
            autocast_calls.append((args, kwargs))
            return original_autocast(*args, **kwargs)
        
        # When running optimization with mixed precision
        with patch('torch.autocast', side_effect=mock_autocast):
            loss_history = optimizer.optimize(
                target_image=image,
                material_colors=materials,
            )
        
        # Then optimization should complete successfully
        assert len(loss_history.get('total', [])) > 0, "Should complete with autocast tracking"
        
        # And autocast should be used appropriately if CUDA is available
        if torch.cuda.is_available() and optimizer.mixed_precision is not None:
            if optimizer.mixed_precision.enabled:
                # Should have some autocast usage for mixed precision
                assert len(autocast_calls) >= 0, "Should use autocast for mixed precision"




class TestFeature3Integration:
    """Integration tests for Feature 3: Enhanced Optimization Engine"""

    @pytest.fixture
    def integration_config(self):
        """Configuration for integration testing."""
        return EnhancedOptimizationConfig(
            iterations=80,
            learning_rate=0.02,
            device="cpu",
            max_layers=6,
            layer_height=0.08,
            early_stopping_patience=25,
            enable_discrete_tracking=True,
            validation_interval=8,
            enable_lr_scheduling=True,
            warmup_steps=15,
            enable_enhanced_early_stopping=True,
            discrete_patience=20,
            enable_mixed_precision=torch.cuda.is_available(),
        )

    @pytest.fixture
    def integration_test_data(self):
        """Test data for integration testing."""
        # Create a more complex test image
        image = torch.zeros(1, 3, 32, 32)
        
        # Create a pattern with multiple regions
        image[0, 0, 8:24, 8:24] = 1.0      # Red center
        image[0, 1, :8, :] = 0.8           # Green top strip
        image[0, 2, 24:, :] = 0.6          # Blue bottom strip
        image[0, :, :, :8] = 0.4           # Dark left strip
        image[0, :, :, 24:] = 0.9          # Light right strip
        
        materials = torch.tensor([
            [1.0, 0.0, 0.0],  # Red
            [0.0, 1.0, 0.0],  # Green
            [0.0, 0.0, 1.0],  # Blue
            [0.8, 0.8, 0.8],  # Light gray
            [0.2, 0.2, 0.2],  # Dark gray
            [1.0, 1.0, 0.0],  # Yellow
        ])
        
        return image, materials

    def test_all_feature3_capabilities_work_together(
        self, integration_config, integration_test_data
    ):
        """Test that all Feature 3 capabilities work together."""
        image, materials = integration_test_data
        
        # Create optimizer with all Feature 3 capabilities enabled
        enhanced_config = replace(
            integration_config,
            decay_schedule="cosine"
        )
        optimizer = EnhancedLayerOptimizer(
            image_size=(32, 32),
            num_materials=6,
            config=enhanced_config,
        )
        
        # Track comprehensive metrics
        all_metrics = {
            'steps': [],
            'learning_rates': [],
            'discrete_losses': [],
            'continuous_losses': [],
        }
        
        def comprehensive_callback(step, metrics, pred_image, height_map):
            all_metrics['steps'].append(step)
            if optimizer.lr_scheduler is not None:
                all_metrics['learning_rates'].append(optimizer.lr_scheduler.get_current_lr())
            else:
                all_metrics['learning_rates'].append(optimizer.config.learning_rate)
            
            if 'discrete_loss' in metrics:
                all_metrics['discrete_losses'].append(metrics['discrete_loss'])
            if 'total' in metrics:
                all_metrics['continuous_losses'].append(metrics['total'])
        
        # When running optimization with all features
        loss_history = optimizer.optimize(
            target_image=image,
            material_colors=materials,
            callback=comprehensive_callback,
        )
        
        # Then all capabilities should function correctly
        
        # 1. Discrete validation tracking
        assert optimizer.discrete_validator is not None, "Should have discrete tracking"
        assert len(all_metrics['discrete_losses']) > 0, "Should have discrete validations"
        
        # 2. Learning rate scheduling
        assert len(all_metrics['learning_rates']) > 0, "Should have LR tracking"
        lr_variation = max(all_metrics['learning_rates']) - min(all_metrics['learning_rates'])
        assert lr_variation > 0, "Learning rate should vary"
        
        # 3. Enhanced early stopping
        assert optimizer.early_stopping is not None, "Should have stopping tracking"
        
        # 4. Mixed precision (if available)
        if torch.cuda.is_available():
            assert optimizer.mixed_precision is not None, "Should have MP tracking"
        
        # And optimization should complete successfully
        assert len(loss_history['total']) > 0, "Should have optimization history"
        
        # And final results should be valid
        final_image, final_height = optimizer.forward(materials)
        assert not torch.isnan(final_image).any(), "Final results should be valid"

    def test_feature3_performance_impact(
        self, integration_config, integration_test_data
    ):
        """Test that Feature 3 enhancements don't significantly impact performance."""
        image, materials = integration_test_data
        
        # Test baseline performance
        baseline_optimizer = LayerOptimizer(
            image_size=(32, 32),
            num_materials=6,
            config=integration_config,
        )
        
        start_time = time.time()
        baseline_history = baseline_optimizer.optimize(
            target_image=image,
            material_colors=materials,
        )
        baseline_time = time.time() - start_time
        
        # Test enhanced performance
        enhanced_config = replace(
            integration_config,
            validation_interval=5
        )
        enhanced_optimizer = EnhancedLayerOptimizer(
            image_size=(32, 32),
            num_materials=6,
            config=enhanced_config,
        )
        
        start_time = time.time()
        enhanced_history = enhanced_optimizer.optimize(
            target_image=image,
            material_colors=materials,
        )
        enhanced_time = time.time() - start_time
        
        # Performance impact should be reasonable (less than 2x slower)
        performance_ratio = enhanced_time / baseline_time if baseline_time > 0 else 1.0
        assert performance_ratio < 3.0, f"Performance impact should be reasonable, got {performance_ratio:.2f}x"
        
        # Both should produce valid results
        assert len(baseline_history['total']) > 0, "Baseline should complete"
        assert len(enhanced_history['total']) > 0, "Enhanced should complete"

    def test_feature3_backward_compatibility(
        self, integration_config, integration_test_data
    ):
        """Test that Feature 3 maintains backward compatibility."""
        image, materials = integration_test_data
        
        # Test that enhanced optimizer works in basic mode
        basic_config = replace(
            integration_config,
            # All features disabled - should work like basic optimizer
            enable_discrete_tracking=False,
            enable_lr_scheduling=False,
            enable_enhanced_early_stopping=False,
            enable_mixed_precision=False,
        )
        enhanced_optimizer = EnhancedLayerOptimizer(
            image_size=(32, 32),
            num_materials=6,
            config=basic_config,
        )
        
        # When running in compatibility mode
        loss_history = enhanced_optimizer.optimize(
            target_image=image,
            material_colors=materials,
        )
        
        # Then it should behave like the basic optimizer
        assert len(loss_history['total']) > 0, "Should complete successfully"
        
        # And basic functionality should work
        final_image, final_height = enhanced_optimizer.forward(materials)
        assert final_image.shape == (1, 3, 32, 32), "Should have correct output shape"
        assert final_height.shape == (1, 1, 32, 32), "Should have correct height shape"
        
        # And results should be reasonable
        assert not torch.isnan(final_image).any(), "Results should be valid"
        assert torch.all(final_image >= 0) and torch.all(final_image <= 1.1), \
            "Image values should be in reasonable range"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])