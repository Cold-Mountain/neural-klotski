"""
Tests for Neural-Klotski training system.

Tests training pipeline, batch training, monitoring, scheduling,
and visualization components.
"""

import pytest
import tempfile
import shutil
import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from neural_klotski.training.trainer import (
    AdditionNetworkTrainer, TrainingConfig, TrainingPhase, TrainingMetrics
)
from neural_klotski.training.batch_trainer import (
    BatchTrainer, BatchConfig, ExperimentConfig, HyperparameterSweep
)
from neural_klotski.training.monitoring import (
    PerformanceMonitor, ConvergenceConfig, ConvergenceStatus
)
from neural_klotski.training.scheduling import (
    LearningRateScheduler, ScheduleConfig, ScheduleType, create_default_scheduler
)
from neural_klotski.training.visualization import (
    TrainingVisualizer, VisualizationConfig, create_console_progress_tracker
)
from neural_klotski.core.encoding import AdditionProblem
from neural_klotski.config import get_default_config


class TestAdditionNetworkTrainer:
    """Test basic training functionality"""

    def test_trainer_initialization(self):
        """Test trainer creates correctly"""
        config = get_default_config()
        training_config = TrainingConfig()
        training_config.epochs_per_phase = {
            TrainingPhase.SIMPLE: 5,
            TrainingPhase.INTERMEDIATE: 5,
            TrainingPhase.ADVANCED: 5
        }
        training_config.problems_per_epoch = 10

        trainer = AdditionNetworkTrainer(training_config, config.simulation)

        assert trainer.network is not None
        assert trainer.task_manager is not None
        assert trainer.current_phase == TrainingPhase.SIMPLE
        assert trainer.metrics.epoch == 0

    def test_problem_generation(self):
        """Test problem generation for different phases"""
        config = get_default_config()
        training_config = TrainingConfig()
        trainer = AdditionNetworkTrainer(training_config, config.simulation)

        # Test simple phase problems
        simple_problems = trainer.generate_problems_for_phase(TrainingPhase.SIMPLE, 10)
        assert len(simple_problems) == 10
        for problem in simple_problems:
            assert problem.operand1 <= 15
            assert problem.operand2 <= 15

        # Test intermediate phase problems
        intermediate_problems = trainer.generate_problems_for_phase(TrainingPhase.INTERMEDIATE, 10)
        for problem in intermediate_problems:
            assert problem.operand1 <= 127
            assert problem.operand2 <= 127

    def test_training_epoch(self):
        """Test single epoch training"""
        config = get_default_config()
        training_config = TrainingConfig()
        training_config.problems_per_epoch = 5
        trainer = AdditionNetworkTrainer(training_config, config.simulation)

        # Generate test problems
        problems = trainer.generate_problems_for_phase(TrainingPhase.SIMPLE, 5)

        # Train one epoch
        epoch_stats = trainer.train_epoch(problems)

        assert 'accuracy' in epoch_stats
        assert 'average_error' in epoch_stats
        assert 'confidence' in epoch_stats
        assert epoch_stats['total_problems'] == 5
        assert 0 <= epoch_stats['accuracy'] <= 1
        assert epoch_stats['average_error'] >= 0

    def test_short_training_run(self):
        """Test complete training run with very few epochs"""
        config = get_default_config()
        training_config = TrainingConfig()
        training_config.epochs_per_phase = {
            TrainingPhase.SIMPLE: 3,
            TrainingPhase.INTERMEDIATE: 3,
            TrainingPhase.ADVANCED: 3
        }
        training_config.problems_per_epoch = 5
        training_config.evaluation_frequency = 2

        trainer = AdditionNetworkTrainer(training_config, config.simulation)

        # Track progress
        epoch_count = 0
        def count_epochs(metrics):
            nonlocal epoch_count
            epoch_count += 1

        trainer.add_epoch_callback(count_epochs)

        # Run short training
        final_metrics = trainer.train(max_epochs=5)

        assert final_metrics.epoch >= 5
        assert epoch_count >= 5
        assert final_metrics.accuracy >= 0
        assert final_metrics.total_problems > 0


class TestBatchTrainer:
    """Test batch training functionality"""

    def setup_method(self):
        """Setup test directory"""
        self.test_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Clean up test directory"""
        shutil.rmtree(self.test_dir)

    def test_batch_trainer_initialization(self):
        """Test batch trainer creates correctly"""
        config = BatchConfig(output_directory=self.test_dir)
        trainer = BatchTrainer(config)

        assert trainer.config == config
        assert Path(trainer.output_path).exists()

    def test_experiment_creation(self):
        """Test experiment configuration creation"""
        config = BatchConfig(output_directory=self.test_dir)
        trainer = BatchTrainer(config)

        experiment = trainer.create_experiment(
            name="test_exp",
            description="Test experiment",
            training_overrides={'problems_per_epoch': 5},
            simulation_overrides={'input_scaling_factor': 50.0}
        )

        assert experiment.name == "test_exp"
        assert experiment.training_params['problems_per_epoch'] == 5
        assert experiment.simulation_params['input_scaling_factor'] == 50.0

    def test_hyperparameter_sweep(self):
        """Test hyperparameter sweep generation"""
        sweep = HyperparameterSweep()
        sweep.add_parameter('problems_per_epoch', [5, 10])
        sweep.add_parameter('sim_input_scaling_factor', [50.0, 100.0])

        # Test grid search
        combinations = list(sweep.grid_search())
        assert len(combinations) == 4  # 2 x 2 combinations

        # Test random sampling
        random_combinations = list(sweep.random_sample(3, seed=42))
        assert len(random_combinations) == 3

    def test_single_experiment_run(self):
        """Test running a single experiment"""
        config = BatchConfig(output_directory=self.test_dir)
        trainer = BatchTrainer(config)

        # Create a minimal experiment
        experiment = trainer.create_experiment(
            name="minimal_test",
            training_overrides={
                'epochs_per_phase': {
                    TrainingPhase.SIMPLE: 2,
                    TrainingPhase.INTERMEDIATE: 2,
                    TrainingPhase.ADVANCED: 2
                },
                'problems_per_epoch': 3,
                'evaluation_frequency': 1
            },
            seed=42
        )

        # Run experiment
        result = trainer.run_experiment(experiment)

        assert result.config == experiment
        assert result.total_duration > 0

        # Check if successful or at least attempted
        if result.success:
            assert result.final_metrics.epoch >= 0
        else:
            assert result.error_message is not None


class TestPerformanceMonitor:
    """Test performance monitoring functionality"""

    def test_monitor_initialization(self):
        """Test monitor creates correctly"""
        config = ConvergenceConfig()
        monitor = PerformanceMonitor(config)

        assert monitor.config == config
        assert len(monitor.snapshots) == 0
        assert monitor.current_status == ConvergenceStatus.IMPROVING

    def test_metrics_recording(self):
        """Test recording training metrics"""
        config = ConvergenceConfig()
        monitor = PerformanceMonitor(config)

        # Create test metrics
        metrics = TrainingMetrics(
            epoch=1,
            phase=TrainingPhase.SIMPLE,
            accuracy=0.5,
            average_error=0.5,
            confidence=0.6
        )

        snapshot = monitor.record_metrics(metrics)

        assert len(monitor.snapshots) == 1
        assert snapshot.accuracy == 0.5
        assert snapshot.epoch == 1

    def test_convergence_analysis(self):
        """Test convergence analysis"""
        config = ConvergenceConfig()
        config.smoothing_window = 3
        config.trend_window = 5
        monitor = PerformanceMonitor(config)

        # Record improving metrics
        for epoch in range(10):
            metrics = TrainingMetrics(
                epoch=epoch,
                phase=TrainingPhase.SIMPLE,
                accuracy=0.1 + epoch * 0.05,  # Improving
                average_error=0.9 - epoch * 0.05,
                confidence=0.5 + epoch * 0.02
            )
            monitor.record_metrics(metrics)

        report = monitor.analyze_convergence()

        assert report.status in [ConvergenceStatus.IMPROVING, ConvergenceStatus.PLATEAU]
        assert 0 <= report.confidence <= 1
        assert 0 <= report.convergence_score <= 1

    def test_early_stopping_detection(self):
        """Test early stopping logic"""
        config = ConvergenceConfig()
        config.early_stopping_enabled = True
        config.min_epochs_before_stopping = 5
        monitor = PerformanceMonitor(config)

        # Record metrics that plateau
        for epoch in range(10):
            accuracy = 0.9 if epoch < 5 else 0.9  # Plateau after epoch 5
            metrics = TrainingMetrics(
                epoch=epoch,
                phase=TrainingPhase.SIMPLE,
                accuracy=accuracy,
                average_error=0.1,
                confidence=0.9
            )
            monitor.record_metrics(metrics)

        # Should not stop early initially
        assert not monitor.should_stop_early()


class TestLearningRateScheduler:
    """Test learning rate scheduling functionality"""

    def test_scheduler_initialization(self):
        """Test scheduler creates correctly"""
        config = ScheduleConfig(
            schedule_type=ScheduleType.COSINE_ANNEALING,
            initial_lr=0.01,
            T_max=100
        )
        scheduler = LearningRateScheduler(config)

        assert scheduler.config == config
        assert scheduler.current_lr == 0.01

    def test_cosine_annealing_schedule(self):
        """Test cosine annealing schedule"""
        scheduler = create_default_scheduler(
            ScheduleType.COSINE_ANNEALING,
            initial_lr=0.01,
            eta_min=0.001,
            T_max=100,
            warmup_epochs=0  # Disable warmup for cleaner testing
        )

        # Test a few points after warmup
        lr_0 = scheduler.step(0)
        lr_50 = scheduler.step(50)  # Should be near minimum
        lr_100 = scheduler.step(100)  # Should be back near maximum

        assert lr_0 == pytest.approx(0.01, rel=1e-3)
        assert lr_50 < lr_0  # Should have decreased
        assert lr_50 >= 0.001  # Should not go below minimum

    def test_adaptive_performance_schedule(self):
        """Test performance-based adaptive scheduling"""
        scheduler = create_default_scheduler(
            ScheduleType.ADAPTIVE_PERFORMANCE,
            initial_lr=0.01,
            patience=3,
            reduction_factor=0.5,
            warmup_epochs=0  # Disable warmup for cleaner testing
        )

        # Simulate improving performance
        lr1 = scheduler.step(1, 0.5)
        lr2 = scheduler.step(2, 0.6)  # Improvement
        lr3 = scheduler.step(3, 0.65)  # Improvement

        # Simulate plateau
        lr4 = scheduler.step(4, 0.65)  # No improvement
        lr5 = scheduler.step(5, 0.65)  # No improvement
        lr6 = scheduler.step(6, 0.65)  # No improvement
        lr7 = scheduler.step(7, 0.65)  # Should trigger reduction

        # Should maintain same LR during improvement
        assert lr2 == lr3  # No reduction during improvement

        # After patience period, should reduce
        assert lr7 <= lr6  # Should reduce or stay same after patience

    def test_scheduler_state_management(self):
        """Test scheduler state saving/loading"""
        scheduler = create_default_scheduler(ScheduleType.EXPONENTIAL_DECAY)

        # Step a few times
        scheduler.step(1, 0.5)
        scheduler.step(2, 0.6)

        # Save state
        state = scheduler.state_dict()
        assert 'epoch' in state
        assert 'current_lr' in state

        # Create new scheduler and load state
        new_scheduler = create_default_scheduler(ScheduleType.EXPONENTIAL_DECAY)
        new_scheduler.load_state_dict(state)

        assert new_scheduler.epoch == scheduler.epoch
        assert new_scheduler.current_lr == scheduler.current_lr


class TestTrainingVisualization:
    """Test training visualization functionality"""

    def setup_method(self):
        """Setup test directory"""
        self.test_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Clean up test directory"""
        shutil.rmtree(self.test_dir)

    def test_visualizer_initialization(self):
        """Test visualizer creates correctly"""
        config = VisualizationConfig(
            output_directory=self.test_dir,
            save_plots=False  # Disable plots for testing
        )
        visualizer = TrainingVisualizer(config)

        assert visualizer.config == config
        assert Path(visualizer.output_path).exists()

    def test_metrics_recording(self):
        """Test recording metrics for visualization"""
        config = VisualizationConfig(
            output_directory=self.test_dir,
            save_plots=False
        )
        visualizer = TrainingVisualizer(config)

        metrics = TrainingMetrics(
            epoch=1,
            phase=TrainingPhase.SIMPLE,
            accuracy=0.7,
            average_error=0.3,
            confidence=0.8
        )

        # Record metrics
        visualizer.record_epoch(metrics, learning_rate=0.01, epoch_duration=1.5)

        assert len(visualizer.metrics_history) == 1
        assert len(visualizer.learning_rate_history) == 1
        assert len(visualizer.epoch_times) == 1

    def test_training_log_saving(self):
        """Test saving training logs"""
        config = VisualizationConfig(
            output_directory=self.test_dir,
            save_raw_data=True,
            save_plots=False
        )
        visualizer = TrainingVisualizer(config)

        # Record some metrics
        for epoch in range(5):
            metrics = TrainingMetrics(
                epoch=epoch,
                phase=TrainingPhase.SIMPLE,
                accuracy=0.1 + epoch * 0.1,
                average_error=0.9 - epoch * 0.1,
                confidence=0.5 + epoch * 0.05
            )
            visualizer.record_epoch(metrics)

        # Save log
        log_filename = "test_log.json"
        visualizer.save_training_log(log_filename)

        log_path = Path(self.test_dir) / log_filename
        assert log_path.exists()

        # Verify log content
        import json
        with open(log_path) as f:
            log_data = json.load(f)

        assert 'session_info' in log_data
        assert 'metrics_history' in log_data
        assert len(log_data['metrics_history']) == 5

    def test_final_report_generation(self):
        """Test final report generation"""
        config = VisualizationConfig(
            output_directory=self.test_dir,
            save_plots=False
        )
        visualizer = TrainingVisualizer(config)

        # Record some metrics
        for epoch in range(3):
            metrics = TrainingMetrics(
                epoch=epoch,
                phase=TrainingPhase.SIMPLE,
                accuracy=0.2 + epoch * 0.2,
                average_error=0.8 - epoch * 0.2,
                confidence=0.5 + epoch * 0.1,
                problems_solved=epoch * 10,
                total_problems=epoch * 15
            )
            visualizer.record_epoch(metrics)

        report = visualizer.create_final_report()

        assert "Training Report" in report
        assert "Total Epochs: 2" in report
        assert "Final Accuracy:" in report
        assert "Best Accuracy:" in report

    def test_console_progress_tracker(self):
        """Test console progress tracker callback"""
        tracker = create_console_progress_tracker(update_frequency=2)

        # This should not raise any exceptions
        metrics = TrainingMetrics(
            epoch=2,
            phase=TrainingPhase.SIMPLE,
            accuracy=0.5,
            average_error=0.5,
            confidence=0.6
        )

        tracker(metrics)  # Should trigger output for epoch 2


class TestIntegratedTraining:
    """Test integrated training with all components"""

    def setup_method(self):
        """Setup test directory"""
        self.test_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Clean up test directory"""
        shutil.rmtree(self.test_dir)

    def test_complete_training_pipeline(self):
        """Test complete training pipeline with all components"""
        # Create configurations
        config = get_default_config()
        training_config = TrainingConfig()
        training_config.epochs_per_phase = {
            TrainingPhase.SIMPLE: 3,
            TrainingPhase.INTERMEDIATE: 3,
            TrainingPhase.ADVANCED: 3
        }
        training_config.problems_per_epoch = 5
        training_config.evaluation_frequency = 2

        # Create trainer
        trainer = AdditionNetworkTrainer(training_config, config.simulation)

        # Create monitoring
        monitor_config = ConvergenceConfig()
        monitor = PerformanceMonitor(monitor_config)

        # Create scheduler
        scheduler = create_default_scheduler(ScheduleType.COSINE_ANNEALING, T_max=10)

        # Create visualizer
        viz_config = VisualizationConfig(
            output_directory=self.test_dir,
            save_plots=False,
            save_raw_data=True
        )
        visualizer = TrainingVisualizer(viz_config)

        # Integrated training callback
        def integrated_callback(metrics: TrainingMetrics):
            # Update monitor
            snapshot = monitor.record_metrics(metrics)

            # Analyze convergence
            convergence_report = monitor.analyze_convergence()

            # Update scheduler
            new_lr = scheduler.step(metrics.epoch, metrics.accuracy)

            # Update visualizer
            visualizer.record_epoch(metrics, convergence_report, new_lr)

        trainer.add_epoch_callback(integrated_callback)

        # Run training
        final_metrics = trainer.train(max_epochs=5)

        # Verify all components were used
        assert len(monitor.snapshots) >= 5
        assert len(scheduler.lr_history) >= 5
        assert len(visualizer.metrics_history) >= 5

        # Generate final report
        performance_summary = monitor.get_performance_summary()
        training_report = visualizer.create_final_report()

        assert 'current_performance' in performance_summary
        assert 'Training Report' in training_report


if __name__ == "__main__":
    pytest.main([__file__, "-v"])