#!/usr/bin/env python3
"""
Test Neural-Klotski training capability.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_minimal_training():
    """Test minimal training to see if the system can learn"""
    print("🎓 Testing Minimal Training")
    print("=" * 40)

    try:
        from neural_klotski.training.trainer import AdditionNetworkTrainer, TrainingConfig, TrainingPhase
        from neural_klotski.core.encoding import AdditionProblem, AdditionTaskManager
        from neural_klotski.config import get_default_config

        print("1. Setting up minimal training...")
        config = get_default_config()

        # Very minimal training configuration
        training_config = TrainingConfig()
        training_config.epochs_per_phase = {
            TrainingPhase.SIMPLE: 5,        # Just 5 epochs
            TrainingPhase.INTERMEDIATE: 0,  # Skip
            TrainingPhase.ADVANCED: 0       # Skip
        }
        training_config.problems_per_epoch = 5  # Just 5 problems per epoch

        print("2. Creating trainer...")
        trainer = AdditionNetworkTrainer(training_config, config.simulation)

        print("3. Running minimal training (5 epochs, 5 problems each)...")
        final_metrics = trainer.train()

        print(f"   ✓ Training completed")
        print(f"   - Final accuracy: {final_metrics.accuracy:.3f}")
        print(f"   - Total epochs: {final_metrics.total_epochs}")
        print(f"   - Average confidence: {final_metrics.average_confidence:.3f}")

        # Test the trained network on a few problems
        print("4. Testing trained network...")
        task_manager = AdditionTaskManager(config.simulation)

        test_cases = [(1, 1, 2), (2, 2, 4), (3, 1, 4)]
        correct = 0

        for a, b, expected in test_cases:
            problem = AdditionProblem(a, b, expected)
            trainer.network.reset_simulation()
            result, _ = task_manager.execute_addition_task(trainer.network, problem)

            is_correct = result.decoded_sum == expected
            if is_correct:
                correct += 1
            print(f"   {a} + {b} = {expected} | Network: {result.decoded_sum} | {'✓' if is_correct else '✗'}")

        test_accuracy = correct / len(test_cases)
        print(f"\n📊 Results:")
        print(f"   Training reported accuracy: {final_metrics.accuracy:.1%}")
        print(f"   Manual test accuracy: {test_accuracy:.1%}")

        print(f"\n💡 Analysis:")
        if final_metrics.accuracy > 0.1:
            print("   ✓ System shows learning capability (accuracy > random)")
        else:
            print("   ⚠ System shows minimal learning (may need more training)")

        if test_accuracy > 0:
            print("   ✓ Network can solve some addition problems correctly")
        else:
            print("   ⚠ Network struggles with tested problems")

        return True, final_metrics.accuracy, test_accuracy

    except Exception as e:
        print(f"❌ Training test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False, 0, 0

if __name__ == "__main__":
    success, train_acc, test_acc = test_minimal_training()

    print(f"\n🎯 SUMMARY:")
    print(f"   System functionality: {'✓ Working' if success else '✗ Failed'}")
    if success:
        print(f"   Learning capability: {'✓ Yes' if train_acc > 0.1 else '? Minimal'}")
        print(f"   Addition capability: {'✓ Yes' if test_acc > 0 else '? Needs more training'}")

        if train_acc > 0.5 and test_acc > 0.5:
            print("\n✅ Neural-Klotski CAN perform addition with training!")
        elif train_acc > 0.1 or test_acc > 0:
            print("\n🟡 Neural-Klotski shows addition potential, needs more training")
        else:
            print("\n🔍 Neural-Klotski needs investigation - very minimal learning")