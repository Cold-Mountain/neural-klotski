#!/usr/bin/env python3
"""
Minimal test of Neural-Klotski basic functionality.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_basic_functionality():
    """Test basic system functionality without training"""
    print("🔧 Testing Basic System Functionality")
    print("=" * 40)

    try:
        # Test imports
        print("1. Testing imports...")
        from neural_klotski.core.architecture import create_addition_network
        from neural_klotski.core.encoding import AdditionProblem, AdditionTaskManager
        from neural_klotski.config import get_default_config
        print("   ✓ All imports successful")

        # Test configuration
        print("2. Testing configuration...")
        config = get_default_config()
        print(f"   ✓ Config loaded: {len(config.__dict__)} parameters")

        # Test network creation
        print("3. Testing network creation...")
        network = create_addition_network(enable_learning=False)
        print(f"   ✓ Network created with {len(network.blocks)} blocks")

        # Test task manager
        print("4. Testing task manager...")
        task_manager = AdditionTaskManager(config.simulation)
        print("   ✓ Task manager created")

        # Test problem creation
        print("5. Testing problem encoding...")
        problem = AdditionProblem(2, 3, 5)
        print(f"   ✓ Problem created: {problem.operand1} + {problem.operand2} = {problem.expected_sum}")

        # Test basic execution (no training)
        print("6. Testing basic execution...")
        network.reset_simulation()
        result, stats = task_manager.execute_addition_task(network, problem)
        print(f"   ✓ Execution completed")
        print(f"   - Network output: {result.decoded_sum}")
        print(f"   - Expected: {problem.expected_sum}")
        print(f"   - Confidence: {result.confidence:.3f}")
        print(f"   - Correct: {result.decoded_sum == problem.expected_sum}")

        print("\n✅ Basic functionality test PASSED")
        print("\n💡 Analysis:")
        print("   The system can:")
        print("   - Load configuration and create networks")
        print("   - Encode arithmetic problems")
        print("   - Execute inference (produce outputs)")
        print("   - Decode results to integers")
        print("\n   For actual learning capability, training is required.")
        print("   Untrained networks produce essentially random outputs.")

        return True

    except Exception as e:
        print(f"❌ Test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_basic_functionality()