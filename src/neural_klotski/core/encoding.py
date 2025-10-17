"""
Input/Output encoding system for Neural-Klotski addition networks.

Implements binary number encoding for inputs and sum decoding for outputs
according to the addition task specification.
"""

from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from enum import Enum
import numpy as np
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from neural_klotski.core.block import BlockState, BlockColor
from neural_klotski.core.network import NeuralKlotskiNetwork
from neural_klotski.core.architecture import ShelfType
from neural_klotski.config import SimulationConfig


class EncodingType(Enum):
    """Types of number encoding"""
    BINARY = "binary"
    UNARY = "unary"
    THERMOMETER = "thermometer"


@dataclass
class AdditionProblem:
    """Represents an addition problem"""
    operand1: int
    operand2: int
    expected_sum: int
    max_bits: int = 10  # Maximum bits per operand

    def __post_init__(self):
        """Validate addition problem"""
        if self.operand1 < 0 or self.operand2 < 0:
            raise ValueError("Operands must be non-negative")
        if self.operand1.bit_length() > self.max_bits or self.operand2.bit_length() > self.max_bits:
            raise ValueError(f"Operands exceed {self.max_bits} bits")
        if self.expected_sum != self.operand1 + self.operand2:
            raise ValueError("Expected sum doesn't match operand sum")

    def __repr__(self) -> str:
        return f"AdditionProblem({self.operand1} + {self.operand2} = {self.expected_sum})"


@dataclass
class EncodingResult:
    """Result of input encoding"""
    block_forces: Dict[int, float]  # Block ID -> force magnitude
    encoding_info: Dict[str, any]  # Metadata about encoding

    def __repr__(self) -> str:
        return f"EncodingResult({len(self.block_forces)} forces)"


@dataclass
class DecodingResult:
    """Result of output decoding"""
    decoded_sum: int
    confidence: float  # 0.0-1.0
    bit_values: List[float]  # Raw bit values from output blocks
    decoding_info: Dict[str, any]  # Metadata about decoding

    def __repr__(self) -> str:
        return f"DecodingResult(sum={self.decoded_sum}, confidence={self.confidence:.3f})"


class BinaryEncoder:
    """
    Binary number encoder for Neural-Klotski input blocks.

    Converts integers to binary representation and applies appropriate
    forces to input blocks to represent the binary digits.
    """

    def __init__(self, input_scaling_factor: float = 100.0):
        """
        Initialize binary encoder.

        Args:
            input_scaling_factor: Force scaling factor for input stimulation
        """
        self.input_scaling_factor = input_scaling_factor

    def number_to_binary(self, number: int, num_bits: int) -> List[int]:
        """
        Convert number to binary representation.

        Args:
            number: Integer to convert
            num_bits: Number of bits in representation

        Returns:
            List of binary digits (0 or 1), LSB first
        """
        if number < 0:
            raise ValueError("Cannot encode negative numbers")
        if number >= (1 << num_bits):
            raise ValueError(f"Number {number} exceeds {num_bits}-bit capacity")

        binary_digits = []
        for i in range(num_bits):
            binary_digits.append((number >> i) & 1)

        return binary_digits

    def encode_addition_problem(self, problem: AdditionProblem,
                              input_block_ids: List[int]) -> EncodingResult:
        """
        Encode addition problem into input block forces.

        Args:
            problem: Addition problem to encode
            input_block_ids: List of input block IDs (must be 20 blocks)

        Returns:
            Encoding result with block forces
        """
        if len(input_block_ids) != 20:
            raise ValueError(f"Expected 20 input blocks, got {len(input_block_ids)}")

        # Encode both operands (10 bits each)
        operand1_bits = self.number_to_binary(problem.operand1, 10)
        operand2_bits = self.number_to_binary(problem.operand2, 10)

        # Combine operands
        all_bits = operand1_bits + operand2_bits

        # Create block forces
        block_forces = {}
        active_blocks = 0

        for i, (block_id, bit_value) in enumerate(zip(input_block_ids, all_bits)):
            if bit_value == 1:
                # Apply strong positive force for active bits
                block_forces[block_id] = self.input_scaling_factor
                active_blocks += 1
            else:
                # Apply weak negative force for inactive bits (to ensure silence)
                block_forces[block_id] = -self.input_scaling_factor * 0.1

        encoding_info = {
            'operand1': problem.operand1,
            'operand2': problem.operand2,
            'operand1_bits': operand1_bits,
            'operand2_bits': operand2_bits,
            'active_blocks': active_blocks,
            'total_blocks': len(input_block_ids)
        }

        return EncodingResult(block_forces, encoding_info)

    def get_bit_position_info(self, block_position: int) -> Dict[str, any]:
        """
        Get information about what a block position represents.

        Args:
            block_position: 0-based position in input shelf

        Returns:
            Dictionary with bit position information
        """
        if block_position < 0 or block_position >= 20:
            raise ValueError("Block position must be 0-19")

        if block_position < 10:
            return {
                'operand': 1,
                'bit_position': block_position,
                'bit_value': 2 ** block_position,
                'description': f"Operand 1, bit {block_position} (value {2**block_position})"
            }
        else:
            bit_pos = block_position - 10
            return {
                'operand': 2,
                'bit_position': bit_pos,
                'bit_value': 2 ** bit_pos,
                'description': f"Operand 2, bit {bit_pos} (value {2**bit_pos})"
            }


class BinaryDecoder:
    """
    Binary number decoder for Neural-Klotski output blocks.

    Extracts binary representation from output block positions and
    converts to decimal sum.
    """

    def __init__(self, output_threshold: float = 55.0):
        """
        Initialize binary decoder.

        Args:
            output_threshold: Threshold for considering output active
        """
        self.output_threshold = output_threshold

    def extract_bit_values(self, output_blocks: Dict[int, BlockState]) -> List[float]:
        """
        Extract bit values from output block positions.

        Args:
            output_blocks: Dictionary of output block ID -> BlockState

        Returns:
            List of bit values (19 values for 19 output blocks)
        """
        if len(output_blocks) != 19:
            raise ValueError(f"Expected 19 output blocks, got {len(output_blocks)}")

        # Sort blocks by block ID to ensure consistent ordering
        sorted_blocks = sorted(output_blocks.items())

        bit_values = []
        for block_id, block_state in sorted_blocks:
            # Use block position as analog bit value
            bit_values.append(float(block_state.position))

        return bit_values

    def decode_binary_sum(self, bit_values: List[float]) -> DecodingResult:
        """
        Decode binary sum from bit values.

        Args:
            bit_values: List of 19 bit values from output blocks

        Returns:
            Decoding result with sum and confidence
        """
        if len(bit_values) != 19:
            raise ValueError(f"Expected 19 bit values, got {len(bit_values)}")

        # Convert analog values to binary decisions
        binary_bits = []
        confidence_values = []

        for i, value in enumerate(bit_values):
            # Binary decision based on threshold
            is_active = value > self.output_threshold

            # Confidence based on how far from threshold
            distance_from_threshold = abs(value - self.output_threshold)
            max_distance = 100.0  # Assume max reasonable distance
            confidence = min(1.0, distance_from_threshold / max_distance)

            binary_bits.append(1 if is_active else 0)
            confidence_values.append(confidence)

        # Convert binary to decimal (LSB first)
        decoded_sum = 0
        for i, bit in enumerate(binary_bits):
            if bit == 1:
                decoded_sum += (1 << i)

        # Overall confidence is minimum of all bit confidences
        overall_confidence = min(confidence_values) if confidence_values else 0.0

        decoding_info = {
            'binary_bits': binary_bits,
            'bit_confidences': confidence_values,
            'threshold_used': self.output_threshold,
            'max_possible_sum': (1 << 19) - 1
        }

        return DecodingResult(decoded_sum, overall_confidence, bit_values, decoding_info)

    def get_expected_bit_pattern(self, expected_sum: int) -> List[int]:
        """
        Get expected binary bit pattern for a sum.

        Args:
            expected_sum: Expected sum value

        Returns:
            List of expected binary bits (19 bits, LSB first)
        """
        if expected_sum < 0:
            raise ValueError("Sum must be non-negative")
        if expected_sum >= (1 << 19):
            raise ValueError(f"Sum {expected_sum} exceeds 19-bit capacity")

        binary_bits = []
        for i in range(19):
            binary_bits.append((expected_sum >> i) & 1)

        return binary_bits


class AdditionTaskManager:
    """
    Complete task manager for addition problems.

    Coordinates encoding, network execution, and decoding for
    complete addition task workflow.
    """

    def __init__(self, config: SimulationConfig):
        """
        Initialize addition task manager.

        Args:
            config: Simulation configuration
        """
        self.config = config
        self.encoder = BinaryEncoder(config.input_scaling_factor)
        self.decoder = BinaryDecoder(config.output_threshold)

        # Task statistics
        self.problems_attempted = 0
        self.problems_correct = 0
        self.total_error = 0.0

    def generate_random_problem(self, max_operand: int = 511) -> AdditionProblem:
        """
        Generate random addition problem.

        Args:
            max_operand: Maximum value for each operand (default 511 for 10-bit)

        Returns:
            Random addition problem
        """
        import random

        operand1 = random.randint(0, max_operand)
        operand2 = random.randint(0, max_operand)
        expected_sum = operand1 + operand2

        return AdditionProblem(operand1, operand2, expected_sum)

    def execute_addition_task(self, network: NeuralKlotskiNetwork,
                            problem: AdditionProblem) -> Tuple[DecodingResult, Dict[str, any]]:
        """
        Execute complete addition task on network.

        Args:
            network: Neural-Klotski network
            problem: Addition problem to solve

        Returns:
            Tuple of (decoding_result, task_statistics)
        """
        # Get input and output blocks
        input_blocks = self._get_shelf_blocks(network, "input")
        output_blocks = self._get_shelf_blocks(network, "output")

        if len(input_blocks) != 20:
            raise ValueError(f"Expected 20 input blocks, got {len(input_blocks)}")
        if len(output_blocks) != 19:
            raise ValueError(f"Expected 19 output blocks, got {len(output_blocks)}")

        # Encode problem
        encoding_result = self.encoder.encode_addition_problem(
            problem, list(input_blocks.keys())
        )

        # Apply input forces
        for block_id, force in encoding_result.block_forces.items():
            network.blocks[block_id].add_force(force)

        # Run computation phase
        computation_stats = []
        for timestep in range(self.config.computation_timesteps):
            step_stats = network.execute_timestep()
            computation_stats.append(step_stats)

        # Extract output
        bit_values = self.decoder.extract_bit_values(output_blocks)
        decoding_result = self.decoder.decode_binary_sum(bit_values)

        # Update statistics
        self.problems_attempted += 1
        is_correct = (decoding_result.decoded_sum == problem.expected_sum)
        if is_correct:
            self.problems_correct += 1

        error = abs(decoding_result.decoded_sum - problem.expected_sum)
        self.total_error += error

        task_stats = {
            'problem': problem,
            'encoding_info': encoding_result.encoding_info,
            'computation_timesteps': len(computation_stats),
            'final_firing_rate': computation_stats[-1]['blocks_fired'] if computation_stats else 0,
            'is_correct': is_correct,
            'error_magnitude': error,
            'confidence': decoding_result.confidence
        }

        return decoding_result, task_stats

    def _get_shelf_blocks(self, network: NeuralKlotskiNetwork, shelf_name: str) -> Dict[int, BlockState]:
        """Get blocks from specific shelf based on lag position"""
        shelf_ranges = {
            "input": (45, 55),      # Shelf 1
            "hidden1": (95, 105),   # Shelf 2
            "hidden2": (145, 155),  # Shelf 3
            "output": (195, 205)    # Shelf 4
        }

        if shelf_name not in shelf_ranges:
            raise ValueError(f"Unknown shelf: {shelf_name}")

        min_lag, max_lag = shelf_ranges[shelf_name]
        shelf_blocks = {}

        for block_id, block in network.blocks.items():
            if min_lag <= block.lag_position <= max_lag:
                shelf_blocks[block_id] = block

        return shelf_blocks

    def evaluate_accuracy(self) -> float:
        """Get current accuracy rate"""
        if self.problems_attempted == 0:
            return 0.0
        return self.problems_correct / self.problems_attempted

    def get_task_statistics(self) -> Dict[str, any]:
        """Get comprehensive task statistics"""
        return {
            'problems_attempted': self.problems_attempted,
            'problems_correct': self.problems_correct,
            'accuracy_rate': self.evaluate_accuracy(),
            'average_error': self.total_error / max(1, self.problems_attempted),
            'total_error': self.total_error
        }

    def reset_statistics(self):
        """Reset all task statistics"""
        self.problems_attempted = 0
        self.problems_correct = 0
        self.total_error = 0.0


if __name__ == "__main__":
    # Test encoding/decoding system
    from neural_klotski.config import get_default_config
    from neural_klotski.core.architecture import create_addition_network

    print("Testing Neural-Klotski Encoding/Decoding System...")

    config = get_default_config()

    # Test binary encoding
    encoder = BinaryEncoder(config.simulation.input_scaling_factor)
    problem = AdditionProblem(42, 27, 69)

    print(f"Testing problem: {problem}")

    # Test encoding
    input_block_ids = list(range(1, 21))  # Mock block IDs
    encoding_result = encoder.encode_addition_problem(problem, input_block_ids)

    print(f"Encoding result: {encoding_result}")
    print(f"Active blocks: {encoding_result.encoding_info['active_blocks']}")

    # Test decoding
    decoder = BinaryDecoder(config.simulation.output_threshold)

    # Create mock output with expected sum (69 = 1000101 in binary)
    expected_bits = decoder.get_expected_bit_pattern(69)
    print(f"Expected bits for sum 69: {expected_bits}")

    # Mock bit values (convert binary to positions above/below threshold)
    mock_bit_values = []
    for bit in expected_bits:
        if bit == 1:
            mock_bit_values.append(config.simulation.output_threshold + 20.0)
        else:
            mock_bit_values.append(config.simulation.output_threshold - 20.0)

    decoding_result = decoder.decode_binary_sum(mock_bit_values)
    print(f"Decoding result: {decoding_result}")

    # Test complete network
    print(f"\nTesting with complete network...")
    network = create_addition_network(enable_learning=False)

    task_manager = AdditionTaskManager(config.simulation)
    simple_problem = AdditionProblem(5, 3, 8)

    result, stats = task_manager.execute_addition_task(network, simple_problem)
    print(f"Network result: {result}")
    print(f"Task statistics: {stats}")

    print(f"\nEncoding/decoding system test completed successfully!")