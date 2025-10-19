"""
Block Renderer for Neural-Klotski Visualization

Comprehensive block rendering system with state visualization,
interactive features, and performance optimization.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Callable, Set
from enum import Enum
import time
import math
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from neural_klotski.core.block import Block, BlockColor, BlockState
from neural_klotski.visualization.layout.coordinate_system import Point2D, BoundingBox
from neural_klotski.visualization.layout.block_layout import BlockLayoutManager
from neural_klotski.visualization.rendering.visual_elements import (
    VisualElement, CircularBlock, RectangularBlock, HexagonalBlock,
    BlockLabel, BlockIndicator, Color, VisualStyle, RenderState, ShapeType
)


class BlockVisualizationMode(Enum):
    """Block visualization modes"""
    ACTIVATION_LAG = "activation_lag"
    STATE_OVERVIEW = "state_overview"
    CONNECTIVITY = "connectivity"
    LEARNING = "learning"
    DEBUG = "debug"


@dataclass
class BlockRenderConfig:
    """Configuration for block rendering"""
    # Shape and sizing
    shape_type: ShapeType = ShapeType.CIRCLE
    base_size: float = 20.0
    size_scale_factor: float = 1.0
    min_size: float = 10.0
    max_size: float = 50.0

    # Visualization mode
    visualization_mode: BlockVisualizationMode = BlockVisualizationMode.ACTIVATION_LAG

    # Visual features
    show_labels: bool = True
    show_activation_indicators: bool = True
    show_state_indicators: bool = True
    show_connection_indicators: bool = False

    # Animation
    enable_animations: bool = True
    animation_speed: float = 1.0
    firing_animation_enabled: bool = True

    # Color coding
    use_color_coding: bool = True
    color_by_state: bool = True
    color_intensity_by_activation: bool = True

    # Performance
    enable_level_of_detail: bool = True
    label_visibility_threshold: float = 0.5  # Zoom level
    indicator_visibility_threshold: float = 0.8


@dataclass
class BlockInteractionState:
    """State of block interaction"""
    hovered_block_id: Optional[int] = None
    selected_block_ids: Set[int] = field(default_factory=set)
    highlighted_block_ids: Set[int] = field(default_factory=set)

    # Interaction callbacks
    click_callbacks: List[Callable[[int], None]] = field(default_factory=list)
    hover_callbacks: List[Callable[[Optional[int]], None]] = field(default_factory=list)
    selection_callbacks: List[Callable[[Set[int]], None]] = field(default_factory=list)


@dataclass
class BlockVisual:
    """Complete visual representation of a block"""
    block_id: int
    main_element: VisualElement
    label: Optional[BlockLabel] = None
    indicators: List[BlockIndicator] = field(default_factory=list)

    # State tracking
    last_update_time: float = 0.0
    needs_update: bool = True

    # Visual state
    current_activation: float = 0.0
    current_state: BlockState = BlockState.INACTIVE
    visual_size: float = 20.0

    def update_visual_state(self, block: Block, current_time: float) -> bool:
        """
        Update visual state from block data.

        Returns:
            True if visual changed
        """
        changed = False

        # Update activation level
        new_activation = getattr(block, 'position', 0.0) / 100.0  # Normalize to 0-1
        if abs(new_activation - self.current_activation) > 0.01:
            self.current_activation = new_activation
            changed = True

        # Update block state
        if block.state != self.current_state:
            self.current_state = block.state
            changed = True

        # Update visual elements
        if isinstance(self.main_element, CircularBlock):
            self.main_element.set_activation_level(self.current_activation)
        elif isinstance(self.main_element, RectangularBlock):
            self.main_element.set_progress_value(self.current_activation)

        # Update render state based on block state
        if block.state == BlockState.FIRING:
            self.main_element.set_render_state(RenderState.FIRING)
            if isinstance(self.main_element, CircularBlock):
                self.main_element.start_pulse_animation()
        elif block.state == BlockState.REFRACTORY:
            self.main_element.set_render_state(RenderState.REFRACTORY)
        else:
            self.main_element.set_render_state(RenderState.NORMAL)

        if changed:
            self.last_update_time = current_time
            self.needs_update = True

        return changed

    def get_all_elements(self) -> List[VisualElement]:
        """Get all visual elements for this block"""
        elements = [self.main_element]
        if self.label:
            elements.append(self.label)
        elements.extend(self.indicators)
        return elements


class BlockRenderer:
    """
    Main block rendering system.

    Manages visual representation of all blocks in the network,
    handles state updates, interactions, and rendering optimization.
    """

    def __init__(self,
                 layout_manager: BlockLayoutManager,
                 config: Optional[BlockRenderConfig] = None):
        """Initialize block renderer"""
        self.layout_manager = layout_manager
        self.config = config or BlockRenderConfig()

        # Visual storage
        self.block_visuals: Dict[int, BlockVisual] = {}
        self.render_layers: Dict[str, List[VisualElement]] = {
            'shadows': [],
            'blocks': [],
            'labels': [],
            'indicators': [],
            'effects': []
        }

        # Interaction state
        self.interaction_state = BlockInteractionState()

        # Performance tracking
        self.render_stats = {
            'blocks_rendered': 0,
            'elements_rendered': 0,
            'render_time_ms': 0.0,
            'last_render_time': 0.0
        }

        # Color schemes
        self.color_schemes = self._create_default_color_schemes()
        self.current_color_scheme = "default"

    def add_block(self, block: Block) -> None:
        """Add a block to the rendering system"""
        if block.id in self.block_visuals:
            return

        # Get position from layout manager
        position = self.layout_manager.get_block_position(block.id)
        if not position:
            position = Point2D(0, 0)  # Default position

        # Create main visual element
        main_element = self._create_main_element(block, position)

        # Create label if enabled
        label = None
        if self.config.show_labels:
            label_text = str(block.id)
            label_position = Point2D(position.x, position.y + self.config.base_size + 10)
            label = BlockLabel(f"label_{block.id}", label_position, label_text, 10.0)

        # Create indicators
        indicators = []
        if self.config.show_activation_indicators:
            indicator_pos = Point2D(position.x + self.config.base_size/2, position.y - self.config.base_size/2)
            activation_indicator = BlockIndicator(f"activation_{block.id}", indicator_pos, "firing", 6.0)
            indicators.append(activation_indicator)

        if self.config.show_state_indicators:
            indicator_pos = Point2D(position.x - self.config.base_size/2, position.y - self.config.base_size/2)
            state_indicator = BlockIndicator(f"state_{block.id}", indicator_pos, "threshold", 6.0)
            indicators.append(state_indicator)

        # Create block visual
        block_visual = BlockVisual(
            block_id=block.id,
            main_element=main_element,
            label=label,
            indicators=indicators
        )

        # Apply initial styling
        self._apply_block_styling(block_visual, block)

        # Store visual
        self.block_visuals[block.id] = block_visual

    def remove_block(self, block_id: int) -> None:
        """Remove a block from the rendering system"""
        if block_id in self.block_visuals:
            del self.block_visuals[block_id]

    def update_blocks(self, blocks: Dict[int, Block], current_time: Optional[float] = None) -> None:
        """Update all block visuals with current block data"""
        if current_time is None:
            current_time = time.time()

        # Add new blocks
        for block_id, block in blocks.items():
            if block_id not in self.block_visuals:
                self.add_block(block)

        # Update existing blocks
        for block_id, block_visual in self.block_visuals.items():
            if block_id in blocks:
                block = blocks[block_id]

                # Update position from layout manager
                new_position = self.layout_manager.get_block_position(block_id)
                if new_position:
                    block_visual.main_element.set_position(new_position)

                    # Update associated element positions
                    if block_visual.label:
                        label_pos = Point2D(new_position.x, new_position.y + self.config.base_size + 10)
                        block_visual.label.set_position(label_pos)

                    for i, indicator in enumerate(block_visual.indicators):
                        if i == 0:  # Activation indicator
                            indicator_pos = Point2D(new_position.x + self.config.base_size/2, new_position.y - self.config.base_size/2)
                        else:  # State indicator
                            indicator_pos = Point2D(new_position.x - self.config.base_size/2, new_position.y - self.config.base_size/2)
                        indicator.set_position(indicator_pos)

                # Update visual state
                block_visual.update_visual_state(block, current_time)

        # Remove blocks that no longer exist
        blocks_to_remove = []
        for block_id in self.block_visuals.keys():
            if block_id not in blocks:
                blocks_to_remove.append(block_id)

        for block_id in blocks_to_remove:
            self.remove_block(block_id)

        # Update interaction states
        self._update_interaction_states()

    def get_render_elements(self, viewport_bounds: Optional[BoundingBox] = None) -> Dict[str, List[VisualElement]]:
        """
        Get all visual elements organized by render layers.

        Args:
            viewport_bounds: Optional viewport bounds for culling

        Returns:
            Dictionary of render layers with visual elements
        """
        render_start = time.perf_counter()

        # Clear render layers
        for layer in self.render_layers.values():
            layer.clear()

        elements_count = 0
        blocks_rendered = 0

        # Process each block visual
        for block_visual in self.block_visuals.values():
            # Viewport culling
            if viewport_bounds:
                block_bounds = block_visual.main_element.get_bounds()
                if not viewport_bounds.intersects(block_bounds):
                    continue

            blocks_rendered += 1

            # Get all elements for this block
            elements = block_visual.get_all_elements()

            for element in elements:
                if not element.visible:
                    continue

                elements_count += 1

                # Categorize elements by type
                if isinstance(element, (CircularBlock, RectangularBlock, HexagonalBlock)):
                    self.render_layers['blocks'].append(element)
                elif isinstance(element, BlockLabel):
                    self.render_layers['labels'].append(element)
                elif isinstance(element, BlockIndicator):
                    self.render_layers['indicators'].append(element)

        # Sort layers by z-order
        for layer in self.render_layers.values():
            layer.sort(key=lambda x: x.z_order)

        # Update render stats
        render_time = (time.perf_counter() - render_start) * 1000
        self.render_stats.update({
            'blocks_rendered': blocks_rendered,
            'elements_rendered': elements_count,
            'render_time_ms': render_time,
            'last_render_time': time.time()
        })

        return self.render_layers

    def handle_mouse_click(self, screen_position: Point2D) -> Optional[int]:
        """Handle mouse click and return clicked block ID"""
        # Convert screen to world coordinates (would need viewport reference)
        world_position = screen_position  # Simplified for now

        # Find clicked block
        for block_visual in self.block_visuals.values():
            if block_visual.main_element.contains_point(world_position):
                # Update selection
                if block_visual.block_id in self.interaction_state.selected_block_ids:
                    self.interaction_state.selected_block_ids.remove(block_visual.block_id)
                else:
                    self.interaction_state.selected_block_ids.add(block_visual.block_id)

                # Notify callbacks
                for callback in self.interaction_state.click_callbacks:
                    callback(block_visual.block_id)

                # Notify selection callbacks
                for callback in self.interaction_state.selection_callbacks:
                    callback(self.interaction_state.selected_block_ids.copy())

                return block_visual.block_id

        return None

    def handle_mouse_hover(self, screen_position: Point2D) -> Optional[int]:
        """Handle mouse hover and return hovered block ID"""
        world_position = screen_position  # Simplified for now

        new_hovered_id = None

        # Find hovered block
        for block_visual in self.block_visuals.values():
            if block_visual.main_element.contains_point(world_position):
                new_hovered_id = block_visual.block_id
                break

        # Update hover state
        if new_hovered_id != self.interaction_state.hovered_block_id:
            self.interaction_state.hovered_block_id = new_hovered_id

            # Notify callbacks
            for callback in self.interaction_state.hover_callbacks:
                callback(new_hovered_id)

        return new_hovered_id

    def set_visualization_mode(self, mode: BlockVisualizationMode) -> None:
        """Change visualization mode"""
        self.config.visualization_mode = mode

        # Update all block visuals
        for block_visual in self.block_visuals.values():
            self._apply_visualization_mode(block_visual, mode)

    def set_block_highlight(self, block_ids: Set[int], highlight: bool = True) -> None:
        """Set highlight state for specific blocks"""
        if highlight:
            self.interaction_state.highlighted_block_ids.update(block_ids)
        else:
            self.interaction_state.highlighted_block_ids -= block_ids

    def register_click_callback(self, callback: Callable[[int], None]) -> None:
        """Register callback for block clicks"""
        self.interaction_state.click_callbacks.append(callback)

    def register_hover_callback(self, callback: Callable[[Optional[int]], None]) -> None:
        """Register callback for block hover events"""
        self.interaction_state.hover_callbacks.append(callback)

    def register_selection_callback(self, callback: Callable[[Set[int]], None]) -> None:
        """Register callback for selection changes"""
        self.interaction_state.selection_callbacks.append(callback)

    def _create_main_element(self, block: Block, position: Point2D) -> VisualElement:
        """Create the main visual element for a block"""
        size = self.config.base_size * self.config.size_scale_factor

        if self.config.shape_type == ShapeType.CIRCLE:
            element = CircularBlock(f"block_{block.id}", position, size / 2)
        elif self.config.shape_type == ShapeType.RECTANGLE:
            element = RectangularBlock(f"block_{block.id}", position, Point2D(size, size), corner_radius=3.0)
        elif self.config.shape_type == ShapeType.HEXAGON:
            element = HexagonalBlock(f"block_{block.id}", position, size / 2)
        else:
            # Default to circle
            element = CircularBlock(f"block_{block.id}", position, size / 2)

        return element

    def _apply_block_styling(self, block_visual: BlockVisual, block: Block) -> None:
        """Apply styling to block visual based on block properties"""
        # Get base color from block color
        base_color = Color.from_block_color(block.color)

        # Apply color intensity based on activation if enabled
        if self.config.color_intensity_by_activation:
            activation_level = getattr(block, 'position', 0.0) / 100.0
            intensity = 0.3 + 0.7 * activation_level  # 0.3 to 1.0 range
            base_color = Color(base_color.r * intensity, base_color.g * intensity, base_color.b * intensity)

        # Create style
        style = VisualStyle(
            fill_color=base_color,
            border_color=Color(0.2, 0.2, 0.2),
            border_width=2.0,
            shadow_enabled=True
        )

        # Apply style to main element
        block_visual.main_element.style = style

        # Create hover style
        hover_color = base_color.blend_with(Color(1, 1, 1), 0.3)
        hover_style = style.copy()
        hover_style.fill_color = hover_color
        hover_style.border_width = 3.0
        block_visual.main_element.hover_style = hover_style

        # Create selected style
        selected_style = style.copy()
        selected_style.border_color = Color(1, 1, 0)  # Yellow border
        selected_style.border_width = 4.0
        block_visual.main_element.selected_style = selected_style

    def _apply_visualization_mode(self, block_visual: BlockVisual, mode: BlockVisualizationMode) -> None:
        """Apply visualization mode to block visual"""
        if mode == BlockVisualizationMode.ACTIVATION_LAG:
            # Standard activation/lag visualization
            block_visual.main_element.visible = True
            if block_visual.label:
                block_visual.label.visible = True

        elif mode == BlockVisualizationMode.STATE_OVERVIEW:
            # Focus on block states
            for indicator in block_visual.indicators:
                if indicator.indicator_type in ["firing", "threshold"]:
                    indicator.visible = True
                else:
                    indicator.visible = False

        elif mode == BlockVisualizationMode.CONNECTIVITY:
            # Show connection indicators
            for indicator in block_visual.indicators:
                if indicator.indicator_type == "connection":
                    indicator.visible = True

        elif mode == BlockVisualizationMode.DEBUG:
            # Show all visual elements
            block_visual.main_element.visible = True
            if block_visual.label:
                block_visual.label.visible = True
            for indicator in block_visual.indicators:
                indicator.visible = True

    def _update_interaction_states(self) -> None:
        """Update visual states based on interaction"""
        for block_visual in self.block_visuals.values():
            block_id = block_visual.block_id

            # Update render states based on interaction
            if block_id in self.interaction_state.selected_block_ids:
                block_visual.main_element.set_render_state(RenderState.SELECTED)
            elif block_id == self.interaction_state.hovered_block_id:
                block_visual.main_element.set_render_state(RenderState.HIGHLIGHTED)
            elif block_id in self.interaction_state.highlighted_block_ids:
                block_visual.main_element.set_render_state(RenderState.HIGHLIGHTED)
            elif block_visual.current_state == BlockState.FIRING:
                block_visual.main_element.set_render_state(RenderState.FIRING)
            elif block_visual.current_state == BlockState.REFRACTORY:
                block_visual.main_element.set_render_state(RenderState.REFRACTORY)
            else:
                block_visual.main_element.set_render_state(RenderState.NORMAL)

    def _create_default_color_schemes(self) -> Dict[str, Dict[BlockColor, Color]]:
        """Create default color schemes"""
        return {
            "default": {
                BlockColor.RED: Color(0.9, 0.2, 0.2),
                BlockColor.BLUE: Color(0.2, 0.4, 0.9),
                BlockColor.YELLOW: Color(0.9, 0.9, 0.2),
                BlockColor.GREEN: Color(0.2, 0.8, 0.3),
                BlockColor.PURPLE: Color(0.7, 0.2, 0.8)
            },
            "pastel": {
                BlockColor.RED: Color(1.0, 0.6, 0.6),
                BlockColor.BLUE: Color(0.6, 0.7, 1.0),
                BlockColor.YELLOW: Color(1.0, 1.0, 0.6),
                BlockColor.GREEN: Color(0.6, 1.0, 0.7),
                BlockColor.PURPLE: Color(0.9, 0.6, 1.0)
            },
            "high_contrast": {
                BlockColor.RED: Color(1.0, 0.0, 0.0),
                BlockColor.BLUE: Color(0.0, 0.0, 1.0),
                BlockColor.YELLOW: Color(1.0, 1.0, 0.0),
                BlockColor.GREEN: Color(0.0, 1.0, 0.0),
                BlockColor.PURPLE: Color(1.0, 0.0, 1.0)
            }
        }

    def get_render_statistics(self) -> Dict[str, Any]:
        """Get comprehensive rendering statistics"""
        total_elements = sum(len(layer) for layer in self.render_layers.values())

        return {
            'total_blocks': len(self.block_visuals),
            'blocks_rendered': self.render_stats['blocks_rendered'],
            'total_elements': total_elements,
            'elements_rendered': self.render_stats['elements_rendered'],
            'render_time_ms': self.render_stats['render_time_ms'],
            'avg_elements_per_block': total_elements / max(1, len(self.block_visuals)),
            'selected_blocks': len(self.interaction_state.selected_block_ids),
            'highlighted_blocks': len(self.interaction_state.highlighted_block_ids),
            'visualization_mode': self.config.visualization_mode.value,
            'last_render_time': self.render_stats['last_render_time']
        }