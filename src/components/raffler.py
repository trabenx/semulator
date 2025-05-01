import random
import logging
from ..core.utils import parse_value

logger = logging.getLogger(__name__)

class Raffler:
    def __init__(self, raffle_config, rng):
        self.config = raffle_config
        self.rng = rng
        self.max_effects = raffle_config.get('max_effects_per_image', 100) # Default high if not set
        self.applied_count = 0

    def raffle_effects(self, category):
        """Selects and parameterizes effects for a given category."""
        selected_effects = []
        if category not in self.config.get('categories', {}):
            logger.warning(f"Raffle category '{category}' not found in config.")
            return []

        available_effects = self.config['categories'][category]

        # Shuffle for random selection order if multiple effects are chosen
        self.rng.shuffle(available_effects)

        for effect_config in available_effects:
            if self.applied_count >= self.max_effects:
                break

            name = effect_config['name']
            probability = effect_config.get('probability', 1.0) # Default to 1 if not specified

            if self.rng.random() < probability:
                # Effect selected, randomize its parameters
                randomized_params = {}
                for param, value in effect_config.get('params', {}).items():
                     # Use the recursive randomize logic or just parse_value?
                     # Parse value should handle ranges/choices directly here
                    randomized_params[param.replace('_range', '').replace('_choices','')] = parse_value(value, self.rng)

                is_enabled = randomized_params.get('enabled', True)
                if is_enabled:
                    effect_instance = {
                        'name': name,
                        'params': randomized_params
                    }
                    selected_effects.append(effect_instance)
                    self.applied_count += 1
                    logger.debug(f"Raffled effect '{name}' in category '{category}' with params: {randomized_params}")

        # Determine application order (optional, default to raffle order)
        # Can add logic here based on config if needed.

        return selected_effects
