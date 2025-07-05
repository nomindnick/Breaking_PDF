"""
Model-specific formatting utilities for LLM prompts.

This module handles the conversion of generic prompts into model-specific formats,
applying the appropriate control tokens and structure for each model family.
"""

from typing import Dict, Optional, Tuple


class ModelFormatter:
    """Handles model-specific prompt formatting."""
    
    # Model family patterns
    PHI_MODELS = ["phi", "phi3", "phi4", "phi-3", "phi-4"]
    GEMMA_MODELS = ["gemma", "gemma2", "gemma3", "gemma-2", "gemma-3"]
    
    @staticmethod
    def detect_model_family(model_name: str) -> str:
        """Detect model family from model name."""
        model_lower = model_name.lower()
        
        for pattern in ModelFormatter.PHI_MODELS:
            if pattern in model_lower:
                return "phi"
        
        for pattern in ModelFormatter.GEMMA_MODELS:
            if pattern in model_lower:
                return "gemma"
        
        # Default to generic
        return "generic"
    
    @staticmethod
    def format_prompt(
        prompt: str,
        model_name: str,
        system_prompt: Optional[str] = None,
        include_assistant_marker: bool = True
    ) -> Tuple[str, Dict[str, any]]:
        """
        Format a prompt for a specific model.
        
        Args:
            prompt: The user prompt
            model_name: The model name (e.g., "phi4-mini:3.8b")
            system_prompt: Optional system prompt
            include_assistant_marker: Whether to include assistant start marker
            
        Returns:
            Tuple of (formatted_prompt, additional_config)
        """
        family = ModelFormatter.detect_model_family(model_name)
        
        if family == "phi":
            return ModelFormatter._format_phi_prompt(
                prompt, system_prompt, include_assistant_marker
            )
        elif family == "gemma":
            return ModelFormatter._format_gemma_prompt(
                prompt, system_prompt, include_assistant_marker
            )
        else:
            # Generic format - just return the prompt as-is
            return prompt, {}
    
    @staticmethod
    def _format_phi_prompt(
        prompt: str,
        system_prompt: Optional[str] = None,
        include_assistant_marker: bool = True
    ) -> Tuple[str, Dict[str, any]]:
        """Format prompt for Phi models."""
        formatted = ""
        
        # Add system prompt if provided
        if system_prompt:
            formatted += f"<|im_start|>system\n{system_prompt}\n<|im_end|>\n"
        
        # Add user prompt
        formatted += f"<|im_start|>user\n{prompt}\n<|im_end|>\n"
        
        # Add assistant marker if requested
        if include_assistant_marker:
            formatted += "<|im_start|>assistant\n"
        
        # Additional config for Phi models
        config = {
            "stop": ["<|im_end|>", "<|im_start|>"],
        }
        
        return formatted, config
    
    @staticmethod
    def _format_gemma_prompt(
        prompt: str,
        system_prompt: Optional[str] = None,
        include_assistant_marker: bool = True
    ) -> Tuple[str, Dict[str, any]]:
        """Format prompt for Gemma models."""
        # Gemma doesn't have a system role, so we prepend system instructions to user turn
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
        else:
            full_prompt = prompt
        
        # Format with Gemma's turn markers
        formatted = f"<start_of_turn>user\n{full_prompt}\n<end_of_turn>\n"
        
        if include_assistant_marker:
            formatted += "<start_of_turn>model\n"
        
        # Additional config for Gemma models
        config = {
            "stop": ["<end_of_turn>", "<start_of_turn>"],
        }
        
        return formatted, config
    
    @staticmethod
    def extract_response(response: str, model_name: str) -> str:
        """Extract the actual response from model output, removing formatting tokens."""
        family = ModelFormatter.detect_model_family(model_name)
        
        if family == "phi":
            # Remove Phi formatting
            response = response.replace("<|im_end|>", "").replace("<|im_start|>", "")
            # Remove any role markers
            for role in ["assistant", "user", "system"]:
                response = response.replace(f"{role}\n", "")
        elif family == "gemma":
            # Remove Gemma formatting
            response = response.replace("<end_of_turn>", "").replace("<start_of_turn>", "")
            # Remove any role markers
            for role in ["model", "user"]:
                response = response.replace(f"{role}\n", "")
        
        return response.strip()


def apply_model_formatting(
    prompt_template: str,
    model_name: str,
    prompt_config: Dict[str, any]
) -> Tuple[str, Dict[str, any]]:
    """
    Apply model-specific formatting to a prompt template.
    
    This is a convenience function that checks if a prompt already has
    model-specific formatting and applies it if not.
    
    Args:
        prompt_template: The prompt template
        model_name: The model name
        prompt_config: The prompt configuration
        
    Returns:
        Tuple of (formatted_prompt, updated_config)
    """
    formatter = ModelFormatter()
    
    # Check if prompt already has model-specific formatting
    if any(marker in prompt_template for marker in [
        "<|im_start|>", "<start_of_turn>", "<|im_end|>", "<end_of_turn>"
    ]):
        # Already formatted, just return as-is
        return prompt_template, prompt_config
    
    # Extract system prompt if embedded in template
    system_prompt = None
    if prompt_template.startswith("You are"):
        # Try to extract first paragraph as system prompt
        lines = prompt_template.split("\n\n", 1)
        if len(lines) == 2:
            system_prompt = lines[0]
            prompt_template = lines[1]
    
    # Apply formatting
    formatted_prompt, format_config = formatter.format_prompt(
        prompt_template,
        model_name,
        system_prompt=system_prompt,
        include_assistant_marker=True
    )
    
    # Merge configs
    updated_config = prompt_config.copy()
    updated_config.update(format_config)
    
    return formatted_prompt, updated_config