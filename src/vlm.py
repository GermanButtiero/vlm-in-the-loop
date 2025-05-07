import torch 
from transformers import (
    AutoModelForCausalLM,
)

class LocalVLM:
    """Class to handle local or Ovis VLM inference"""
    
    def __init__(self, model_name=None, device="cuda"):
        """
        Initialize the VLM model.
        
        Args:
            model_name: Name of the model to load from HuggingFace or path for Ovis model
            device: Device to run the model on
        """
        print(f"Loading VLM model: {model_name}")
        self.model_name = model_name
        self.device = device
        self.model_type = "Ovis2"
        
        # Check if this is an Ovis2-4B model
        is_ovis2_model = "Ovis2" in model_name
        
        if is_ovis2_model:
            try:
                # Initialize Ovis2-4B directly with transformers
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.bfloat16,
                    multimodal_max_length=32768,
                    trust_remote_code=True, use_flash_attention_2=False
                ).to(device)
                
                # Get tokenizers
                self.text_tokenizer = self.model.get_text_tokenizer()
                self.visual_tokenizer = self.model.get_visual_tokenizer()
                
                self.model_type = "ovis2"
                print("Ovis2 VLM model loaded successfully")
            except Exception as e:
                raise ValueError(f"Failed to load Ovis2 model: {e}. Make sure to install required dependencies: torch==2.4.0 transformers==4.46.2 numpy==1.25.0 pillow==10.3.0 flash-attn==2.7.0.post2")
        

    def evaluate_segmentation(self, image, prompt):
        """
        Evaluate segmentation quality using the VLM.

        Args:
            image: PIL Image with segmentation visualization
            prompt: Prompt to evaluate the segmentation
            
        Returns:
            Boolean indicating approval and the model's response
        """
        try:
            if self.model_type == "ovis2":
                # Format query for Ovis2-4B
                query = f"<image>\n{prompt}"
                
                # Preprocess inputs using Ovis2-4B's specific method
                _, input_ids, pixel_values = self.model.preprocess_inputs(query, [image], max_partition=9)
                attention_mask = torch.ne(input_ids, self.text_tokenizer.pad_token_id)
                input_ids = input_ids.unsqueeze(0).to(device=self.model.device)
                attention_mask = attention_mask.unsqueeze(0).to(device=self.model.device)
                
                if pixel_values is not None:
                    pixel_values = pixel_values.to(
                        dtype=self.visual_tokenizer.dtype, 
                        device=self.visual_tokenizer.device
                    )
                pixel_values = [pixel_values]
                
                # Generate response
                with torch.inference_mode():
                    gen_kwargs = dict(
                        max_new_tokens=256,  # Shorter response for segmentation quality evaluation
                        do_sample=False,
                        eos_token_id=self.model.generation_config.eos_token_id,
                        pad_token_id=self.text_tokenizer.pad_token_id,
                        use_cache=True
                    )
                    output_ids = self.model.generate(
                        input_ids, 
                        pixel_values=pixel_values, 
                        attention_mask=attention_mask, 
                        **gen_kwargs
                    )[0]
                    
                    response = self.text_tokenizer.decode(output_ids, skip_special_tokens=True)
                    # Extract the model's answer
                    if prompt in response:
                        answer = response.split(prompt, 1)[1].strip().lower()
                    else:
                        answer = response.lower()
                    
                    # Check if the answer indicates approval
                    is_approved = True if "yes" in answer else False#in answer and not ("no" in answer and len(answer) < 20)
                
                return is_approved, answer
            else:
                # Handle unsupported model types
                return False, "Unsupported VLM model type"
            
        except Exception as e:
            print(f"Error during VLM inference: {e}")
            return False, f"Error generating response: {e}"
