"""
Part 1: Text Generator
----------------------
Uses Google FLAN-T5 (local, via HuggingFace transformers) for answer generation.
Returns generated text along with a confidence proxy based on token scores.
"""

import time
from dataclasses import dataclass
from typing import List, Optional, Dict

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from config import CONFIG


@dataclass
class GenerationResult:
    """Holds the output from the generator."""
    answer: str
    confidence: float        # 0.0–1.0 proxy from sequence score
    generation_time: float   # seconds
    input_tokens: int
    output_tokens: int
    prompt: str


class TextGenerator:
    """FLAN-T5 based text generation with confidence estimation."""

    def __init__(self, config=None):
        self.config = config or CONFIG.model
        self.tokenizer = None
        self.model = None
        self._loaded = False

    def _load_model(self):
        """Lazy-load the model and tokenizer."""
        if self._loaded:
            return

        print(f"[Generator] Loading model: {self.config.generator_model}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.generator_model)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.config.generator_model)
        self.model.eval()

        # Move to device
        device = self.config.generator_device
        if device == "cuda" and torch.cuda.is_available():
            self.model = self.model.to("cuda")
        else:
            self.model = self.model.to("cpu")

        self._loaded = True
        print(f"[Generator] Model loaded on {next(self.model.parameters()).device}")

    def generate(self, query: str, context_chunks: List[str],
                 max_new_tokens: Optional[int] = None,
                 history: Optional[List[Dict[str, str]]] = None) -> GenerationResult:
        """
        Generate an answer given a query and retrieved context chunks.

        Args:
            query: The user's question
            context_chunks: List of relevant text passages
            max_new_tokens: Override for max generation length
            history: Optional list of previous chat messages [{"role": "user"/"ai", "text": "..."}]

        Returns:
            GenerationResult with answer, confidence, and timing info
        """
        self._load_model()

        max_tokens = max_new_tokens or self.config.max_new_tokens

        # Build prompt with history if available
        prompt = self._build_prompt(query, context_chunks, history)

        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=512,
            truncation=True,
        )
        input_tokens = inputs["input_ids"].shape[1]

        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Generate with scores for confidence estimation
        start_time = time.time()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                num_beams=2,
                early_stopping=True,
                return_dict_in_generate=True,
                output_scores=True,
            )
        generation_time = time.time() - start_time

        # Decode
        generated_ids = outputs.sequences[0]
        answer = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        output_tokens = len(generated_ids)

        # Compute confidence proxy from sequence scores
        confidence = self._compute_confidence(outputs)

        return GenerationResult(
            answer=answer,
            confidence=confidence,
            generation_time=generation_time,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            prompt=prompt,
        )

    def _build_prompt(self, query: str, context_chunks: List[str], history: Optional[List[Dict[str, str]]] = None) -> str:
        """Build a structured prompt for FLAN-T5."""
        # Join context with separators
        context = "\n\n".join(context_chunks[:5])  # Limit to avoid token overflow
        
        # Build conversation history string
        history_str = ""
        if history:
            # Only use last 3 turns to avoid token overflow
            recent_history = history[-3:] if len(history) > 3 else history
            for msg in recent_history:
                role = "User" if msg["role"] == "user" else "Assistant"
                history_str += f"{role}: {msg['text']}\n"
            
            history_str = f"Previous Conversation:\n{history_str}\n"

        prompt = (
            f"Answer based solely on the Context. If the Context does not contain the answer, say 'I cannot find the answer in the provided context.'\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            f"Answer:"
        )
        return prompt

    def _compute_confidence(self, outputs) -> float:
        """
        Compute a confidence proxy from generation scores.
        Uses the mean of softmax probabilities of chosen tokens.
        """
        if not hasattr(outputs, "scores") or not outputs.scores:
            return 0.5  # Default if scores unavailable

        try:
            token_probs = []
            generated_ids = outputs.sequences[0]

            # outputs.scores is a tuple of tensors, one per generated token
            for i, score in enumerate(outputs.scores):
                # Get the softmax probability of the chosen token
                probs = torch.softmax(score[0], dim=-1)
                # The token chosen by beam search
                if i + 1 < len(generated_ids):
                    chosen_token_id = generated_ids[i + 1]  # +1 because sequences includes the decoder start
                    token_prob = probs[chosen_token_id].item()
                    token_probs.append(token_prob)

            if token_probs:
                # Mean probability as confidence proxy
                return min(1.0, max(0.0, sum(token_probs) / len(token_probs)))
        except Exception:
            pass

        return 0.5

    def get_stats(self) -> dict:
        """Return generator statistics."""
        return {
            "model": self.config.generator_model,
            "loaded": self._loaded,
            "device": str(next(self.model.parameters()).device) if self._loaded else "not loaded",
            "max_new_tokens": self.config.max_new_tokens,
        }


# ── Quick test ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    gen = TextGenerator()
    result = gen.generate(
        query="What is the Eiffel Tower?",
        context_chunks=[
            "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France.",
            "It is named after the engineer Gustave Eiffel, whose company designed and built the tower.",
            "Constructed from 1887 to 1889, it was initially criticized by some of France's leading artists.",
        ],
    )
    print(f"Answer: {result.answer}")
    print(f"Confidence: {result.confidence:.4f}")
    print(f"Time: {result.generation_time:.2f}s")
