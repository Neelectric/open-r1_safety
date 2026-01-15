#!/usr/bin/env python3
"""
VLLM Inference Script with Real-time CLI Streaming

This script sets up a VLLM engine and provides real-time streaming inference
through a command-line interface.
"""

import asyncio
import argparse
import sys
import select
import termios
import tty
from typing import AsyncGenerator, Optional
import signal

try:
    from vllm import LLM, SamplingParams
    from vllm.engine.async_llm_engine import AsyncLLMEngine
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.utils import random_uuid
except ImportError:
    print("Error: VLLM not installed. Install with: pip install vllm")
    sys.exit(1)


class VLLMStreamer:
    def __init__(self, model_name: str, **kwargs):
        """Initialize VLLM engine with the specified model."""
        self.model_name = model_name
        self.engine = None
        self.engine_args = AsyncEngineArgs(
            model=model_name,
            trust_remote_code=True,
            **kwargs
        )
        self.current_request_id = None

    async def initialize(self):
        """Initialize the async VLLM engine."""
        print(f"🚀 Initializing VLLM engine with model: {self.model_name}")
        self.engine = AsyncLLMEngine.from_engine_args(self.engine_args)
        print("✅ Engine initialized successfully!")

    async def apply_chat_template(self, prompt: str) -> tuple[str, str]:
        """Apply the model's chat template to format the prompt."""
        try:
            # Get the tokenizer from the async engine
            tokenizer = await self.engine.get_tokenizer()

            # Format as a chat message
            messages = [{"role": "user", "content": prompt}]

            # Apply chat template if available
            if hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template:
                formatted_prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                return formatted_prompt, prompt
            else:
                # Fallback to raw prompt if no chat template
                return prompt, prompt
        except Exception as e:
            print(f"⚠️  Warning: Could not apply chat template ({e}), using raw prompt")
            return prompt, prompt

    async def abort_request(self, request_id: str):
        """Abort an ongoing request."""
        if self.engine and request_id:
            try:
                await self.engine.abort(request_id)
            except Exception as e:
                # Silently handle abort errors as the request might have already completed
                pass

    async def stream_generate(
        self,
        prompt: str,
        max_tokens: int = 127000,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stream: bool = True,
        interrupt_event: Optional[asyncio.Event] = None
    ) -> AsyncGenerator[str, None]:
        """Generate streaming response for a given prompt."""
        if not self.engine:
            raise RuntimeError("Engine not initialized. Call initialize() first.")

        # Apply chat template to the prompt
        formatted_prompt, user_input = await self.apply_chat_template(prompt)

        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p
        )

        request_id = random_uuid()
        self.current_request_id = request_id

        # Add the request to the engine
        results_generator = self.engine.generate(
            formatted_prompt, sampling_params, request_id
        )

        # Stream the results
        prev_text = ""
        try:
            async for request_output in results_generator:
                # Check if we should interrupt
                if interrupt_event and interrupt_event.is_set():
                    await self.abort_request(request_id)
                    return

                if request_output.outputs:
                    output = request_output.outputs[0]
                    current_text = output.text
                    # Yield only the new part of the text
                    new_text = current_text[len(prev_text):]
                    if new_text:
                        # Make special tokens visible in the output
                        visible_text = new_text
                        # Common special tokens across different model families
                        special_tokens = [
                            '<|endoftext|>', '<|user|>', '<|assistant|>', '<|system|>',
                            '<|im_start|>', '<|im_end|>', '<s>', '</s>', '<unk>',
                            '<pad>', '<|begin_of_text|>', '<|end_of_text|>',
                            '<|start_header_id|>', '<|end_header_id|>',
                            '<|eot_id|>', '<|reserved_special_token_'
                        ]
                        # Ensure special tokens are visible (they should already be in the text)
                        yield visible_text
                    prev_text = current_text
        except asyncio.CancelledError:
            await self.abort_request(request_id)
            raise
        finally:
            self.current_request_id = None

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 127000,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> str:
        """Generate a complete response (non-streaming)."""
        full_response = ""
        async for chunk in self.stream_generate(
            prompt, max_tokens, temperature, top_p, stream=True
        ):
            full_response += chunk
        return full_response


class KeyboardListener:
    """Non-blocking keyboard input listener."""

    def __init__(self):
        self.old_settings = None
        self.buffer = ""

    def __enter__(self):
        """Set terminal to raw mode."""
        if sys.stdin.isatty():
            self.old_settings = termios.tcgetattr(sys.stdin)
            tty.setraw(sys.stdin.fileno())
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Restore terminal settings."""
        if self.old_settings is not None:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)

    def check_keypress(self) -> Optional[str]:
        """Check if any key has been pressed (non-blocking)."""
        if sys.stdin.isatty():
            if select.select([sys.stdin], [], [], 0)[0]:
                char = sys.stdin.read(1)
                # Handle escape sequences
                if char == '\x1b':  # ESC
                    # Read the rest of the escape sequence
                    if select.select([sys.stdin], [], [], 0)[0]:
                        sys.stdin.read(2)  # Skip [A, [B, etc.
                    return '\x1b'
                return char
        return None


class CLIInterface:
    def __init__(self, streamer: VLLMStreamer):
        self.streamer = streamer
        self.running = True
        self.interrupt_event = asyncio.Event()
        self.pending_input = ""

    async def interactive_mode(self):
        """Run interactive CLI mode with interrupt support."""
        print("\n🤖 VLLM Interactive Chat")
        print("Type 'quit', 'exit', or 'q' to exit")
        print("Type 'clear' to clear the screen")
        print("Press ANY KEY during generation to interrupt and start a new prompt")
        print("=" * 50)

        while self.running:
            try:
                # Get user input
                if self.pending_input:
                    # Use pending input from interrupt
                    prompt = input(f"\n👤 You: {self.pending_input}").strip()
                    prompt = self.pending_input + prompt
                    self.pending_input = ""
                else:
                    prompt = input("\n👤 You: ").strip()

                if not prompt:
                    continue

                if prompt.lower() in ['quit', 'exit', 'q']:
                    break

                if prompt.lower() == 'clear':
                    print("\033[2J\033[H", end="")
                    continue

                # Reset interrupt event
                self.interrupt_event.clear()

                # Generate and stream response
                print("\n🤖 Assistant: ", end="", flush=True)

                # Show the full formatted prompt first
                formatted_prompt, user_input = await self.streamer.apply_chat_template(prompt)

                # Display the formatted prompt (includes system prompt and special tokens)
                print(f"\n📝 Full prompt sent to model:\n{formatted_prompt}", flush=True)
                print("\n🤖 Response: ", end="", flush=True)

                # Stream with keyboard interrupt detection
                interrupted, captured_input = await self._stream_with_interrupt(prompt)

                if interrupted:
                    print("\n\n⚠️  Generation interrupted!", flush=True)
                    self.pending_input = captured_input
                else:
                    print()  # New line after response

            except EOFError:
                break
            except KeyboardInterrupt:
                # Ctrl+C exits completely
                print("\n\n🛑 Exiting...")
                break
            except Exception as e:
                print(f"\n❌ Error during generation: {e}")

        print("\n👋 Goodbye!")

    async def _stream_with_interrupt(self, prompt: str) -> tuple[bool, str]:
        """
        Stream response while checking for keyboard interrupts.
        Returns (was_interrupted, captured_input).
        """
        interrupted = False
        captured_input = ""

        # Create streaming task
        stream_task = asyncio.create_task(self._stream_chunks(prompt))

        # Monitor for keyboard input
        loop = asyncio.get_event_loop()
        listener = KeyboardListener()

        try:
            # Enter raw terminal mode to detect keypresses
            listener.__enter__()

            while not stream_task.done():
                # Check for keypress (non-blocking)
                await asyncio.sleep(0.05)  # Small delay to prevent CPU spinning

                char = await loop.run_in_executor(None, listener.check_keypress)

                if char is not None:
                    # Key was pressed - interrupt!
                    interrupted = True
                    self.interrupt_event.set()

                    # Cancel the streaming task
                    stream_task.cancel()
                    try:
                        await stream_task
                    except asyncio.CancelledError:
                        pass

                    # Exit raw mode before capturing more input
                    listener.__exit__(None, None, None)

                    # Capture the rest of the input
                    # Convert raw characters to proper input
                    if char == '\r' or char == '\n':
                        # Enter key - just interrupt, no captured input
                        captured_input = ""
                    else:
                        captured_input = char
                        # Read remaining characters quickly
                        for _ in range(100):  # Arbitrary limit
                            await asyncio.sleep(0.01)
                            next_char = await loop.run_in_executor(None, listener.check_keypress)
                            if next_char and next_char not in ('\r', '\n'):
                                captured_input += next_char
                            else:
                                break

                    # Re-enter normal mode is already done above
                    return interrupted, captured_input

            # Wait for stream to complete if not interrupted
            if not interrupted:
                await stream_task

        except Exception as e:
            print(f"\n⚠️  Error during streaming: {e}")
            stream_task.cancel()
            try:
                await stream_task
            except asyncio.CancelledError:
                pass
        finally:
            # Ensure we exit raw mode
            listener.__exit__(None, None, None)

        return interrupted, captured_input

    async def _stream_chunks(self, prompt: str):
        """Stream the response chunks."""
        try:
            async for chunk in self.streamer.stream_generate(
                prompt,
                interrupt_event=self.interrupt_event
            ):
                # In raw mode, we need \r\n for proper line breaks
                formatted_chunk = chunk.replace('\n', '\r\n')
                sys.stdout.write(formatted_chunk)
                sys.stdout.flush()
        except asyncio.CancelledError:
            raise

    async def single_prompt_mode(self, prompt: str, stream: bool = True):
        """Process a single prompt and exit."""

        # Show the full formatted prompt first
        formatted_prompt, user_input = await self.streamer.apply_chat_template(prompt)

        # Display the formatted prompt (includes system prompt and special tokens)
        print(f"📝 Full prompt sent to model:\n{formatted_prompt}\n")
        print("🤖 Response: ", end="", flush=True)

        if stream:
            async for chunk in self.streamer.stream_generate(prompt):
                print(chunk, end="", flush=True)
            print()
        else:
            response = await self.streamer.generate(prompt)
            print(response)


async def main():
    parser = argparse.ArgumentParser(description="VLLM Inference with CLI Streaming")
    parser.add_argument("--model", "-m", required=True, help="HuggingFace model name or path")
    parser.add_argument("--revision", "-r", help="HuggingFace revision name")
    parser.add_argument("--prompt", "-p", help="Single prompt to process (non-interactive)")
    parser.add_argument("--max-tokens", type=int, default=8192, help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p sampling parameter")
    parser.add_argument("--no-stream", action="store_true", help="Disable streaming output")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9, help="GPU memory utilization")
    parser.add_argument("--max-model-len", type=int, default=None, help="Maximum model sequence length")

    args = parser.parse_args()

    # Initialize VLLM streamer
    kwargs = {
        'tensor_parallel_size': args.tensor_parallel_size,
        'gpu_memory_utilization': args.gpu_memory_utilization,
        'revision': getattr(args, "revision", "main")
    }

    if args.max_model_len is not None:
        kwargs['max_model_len'] = args.max_model_len

    streamer = VLLMStreamer(
        model_name=args.model,
        **kwargs
    )

    try:
        # Initialize the engine
        await streamer.initialize()

        # Create CLI interface
        cli = CLIInterface(streamer)

        # Run in single prompt or interactive mode
        if args.prompt:
            await cli.single_prompt_mode(
                args.prompt,
                stream=not args.no_stream
            )
        else:
            await cli.interactive_mode()

    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())