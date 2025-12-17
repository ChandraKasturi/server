"""TTS service for streaming audio chunks with Kokoro model"""

import struct
import re
import time
from io import BytesIO
from typing import Optional, Generator, AsyncGenerator, List, Dict, Tuple
import asyncio
import concurrent.futures
import tempfile
import os

import av
import numpy as np
import soundfile as sf



# Settings for text chunking
class Settings:
    absolute_max_tokens = 450
    target_max_tokens = 250
    target_min_tokens = 175

settings = Settings()


def get_vocab():
    """Get the vocabulary dictionary mapping characters to token IDs"""
    _pad = "$"
    _punctuation = ';:,.!?¡¿—…"«»"" '
    _letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    _letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"

    # Create vocabulary dictionary
    symbols = [_pad] + list(_punctuation) + list(_letters) + list(_letters_ipa)
    return {symbol: i for i, symbol in enumerate(symbols)}


# Initialize vocabulary
VOCAB = get_vocab()


def tokenize(phonemes: str) -> List[int]:
    """Convert phonemes string to token IDs
    
    Args:
        phonemes: String of phonemes to tokenize
        
    Returns:
        List of token IDs
    """
    return [i for i in map(VOCAB.get, phonemes) if i is not None]


def process_text_chunk(text: str, language: str = "a", skip_phonemize: bool = False) -> List[int]:
    """Process a chunk of text through normalization, phonemization, and tokenization.
    
    Args:
        text: Text chunk to process
        language: Language code for phonemization
        skip_phonemize: If True, treat input as phonemes and skip normalization/phonemization
        
    Returns:
        List of token IDs
    """
    # For simplification, we'll just tokenize the text directly
    # In a full implementation, this would include phonemization
    return tokenize(text)


def get_sentence_info(text: str, custom_phenomes_list: Dict[str, str], lang_code: str = "a") -> List[Tuple[str, List[int], int]]:
    """Process all sentences and return info"""
    # 判断是否为中文
    is_chinese = lang_code.startswith("z") or re.search(r"[\u4e00-\u9fff]", text)
    if is_chinese:
        # 按中文标点断句
        sentences = re.split(r"([，。！？；])+", text)
    else:
        sentences = re.split(r"([.!?;:])(?=\s|$)", text)
    
    phoneme_length, min_value = len(custom_phenomes_list), 0
    
    results = []
    for i in range(0, len(sentences), 2):
        sentence = sentences[i].strip()
        for replaced in range(min_value, phoneme_length):
            current_id = f"</|custom_phonemes_{replaced}|/>"
            if current_id in sentence:
                sentence = sentence.replace(
                    current_id, custom_phenomes_list.pop(current_id)
                )
                min_value += 1
        punct = sentences[i + 1] if i + 1 < len(sentences) else ""
        if not sentence:
            continue
        full = sentence + punct
        tokens = process_text_chunk(full)
        results.append((full, tokens, len(tokens)))
    return results


def smart_split(text: str, max_tokens: int = settings.absolute_max_tokens, lang_code: str = "a") -> List[Tuple[str, List[int]]]:
    """Build optimal chunks targeting 175-250 tokens, never exceeding max_tokens."""
    start_time = time.time()
    chunk_count = 0
    print(f"Starting smart split for {len(text)} chars")
    
    custom_phoneme_list = {}
    
    # Process all sentences
    sentences = get_sentence_info(text, custom_phoneme_list, lang_code=lang_code)
    
    current_chunk = []
    current_tokens = []
    current_count = 0
    chunks = []
    
    for sentence, tokens, count in sentences:
        # Handle sentences that exceed max tokens
        if count > max_tokens:
            # Yield current chunk if any
            if current_chunk:
                chunk_text = " ".join(current_chunk)
                chunk_count += 1
                print(f"Yielding chunk {chunk_count}: '{chunk_text[:50]}{'...' if len(chunk_text) > 50 else ''}' ({current_count} tokens)")
                chunks.append((chunk_text, current_tokens))
                current_chunk = []
                current_tokens = []
                current_count = 0
            
            # Split long sentence on commas
            clauses = re.split(r"([,])", sentence)
            clause_chunk = []
            clause_tokens = []
            clause_count = 0
            
            for j in range(0, len(clauses), 2):
                clause = clauses[j].strip()
                comma = clauses[j + 1] if j + 1 < len(clauses) else ""
                
                if not clause:
                    continue
                
                full_clause = clause + comma
                tokens = process_text_chunk(full_clause)
                count = len(tokens)
                
                # If adding clause keeps us under max and not optimal yet
                if (clause_count + count <= max_tokens and 
                    clause_count + count <= settings.target_max_tokens):
                    clause_chunk.append(full_clause)
                    clause_tokens.extend(tokens)
                    clause_count += count
                else:
                    # Yield clause chunk if we have one
                    if clause_chunk:
                        chunk_text = " ".join(clause_chunk)
                        chunk_count += 1
                        print(f"Yielding clause chunk {chunk_count}: '{chunk_text[:50]}{'...' if len(chunk_text) > 50 else ''}' ({clause_count} tokens)")
                        chunks.append((chunk_text, clause_tokens))
                    clause_chunk = [full_clause]
                    clause_tokens = tokens
                    clause_count = count
            
            # Don't forget last clause chunk
            if clause_chunk:
                chunk_text = " ".join(clause_chunk)
                chunk_count += 1
                print(f"Yielding final clause chunk {chunk_count}: '{chunk_text[:50]}{'...' if len(chunk_text) > 50 else ''}' ({clause_count} tokens)")
                chunks.append((chunk_text, clause_tokens))
        
        # Regular sentence handling
        elif (current_count >= settings.target_min_tokens and 
              current_count + count > settings.target_max_tokens):
            # If we have a good sized chunk and adding next sentence exceeds target,
            # yield current chunk and start new one
            chunk_text = " ".join(current_chunk)
            chunk_count += 1
            print(f"Yielding chunk {chunk_count}: '{chunk_text[:50]}{'...' if len(chunk_text) > 50 else ''}' ({current_count} tokens)")
            chunks.append((chunk_text, current_tokens))
            current_chunk = [sentence]
            current_tokens = tokens
            current_count = count
        elif current_count + count <= settings.target_max_tokens:
            # Keep building chunk while under target max
            current_chunk.append(sentence)
            current_tokens.extend(tokens)
            current_count += count
        elif (current_count + count <= max_tokens and 
              current_count < settings.target_min_tokens):
            # Only exceed target max if we haven't reached minimum size yet
            current_chunk.append(sentence)
            current_tokens.extend(tokens)
            current_count += count
        else:
            # Yield current chunk and start new one
            if current_chunk:
                chunk_text = " ".join(current_chunk)
                chunk_count += 1
                print(f"Yielding chunk {chunk_count}: '{chunk_text[:50]}{'...' if len(chunk_text) > 50 else ''}' ({current_count} tokens)")
                chunks.append((chunk_text, current_tokens))
            current_chunk = [sentence]
            current_tokens = tokens
            current_count = count
    
    # Don't forget the last chunk
    if current_chunk:
        chunk_text = " ".join(current_chunk)
        chunk_count += 1
        print(f"Yielding final chunk {chunk_count}: '{chunk_text[:50]}{'...' if len(chunk_text) > 50 else ''}' ({current_count} tokens)")
        chunks.append((chunk_text, current_tokens))
    
    total_time = time.time() - start_time
    print(f"Split completed in {total_time * 1000:.2f}ms, produced {chunk_count} chunks")
    
    return chunks


class StreamingAudioWriter:
    """Handles streaming audio format conversions"""

    def __init__(self, format: str, sample_rate: int, channels: int = 1):
        self.format = format.lower()
        self.sample_rate = sample_rate
        self.channels = channels
        self.bytes_written = 0
        self.pts = 0

        codec_map = {
            "wav": "pcm_s16le",
            "mp3": "mp3",
            "opus": "libopus",
            "flac": "flac",
            "aac": "aac",
        }
        # Format-specific setup
        if self.format in ["wav", "flac", "mp3", "pcm", "aac", "opus"]:
            if self.format != "pcm":
                self.output_buffer = BytesIO()
                self.container = av.open(
                    self.output_buffer,
                    mode="w",
                    format=self.format if self.format != "aac" else "adts",
                )
                self.stream = self.container.add_stream(
                    codec_map[self.format],
                    sample_rate=self.sample_rate,
                    layout="mono" if self.channels == 1 else "stereo",
                )
                self.stream.bit_rate = 128000
        else:
            raise ValueError(f"Unsupported format: {format}")

    def close(self):
        if hasattr(self, "container"):
            self.container.close()

        if hasattr(self, "output_buffer"):
            self.output_buffer.close()

    def write_chunk(
        self, audio_data: Optional[np.ndarray] = None, finalize: bool = False
    ) -> bytes:
        """Write a chunk of audio data and return bytes in the target format.

        Args:
            audio_data: Audio data to write, or None if finalizing
            finalize: Whether this is the final write to close the stream
        """

        if finalize:
            if self.format != "pcm":
                packets = self.stream.encode(None)
                for packet in packets:
                    self.container.mux(packet)

                data = self.output_buffer.getvalue()
                self.close()
                return data

        if audio_data is None or len(audio_data) == 0:
            return b""

        if self.format == "pcm":
            # Write raw bytes
            return audio_data.tobytes()
        else:
            # Properly reshape audio for mono vs stereo
            if self.channels == 1:
                # For mono, ensure it's (1, samples)
                if len(audio_data.shape) == 1:
                    audio_data = audio_data.reshape(1, -1)
                elif audio_data.shape[0] == 2:
                    # If stereo input but mono output, average the channels
                    audio_data = np.mean(audio_data, axis=0).reshape(1, -1)
            else:
                # For stereo, ensure it's (2, samples)
                if len(audio_data.shape) == 1:
                    # If mono input but stereo output, duplicate the channel
                    audio_data = np.stack([audio_data, audio_data], axis=0)
                elif audio_data.shape[0] == 1:
                    # If (1, samples), duplicate to (2, samples)
                    audio_data = np.vstack([audio_data, audio_data])
                # If already (2, samples), keep as is
            
            frame = av.AudioFrame.from_ndarray(
                audio_data,
                format="s16",
                layout="mono" if self.channels == 1 else "stereo",
            )
            frame.sample_rate = self.sample_rate

            frame.pts = self.pts
            self.pts += frame.samples

            packets = self.stream.encode(frame)
            for packet in packets:
                self.container.mux(packet)

            data = self.output_buffer.getvalue()
            self.output_buffer.seek(0)
            self.output_buffer.truncate(0)
            return data


class TTSService:
    """Service for text-to-speech conversion with streaming support"""
    
    def __init__(self):
        """Initialize the TTS service."""
       # self.pipeline = KPipeline(model=KModel.KOKORO_V2_1_2B, lang_code='a')
        # Thread pool for blocking operations
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        print("TTS service initialized with soundfile for audio transformation")
    
    def _transform_chunk_with_soundfile(self, audio_chunk: bytes) -> bytes:
        """Transform audio chunk using soundfile.
        
        Args:
            audio_chunk: Input audio chunk as bytes (big endian int16)
            
        Returns:
            Transformed audio chunk as bytes (little endian int16)
        """
        try:
            # Create temporary files for input and output
            with tempfile.NamedTemporaryFile(suffix='.raw', delete=False) as temp_input:
                temp_input.write(audio_chunk)
                temp_input_path = temp_input.name
            
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_output:
                temp_output_path = temp_output.name
            
            try:
                # Read raw PCM data with soundfile
                # Input: s16be (big endian), 24000 Hz, mono
                data, samplerate_read = sf.read(
                    temp_input_path,
                    samplerate=24000,
                    channels=1,
                    format='RAW',
                    subtype='PCM_16',  # 16-bit PCM
                    endian='LITTLE'       # Big endian input
                )
                
                # Write as WAV with little endian PCM_16
                sf.write(temp_output_path, data, samplerate_read, subtype='PCM_16')
                
                # Read the generated WAV file as bytes
                with open(temp_output_path, 'rb') as f:
                    wav_data = f.read()
                
                return wav_data
                
            finally:
                # Clean up temporary files
                if os.path.exists(temp_input_path):
                    os.remove(temp_input_path)
                if os.path.exists(temp_output_path):
                    os.remove(temp_output_path)
                
        except Exception as e:
            print(f"Error transforming chunk with soundfile: {str(e)}")
            return self._fallback_transform(audio_chunk)
    
    def _fallback_transform(self, audio_chunk: bytes) -> bytes:
        """Fallback transformation when soundfile processing fails.
        
        Args:
            audio_chunk: Input audio chunk as bytes (big endian int16, mono, 24000 Hz)
            
        Returns:
            Transformed audio chunk as WAV bytes (little endian int16, mono, 24000 Hz)
        """
        try:
            # Convert big endian bytes to numpy array
            audio_numpy = np.frombuffer(audio_chunk, dtype='>i2')  # big endian int16
            
            # Convert to little endian
            audio_numpy_le = audio_numpy.astype('<i2')  # little endian int16
            
            # Create a simple WAV header for mono 24000 Hz 16-bit
            sample_rate = 24000
            channels = 1
            bits_per_sample = 16
            byte_rate = sample_rate * channels * bits_per_sample // 8
            block_align = channels * bits_per_sample // 8
            data_size = len(audio_numpy_le) * 2  # 2 bytes per sample
            
            # WAV header
            wav_header = struct.pack('<4sI4s4sIHHIIHH4sI',
                b'RIFF',
                36 + data_size,  # ChunkSize
                b'WAVE',
                b'fmt ',
                16,  # Subchunk1Size
                1,   # AudioFormat (PCM)
                channels,
                sample_rate,
                byte_rate,
                block_align,
                bits_per_sample,
                b'data',
                data_size
            )
            
            return wav_header + audio_numpy_le.tobytes()
            
        except Exception as e:
            print(f"Error in fallback transformation: {str(e)}")
            return audio_chunk  # Last resort: return original
    
    def _generate_audio_chunks_sync(self, text: str, voice: str = 'af_heart', speed: float = 1.0) -> Generator[bytes, None, None]:
        """Synchronously generate audio chunks from text.
        
        Args:
            text: Text to convert to speech
            voice: Voice to use for TTS
            speed: Speed of speech (0.25 to 4.0)
            
        Yields:
            WAV audio chunks as bytes (after soundfile transformation)
        """
        try:
            # Split text into chunks before TTS processing
            print(f"Splitting text into chunks for TTS processing...")
            text_chunks = smart_split(text, lang_code='a')
            print(f"Text split into {len(text_chunks)} chunks")
            
            chunk_count = 0
            
            # Process each text chunk
            for text_chunk_idx, (chunk_text, chunk_tokens) in enumerate(text_chunks):
                print(f"\nProcessing text chunk {text_chunk_idx + 1}/{len(text_chunks)}")
                print(f"Chunk text: {chunk_text[:100]}{'...' if len(chunk_text) > 100 else ''}")
                print(f"Chunk tokens: {len(chunk_tokens)}")
                
                # Generate TTS audio for this chunk
                generator = self.pipeline(chunk_text, voice=voice, speed=speed)
                
                for result in generator:
                    # Get the numpy audio data
                    audio_numpy = result.audio.numpy()
                    
                    # Keep audio as mono (Kokoro's native output format)
                    if len(audio_numpy.shape) > 1 and audio_numpy.shape[0] > 1:
                        # If somehow we get multi-channel, average to mono
                        audio_numpy = np.mean(audio_numpy, axis=0)
                    elif len(audio_numpy.shape) > 1:
                        # If it's (1, samples), flatten to (samples,)
                        audio_numpy = audio_numpy.flatten()
                    
                    # Ensure the audio is in the correct format
                    # Kokoro outputs float32 audio in range [-1, 1], convert to int16
                    if audio_numpy.dtype == np.float32:
                        # Clamp values to [-1, 1] range and convert to int16
                        audio_numpy = np.clip(audio_numpy, -1.0, 1.0)
                        audio_numpy = (audio_numpy * 32767).astype(np.int16)
                    elif audio_numpy.dtype != np.int16:
                        audio_numpy = audio_numpy.astype(np.int16)
                    
                    # Convert numpy array to big endian format for soundfile input
                    audio_numpy_be = audio_numpy.astype('<i2')  # big endian int16
                    raw_pcm_chunk = audio_numpy_be.tobytes()
                    
                    # Transform the raw PCM chunk using soundfile
                    wav_chunk = self._transform_chunk_with_soundfile(raw_pcm_chunk)
                    
                    chunk_count += 1
                    if wav_chunk:
                        # Save individual chunks for debugging (optional)
                        with open(f"chunk_{chunk_count}.wav", "wb") as f:
                            f.write(wav_chunk)
                        yield wav_chunk
                
        except Exception as e:
            print(f"Error generating audio chunks: {str(e)}")
            raise
    
    async def generate_audio_chunks(self, text: str, voice: str = 'af_heart', speed: float = 1.0) -> AsyncGenerator[bytes, None]:
        """Asynchronously generate audio chunks from text.
        
        Args:
            text: Text to convert to speech
            voice: Voice to use for TTS
            speed: Speed of speech (0.25 to 4.0)
            
        Yields:
            WAV audio chunks as bytes
        """
        # Run the synchronous generator in a thread pool
        loop = asyncio.get_event_loop()
        
        # Create a queue to communicate between the thread and async generator
        chunk_queue = asyncio.Queue()
        
        def _generate_chunks():
            """Thread function to generate chunks and put them in the queue."""
            try:
                for chunk in self._generate_audio_chunks_sync(text, voice, speed):
                    # Put chunk in queue (need to use thread-safe method)
                    asyncio.run_coroutine_threadsafe(chunk_queue.put(chunk), loop)
                # Signal end of chunks
                asyncio.run_coroutine_threadsafe(chunk_queue.put(None), loop)
            except Exception as e:
                # Signal error
                asyncio.run_coroutine_threadsafe(chunk_queue.put(e), loop)
        
        # Start the generation in a separate thread
        future = loop.run_in_executor(self.thread_pool, _generate_chunks)
        
        try:
            while True:
                # Get chunk from queue
                chunk = await chunk_queue.get()
                
                # Check for end signal
                if chunk is None:
                    break
                    
                # Check for error
                if isinstance(chunk, Exception):
                    raise chunk
                
                yield chunk
                
        finally:
            # Ensure the thread completes
            await future
    
    def get_available_voices(self) -> list:
        """Get list of available voices.
        
        Returns:
            List of available voice names
        """
        return [
            'af_bella', 'af_nicole', 'af_sarah', 'af_sky', 'af_heart',
            'bf_emma', 'bf_isabella',
            'am_adam', 'am_michael',
            'bm_george', 'bm_lewis'
        ] 