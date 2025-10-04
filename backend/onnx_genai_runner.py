from typing import Optional, Dict
import onnxruntime_genai as og

class ONNXGenAIRunner:
    def __init__(self, model_path: str, execution_provider: str = "cpu",
                 default_max_length: int = 2048, default_temperature: float = 1.0):
        cfg = og.Config(model_path)
        if execution_provider != "follow_config":
            cfg.clear_providers()
            if execution_provider != "cpu":
                cfg.append_provider(execution_provider)
        self.model = og.Model(cfg)
        self.tokenizer = og.Tokenizer(self.model)
        self.tokenizer_stream = self.tokenizer.create_stream()
        self.default_max_length = default_max_length
        self.default_temperature = default_temperature

    def _prepare(self, text: str):
        msg = [{"role": "user", "content": text}]
        import json
        prompt = self.tokenizer.apply_chat_template(json.dumps(msg),
                                                    add_generation_prompt=True)
        return self.tokenizer.encode(prompt)

    def generate(self, text: str, max_length: int = None, temperature: float = None,
                 do_sample: bool = None, top_k: int = None, top_p: float = None,
                 repetition_penalty: float = None) -> str:
        if not text:
            raise ValueError("Input text cannot be empty")
        input_tokens = self._prepare(text)
        params = og.GeneratorParams(self.model)
        search = {}
        search['max_length'] = max_length or self.default_max_length
        search['temperature'] = temperature or self.default_temperature
        if do_sample is not None: search['do_sample'] = do_sample
        if top_k is not None: search['top_k'] = top_k
        if top_p is not None: search['top_p'] = top_p
        if repetition_penalty is not None: search['repetition_penalty'] = repetition_penalty
        params.set_search_options(**search)
        gen = og.Generator(self.model, params)
        gen.append_tokens(input_tokens)
        pieces = []
        while not gen.is_done():
            gen.generate_next_token()
            toks = gen.get_next_tokens()
            if not toks:
                continue
            pieces.append(self.tokenizer_stream.decode(toks[0]))
        return "".join(pieces)
